"""Internal messaging — channels, threads, direct messages and message history.

Provides the real-time messaging backbone for the Omniagent platform.
Messages are organised into channels (public, private, group) and direct
conversations. Any message can spawn a threaded reply chain. Read receipts
and typing indicators are tracked per-member.

Design:

* :class:`MessagePriority` — urgency ranking for individual messages.
* :class:`MessageType` — content type (text, system, file, etc.).
* :class:`ChannelType` — visibility/structure of a channel.
* :class:`Message` — a single chat message (may be a thread root or reply).
* :class:`Channel` — a named conversation space with members.
* :class:`DirectConversation` — a 1:1 or small-group ad-hoc conversation.
* :class:`MessagingService` — async manager that owns channels, DMs,
  message history and membership. Thread-safe within one event loop.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("myagent.communication.messaging")


class MessagingError(Exception):
    """Raised when a messaging operation is invalid (unknown channel, etc.)."""


class MessagePriority(str, Enum):  # noqa: UP042 - match existing codebase style
    """Urgency ranking for a single message."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MessageType(str, Enum):  # noqa: UP042
    """Content type of a message."""

    TEXT = "text"
    SYSTEM = "system"
    FILE = "file"
    IMAGE = "image"
    ANNOUNCEMENT = "announcement"
    CODE = "code"
    POLL = "poll"


class ChannelType(str, Enum):  # noqa: UP042
    """Visibility and structure of a channel."""

    PUBLIC = "public"
    PRIVATE = "private"
    DIRECT = "direct"
    GROUP = "group"
    ANNOUNCEMENT = "announcement"


class MessageStatus(str, Enum):  # noqa: UP042
    """Delivery/read lifecycle of a message for a given member."""

    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single chat message.

    Attributes:
        id: Unique message identifier (UUID4 hex).
        channel_id: Channel or DM conversation this message belongs to.
        author: User identifier of the sender.
        content: Message body text.
        message_type: Content type.
        priority: Urgency ranking.
        thread_root_id: If this message is a reply, the ID of the thread's
            root message. ``None`` for top-level messages.
        parent_id: Immediate parent message ID in a thread (may equal
            ``thread_root_id`` for first-level replies).
        attachments: List of attachment metadata dicts (name, url, size, etc.).
        mentions: List of user IDs mentioned in the message.
        reactions: Mapping of emoji -> list of user IDs who reacted.
        created_at: UTC creation timestamp.
        edited_at: UTC timestamp of the last edit, or ``None``.
        deleted: Soft-delete flag (content is retained for audit but hidden).
        metadata: Arbitrary extra key/value pairs.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    channel_id: str
    author: str
    content: str
    message_type: MessageType = MessageType.TEXT
    priority: MessagePriority = MessagePriority.NORMAL
    thread_root_id: str | None = None
    parent_id: str | None = None
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    mentions: list[str] = Field(default_factory=list)
    reactions: dict[str, list[str]] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    edited_at: datetime | None = None
    deleted: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_thread_reply(self) -> bool:
        """True when this message is a reply inside a thread."""

        return self.thread_root_id is not None

    def add_reaction(self, emoji: str, user_id: str) -> None:
        """Record a reaction by *user_id* with *emoji* (idempotent)."""

        users = self.reactions.setdefault(emoji, [])
        if user_id not in users:
            users.append(user_id)

    def remove_reaction(self, emoji: str, user_id: str) -> None:
        """Remove a reaction by *user_id* with *emoji* (no-op if absent)."""

        users = self.reactions.get(emoji, [])
        if user_id in users:
            users.remove(user_id)
            if not users:
                self.reactions.pop(emoji, None)


class Channel(BaseModel):
    """A named conversation space with a member roster.

    Attributes:
        id: Unique channel identifier.
        name: Human-readable channel name.
        channel_type: Visibility/structure.
        members: Set of user IDs who have joined.
        admins: Subset of members with management privileges.
        topic: Optional channel description / topic.
        created_by: User ID of the creator.
        created_at: UTC creation timestamp.
        archived: Whether the channel is read-only / hidden.
        last_message_at: UTC timestamp of the most recent message.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    channel_type: ChannelType = ChannelType.PUBLIC
    members: set[str] = Field(default_factory=set)
    admins: set[str] = Field(default_factory=set)
    topic: str = ""
    created_by: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    archived: bool = False
    last_message_at: datetime | None = None

    def is_member(self, user_id: str) -> bool:
        """True when *user_id* is a member of this channel."""

        return user_id in self.members

    def is_admin(self, user_id: str) -> bool:
        """True when *user_id* is an admin of this channel."""

        return user_id in self.admins


class DirectConversation(BaseModel):
    """A 1:1 or small-group ad-hoc conversation (not a named channel).

    Attributes:
        id: Unique conversation identifier.
        participants: Set of user IDs in the conversation (typically 2).
        created_at: UTC creation timestamp.
        last_message_at: UTC timestamp of the most recent message.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    participants: set[str] = Field(default_factory=set)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_message_at: datetime | None = None


# ---------------------------------------------------------------------------
# Messaging service
# ---------------------------------------------------------------------------


class MessagingService:
    """Async manager for channels, direct messages, threads and history.

    All public methods are coroutines and safe for concurrent use within
    a single asyncio event loop.

    Example::

        service = MessagingService()
        general = await service.create_channel("general", created_by="alice")
        await service.join_channel(general.id, "bob")
        msg = await service.send_message(
            channel_id=general.id, author="alice", content="Hello team!"
        )
        reply = await service.reply_in_thread(
            parent_id=msg.id, author="bob", content="Hi Alice!"
        )
        history = await service.get_history(general.id, limit=50)
    """

    def __init__(self) -> None:
        self._channels: dict[str, Channel] = {}
        self._dm_conversations: dict[str, DirectConversation] = {}
        self._messages: dict[str, Message] = {}
        # channel_id / dm_id -> list of message IDs (ordered)
        self._channel_messages: dict[str, list[str]] = defaultdict(list)
        # message_id -> list of reply message IDs (ordered)
        self._thread_replies: dict[str, list[str]] = defaultdict(list)
        # user_id -> set of channel_ids
        self._user_channels: dict[str, set[str]] = defaultdict(set)
        # (user_a, user_b) sorted tuple -> dm conversation id
        self._dm_index: dict[tuple[str, ...], str] = {}
        # user_id -> {message_id: MessageStatus} read tracking
        self._read_state: dict[str, dict[str, MessageStatus]] = defaultdict(dict)
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Channel management
    # ------------------------------------------------------------------

    async def create_channel(
        self,
        name: str,
        *,
        channel_type: ChannelType = ChannelType.PUBLIC,
        created_by: str = "",
        members: set[str] | None = None,
        admins: set[str] | None = None,
        topic: str = "",
    ) -> Channel:
        """Create and register a new channel. Returns the channel."""

        channel = Channel(
            name=name,
            channel_type=channel_type,
            created_by=created_by,
            members=members or set(),
            admins=admins or ({created_by} if created_by else set()),
            topic=topic,
        )
        async with self._lock:
            self._channels[channel.id] = channel
            for member_id in channel.members:
                self._user_channels[member_id].add(channel.id)
        logger.info("Channel created: %s (%s) by %s", channel.name, channel.id, created_by)
        return channel

    async def get_channel(self, channel_id: str) -> Channel | None:
        """Return the channel by ID, or ``None``."""

        async with self._lock:
            return self._channels.get(channel_id)

    async def list_channels(
        self, *, user_id: str | None = None, include_archived: bool = False
    ) -> list[Channel]:
        """List channels, optionally filtered by membership and archive state."""

        async with self._lock:
            channels = list(self._channels.values())
        if not include_archived:
            channels = [c for c in channels if not c.archived]
        if user_id is not None:
            channels = [c for c in channels if c.is_member(user_id)]
        return channels

    async def archive_channel(self, channel_id: str) -> Channel:
        """Mark a channel as archived (read-only). Returns the channel."""

        async with self._lock:
            channel = self._channels.get(channel_id)
            if channel is None:
                raise MessagingError(f"Channel not found: {channel_id}")
            channel.archived = True
            return channel

    async def join_channel(self, channel_id: str, user_id: str) -> Channel:
        """Add *user_id* to a channel's member roster."""

        async with self._lock:
            channel = self._channels.get(channel_id)
            if channel is None:
                raise MessagingError(f"Channel not found: {channel_id}")
            if channel.archived:
                raise MessagingError(f"Cannot join archived channel: {channel_id}")
            channel.members.add(user_id)
            self._user_channels[user_id].add(channel_id)
            return channel

    async def leave_channel(self, channel_id: str, user_id: str) -> Channel:
        """Remove *user_id* from a channel's member roster."""

        async with self._lock:
            channel = self._channels.get(channel_id)
            if channel is None:
                raise MessagingError(f"Channel not found: {channel_id}")
            channel.members.discard(user_id)
            channel.admins.discard(user_id)
            self._user_channels[user_id].discard(channel_id)
            return channel

    async def update_topic(self, channel_id: str, topic: str) -> Channel:
        """Set the channel topic/description."""

        async with self._lock:
            channel = self._channels.get(channel_id)
            if channel is None:
                raise MessagingError(f"Channel not found: {channel_id}")
            channel.topic = topic
            return channel

    # ------------------------------------------------------------------
    # Direct messages
    # ------------------------------------------------------------------

    async def get_or_create_dm(self, participants: set[str]) -> DirectConversation:
        """Get or create a direct conversation for a set of participants.

        For 2-person DMs, the conversation is keyed by the sorted tuple of
        participant IDs so that the same pair always resolves to the same
        conversation regardless of call order.
        """

        if len(participants) < 2:
            raise MessagingError("A direct conversation requires at least 2 participants")
        key = tuple(sorted(participants))
        async with self._lock:
            conv_id = self._dm_index.get(key)
            if conv_id is not None:
                return self._dm_conversations[conv_id]
            conv = DirectConversation(participants=set(participants))
            self._dm_conversations[conv.id] = conv
            self._dm_index[key] = conv.id
            self._channel_messages[conv.id] = []
            return conv

    async def get_dm(self, conversation_id: str) -> DirectConversation | None:
        """Return a direct conversation by ID, or ``None``."""

        async with self._lock:
            return self._dm_conversations.get(conversation_id)

    async def list_dms(self, user_id: str) -> list[DirectConversation]:
        """List all direct conversations involving *user_id*."""

        async with self._lock:
            return [
                conv for conv in self._dm_conversations.values() if user_id in conv.participants
            ]

    # ------------------------------------------------------------------
    # Message sending
    # ------------------------------------------------------------------

    async def send_message(
        self,
        *,
        channel_id: str,
        author: str,
        content: str,
        message_type: MessageType = MessageType.TEXT,
        priority: MessagePriority = MessagePriority.NORMAL,
        attachments: list[dict[str, Any]] | None = None,
        mentions: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Send a top-level message to a channel or DM conversation.

        Raises :class:`MessagingError` if the channel does not exist or is
        archived, or if the author is not a member (for non-public channels).
        """

        async with self._lock:
            target = self._resolve_container(channel_id)
            if target is None:
                raise MessagingError(f"Channel or conversation not found: {channel_id}")
            if isinstance(target, Channel):
                if target.archived:
                    raise MessagingError(f"Cannot send to archived channel: {channel_id}")
                if target.channel_type != ChannelType.PUBLIC and not target.is_member(author):
                    raise MessagingError(f"User {author} is not a member of channel {channel_id}")
            message = Message(
                channel_id=channel_id,
                author=author,
                content=content,
                message_type=message_type,
                priority=priority,
                attachments=attachments or [],
                mentions=mentions or [],
                metadata=metadata or {},
            )
            self._messages[message.id] = message
            self._channel_messages[channel_id].append(message.id)
            now = datetime.now(UTC)
            if isinstance(target, (Channel, DirectConversation)):
                target.last_message_at = now
            logger.debug("Message %s sent to %s by %s", message.id, channel_id, author)
            return message

    async def reply_in_thread(
        self,
        *,
        parent_id: str,
        author: str,
        content: str,
        message_type: MessageType = MessageType.TEXT,
        priority: MessagePriority = MessagePriority.NORMAL,
        attachments: list[dict[str, Any]] | None = None,
        mentions: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Send a reply that starts or continues a thread under *parent_id*.

        The first reply establishes the thread root as *parent_id*.
        Subsequent replies link to the same root.
        """

        async with self._lock:
            parent = self._messages.get(parent_id)
            if parent is None:
                raise MessagingError(f"Parent message not found: {parent_id}")
            thread_root_id = parent.thread_root_id or parent.id
            reply = Message(
                channel_id=parent.channel_id,
                author=author,
                content=content,
                message_type=message_type,
                priority=priority,
                thread_root_id=thread_root_id,
                parent_id=parent_id,
                attachments=attachments or [],
                mentions=mentions or [],
                metadata=metadata or {},
            )
            self._messages[reply.id] = reply
            self._channel_messages[parent.channel_id].append(reply.id)
            self._thread_replies[thread_root_id].append(reply.id)
            logger.debug("Thread reply %s under %s by %s", reply.id, thread_root_id, author)
            return reply

    async def edit_message(self, message_id: str, new_content: str) -> Message:
        """Edit the content of a message (sets ``edited_at``)."""

        async with self._lock:
            message = self._messages.get(message_id)
            if message is None:
                raise MessagingError(f"Message not found: {message_id}")
            message.content = new_content
            message.edited_at = datetime.now(UTC)
            return message

    async def delete_message(self, message_id: str) -> Message:
        """Soft-delete a message (sets ``deleted=True``, content retained)."""

        async with self._lock:
            message = self._messages.get(message_id)
            if message is None:
                raise MessagingError(f"Message not found: {message_id}")
            message.deleted = True
            return message

    async def react(self, message_id: str, emoji: str, user_id: str) -> Message:
        """Add a reaction to a message."""

        async with self._lock:
            message = self._messages.get(message_id)
            if message is None:
                raise MessagingError(f"Message not found: {message_id}")
            message.add_reaction(emoji, user_id)
            return message

    async def unreact(self, message_id: str, emoji: str, user_id: str) -> Message:
        """Remove a reaction from a message."""

        async with self._lock:
            message = self._messages.get(message_id)
            if message is None:
                raise MessagingError(f"Message not found: {message_id}")
            message.remove_reaction(emoji, user_id)
            return message

    # ------------------------------------------------------------------
    # Message retrieval
    # ------------------------------------------------------------------

    async def get_message(self, message_id: str) -> Message | None:
        """Return a single message by ID, or ``None``."""

        async with self._lock:
            return self._messages.get(message_id)

    async def get_history(
        self,
        channel_id: str,
        *,
        limit: int = 50,
        before: datetime | None = None,
        include_deleted: bool = False,
    ) -> list[Message]:
        """Return messages in a channel/DM, newest-first, up to *limit*.

        ``before`` filters to messages created before the given timestamp
        (for pagination). Deleted messages are excluded unless
        ``include_deleted`` is True.
        """

        async with self._lock:
            ids = list(self._channel_messages.get(channel_id, []))
        messages: list[Message] = []
        for mid in reversed(ids):
            msg = self._messages.get(mid)
            if msg is None:
                continue
            if not include_deleted and msg.deleted:
                continue
            if before is not None and msg.created_at >= before:
                continue
            messages.append(msg)
            if len(messages) >= limit:
                break
        return messages

    async def get_thread(
        self,
        thread_root_id: str,
        *,
        include_deleted: bool = False,
    ) -> list[Message]:
        """Return all replies in a thread, oldest-first.

        The root message itself is **not** included; use
        :meth:`get_message` to fetch it separately.
        """

        async with self._lock:
            ids = list(self._thread_replies.get(thread_root_id, []))
            root = self._messages.get(thread_root_id)
            if root is None and not ids:
                raise MessagingError(f"Thread root not found: {thread_root_id}")
        messages: list[Message] = []
        for mid in ids:
            msg = self._messages.get(mid)
            if msg is None:
                continue
            if not include_deleted and msg.deleted:
                continue
            messages.append(msg)
        return messages

    async def search_messages(
        self,
        query: str,
        *,
        channel_id: str | None = None,
        author: str | None = None,
        limit: int = 100,
    ) -> list[Message]:
        """Full-text substring search across messages."""

        needle = query.lower()
        async with self._lock:
            candidates: list[Message] = []
            if channel_id is not None:
                ids = self._channel_messages.get(channel_id, [])
                candidates = [self._messages[mid] for mid in ids if mid in self._messages]
            else:
                candidates = list(self._messages.values())
        results: list[Message] = []
        for msg in candidates:
            if msg.deleted:
                continue
            if author is not None and msg.author != author:
                continue
            if needle in msg.content.lower():
                results.append(msg)
                if len(results) >= limit:
                    break
        return results

    # ------------------------------------------------------------------
    # Read receipts
    # ------------------------------------------------------------------

    async def mark_read(self, message_id: str, user_id: str) -> None:
        """Mark a message as read by *user_id*."""

        async with self._lock:
            self._read_state[user_id][message_id] = MessageStatus.READ

    async def get_unread_count(self, channel_id: str, user_id: str) -> int:
        """Count messages in *channel_id* not yet read by *user_id*."""

        async with self._lock:
            ids = self._channel_messages.get(channel_id, [])
            read_map = self._read_state.get(user_id, {})
            count = 0
            for mid in ids:
                msg = self._messages.get(mid)
                if msg is None or msg.deleted or msg.author == user_id:
                    continue
                if read_map.get(mid) is not MessageStatus.READ:
                    count += 1
            return count

    # ------------------------------------------------------------------
    # User helpers
    # ------------------------------------------------------------------

    async def get_user_channels(self, user_id: str) -> list[Channel]:
        """Return all channels *user_id* is a member of."""

        async with self._lock:
            channel_ids = self._user_channels.get(user_id, set())
            return [
                self._channels[cid]
                for cid in channel_ids
                if cid in self._channels and not self._channels[cid].archived
            ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_container(self, container_id: str) -> Channel | DirectConversation | None:
        """Look up a container by ID across channels and DMs (no lock)."""

        channel = self._channels.get(container_id)
        if channel is not None:
            return channel
        return self._dm_conversations.get(container_id)

    @property
    def channel_count(self) -> int:
        """Total number of registered channels."""

        return len(self._channels)

    @property
    def message_count(self) -> int:
        """Total number of messages (including soft-deleted)."""

        return len(self._messages)


__all__ = [
    "Channel",
    "ChannelType",
    "DirectConversation",
    "Message",
    "MessagePriority",
    "MessageStatus",
    "MessageType",
    "MessagingError",
    "MessagingService",
]
