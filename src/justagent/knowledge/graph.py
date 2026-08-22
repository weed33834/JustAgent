"""Knowledge graph with entity extraction and relation mapping.

Builds an in-memory directed graph of entities (people, organisations,
locations, dates, concepts) and the relations between them. Entity
extraction uses rule-based patterns (regex for emails, URLs, dates,
phone numbers, capitalised phrases) by default. When a
:class:`~justagent.adapters.model_gateway.ModelGateway` is provided,
LLM-assisted extraction can produce richer, semantically-typed entities
and relations.

Design:

* :class:`EntityType` — enum of common entity categories (extensible
  via the ``CUSTOM`` value for user-defined types).
* :class:`Entity` — a named node in the graph.
* :class:`Relation` — a typed, weighted edge between two entities.
* :class:`KnowledgeGraph` — the graph itself. Supports adding /
  removing entities and relations, querying by entity or relation type,
  finding neighbours, and extracting entities/relations from text.
* :func:`extract_entities` — rule-based entity extraction from text.
* :func:`extract_relations` — rule-based relation extraction using
  co-occurrence and verb patterns.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import defaultdict
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from justagent.utils import now

logger = logging.getLogger("justagent.knowledge")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EntityType(str, Enum):  # noqa: UP042
    """Common entity categories.

    The ``CUSTOM`` value is a catch-all for user-defined entity types
    that do not fit the built-in categories. The actual type string is
    stored in :attr:`Entity.entity_type`.
    """

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    MONEY = "money"
    CONCEPT = "concept"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Entity(BaseModel):
    """A named entity (node) in the knowledge graph.

    Attributes:
        id: Unique entity identifier.
        name: Canonical name of the entity.
        entity_type: Type string (see :class:`EntityType` or a custom
            string).
        aliases: Alternative names / surface forms.
        metadata: Arbitrary key-value metadata.
        source_documents: IDs of documents this entity was extracted from.
        created_at: Unix timestamp of creation.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    entity_type: str = EntityType.CONCEPT.value
    aliases: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_documents: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=now)


class Relation(BaseModel):
    """A typed, weighted relation (edge) between two entities.

    Attributes:
        id: Unique relation identifier.
        source_entity_id: ID of the source entity.
        target_entity_id: ID of the target entity.
        relation_type: Type of relation (e.g. ``works_at``, ``located_in``).
        weight: Confidence or frequency weight in ``[0, 1]``.
        metadata: Arbitrary key-value metadata.
        source_documents: IDs of documents this relation was found in.
        created_at: Unix timestamp of creation.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    source_entity_id: str
    target_entity_id: str
    relation_type: str = "related_to"
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_documents: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=now)


# ---------------------------------------------------------------------------
# Rule-based extraction patterns
# ---------------------------------------------------------------------------

# Email pattern.
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)

# URL pattern.
_URL_RE = re.compile(
    r"\bhttps?://[^\s<>\"]+|www\.[A-Za-z0-9.-]+\.[A-Za-z]{2,}[^\s<>\"]*"
)

# ISO date (2024-01-15, 2024/01/15).
_DATE_ISO_RE = re.compile(
    r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b"
)

# Written date (January 15, 2024 / Jan 15 2024 / 15 January 2024).
_DATE_WRITTEN_RE = re.compile(
    r"\b("
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|"
    r"Nov|Dec)"
    r"\s+\d{1,2},?\s+\d{4}"
    r"|"
    r"\d{1,2}\s+"
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|"
    r"Nov|Dec)"
    r"\s+\d{4}"
    r")\b",
    re.IGNORECASE,
)

# Phone number (US/international-ish).
_PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"
)

# Money ($100, $1,000.00, 100 USD).
_MONEY_RE = re.compile(
    r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d+)?\s(?:USD|EUR|GBP|CNY|JPY)\b",
    re.IGNORECASE,
)

# Capitalised phrase (potential person / organisation name).
# Matches sequences of 1-4 capitalised words, optionally with conjunctions.
_CAPITALIZED_RE = re.compile(
    r"\b(?:[A-Z][a-z]+(?:[-\s][A-Z][a-z]+){0,3})\b"
)

# Common stop words to filter out from capitalised-phrase matches.
_CAPITALIZED_STOPWORDS = frozenset({
    "The", "This", "That", "These", "Those", "A", "An",
    "In", "On", "At", "To", "For", "Of", "With", "By", "From",
    "And", "Or", "But", "Not", "If", "Then", "Else", "When",
    "While", "Is", "Are", "Was", "Were", "Be", "Been", "Being",
    "Have", "Has", "Had", "Do", "Does", "Did", "Will", "Would",
    "Could", "Should", "May", "Might", "Must", "Can", "Shall",
    "It", "He", "She", "We", "They", "You", "I",
    "There", "Here", "Now", "Today", "Tomorrow", "Yesterday",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday",
    "January", "February", "March", "April", "June",
    "July", "August", "September", "October", "November", "December",
    "Some", "Any", "All", "Each", "Every", "No", "None",
    "Which", "What", "Who", "Whom", "Whose", "How", "Why",
    "However", "Therefore", "Moreover", "Furthermore", "Nevertheless",
    "Meanwhile", "Finally", "First", "Second", "Third", "Last",
})

# Relation patterns: (regex, relation_type, source_group, target_group).
# These patterns look for verb-based relations between two capitalised
# phrases in the same sentence.
_RELATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "X works at Y" / "X worked at Y" / "X is working at Y"
    (
        re.compile(
            r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+"
            r"(?:works?|worked|is\s+working|was\s+working)\s+at\s+"
            r"([A-Z][A-Za-z0-9&\s.]+?)(?:[.,;]|\s+(?:in|as|on|for)\s)",
            re.IGNORECASE,
        ),
        "works_at",
    ),
    # "X is located in Y" / "X is based in Y"
    (
        re.compile(
            r"([A-Z][A-Za-z0-9\s.]+?)\s+"
            r"(?:is\s+)?(?:located|based|headquartered)\s+in\s+"
            r"([A-Z][A-Za-z\s.]+?)(?:[.,;]|$)",
            re.IGNORECASE,
        ),
        "located_in",
    ),
    # "X is a Y" / "X is an Y"
    (
        re.compile(
            r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+is\s+an?\s+"
            r"([a-z]+(?:\s[a-z]+)?)",
            re.IGNORECASE,
        ),
        "is_a",
    ),
    # "X was founded by Y"
    (
        re.compile(
            r"([A-Z][A-Za-z0-9&\s.]+?)\s+was\s+founded\s+by\s+"
            r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
            re.IGNORECASE,
        ),
        "founded_by",
    ),
    # "X is the CEO of Y" / "X is the CTO of Y" etc.
    (
        re.compile(
            r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+is\s+the\s+\w+\s+of\s+"
            r"([A-Z][A-Za-z0-9&\s.]+?)(?:[.,;]|$)",
            re.IGNORECASE,
        ),
        "executive_of",
    ),
    # "X is part of Y" / "X belongs to Y"
    (
        re.compile(
            r"([A-Z][A-Za-z0-9\s.]+?)\s+"
            r"(?:is\s+part\s+of|belongs\s+to|is\s+a\s+subsidiary\s+of)\s+"
            r"([A-Z][A-Za-z0-9&\s.]+?)(?:[.,;]|$)",
            re.IGNORECASE,
        ),
        "part_of",
    ),
]


# ---------------------------------------------------------------------------
# Rule-based extraction functions
# ---------------------------------------------------------------------------


def extract_entities(
    text: str,
    *,
    document_id: str = "",
    include_capitalized: bool = True,
) -> list[Entity]:
    """Extract entities from ``text`` using rule-based patterns.

    Extracts emails, URLs, dates, phone numbers, money amounts, and
    capitalised phrases (potential people / organisations). Each unique
    surface form becomes an :class:`Entity`.

    Args:
        text: The text to extract from.
        document_id: Optional document ID to record as the source.
        include_capitalized: If True, extract capitalised phrases as
            concept/person entities. Set to False for a stricter
            extraction (emails, dates, etc. only).

    Returns:
        List of :class:`Entity` objects (deduplicated by name+type).
    """
    entities: list[Entity] = []
    seen: set[tuple[str, str]] = set()

    def _add(name: str, entity_type: str, **meta: Any) -> None:
        key = (name.lower(), entity_type)
        if key in seen:
            return
        seen.add(key)
        source_docs = [document_id] if document_id else []
        entities.append(
            Entity(
                name=name,
                entity_type=entity_type,
                metadata=meta or {},
                source_documents=source_docs,
            )
        )

    # Emails.
    for match in _EMAIL_RE.finditer(text):
        _add(match.group(), EntityType.EMAIL.value)

    # URLs.
    for match in _URL_RE.finditer(text):
        url = match.group().rstrip(".,;)")
        _add(url, EntityType.URL.value)

    # ISO dates.
    for match in _DATE_ISO_RE.finditer(text):
        _add(match.group(), EntityType.DATE.value)

    # Written dates.
    for match in _DATE_WRITTEN_RE.finditer(text):
        _add(match.group().strip(), EntityType.DATE.value)

    # Money.
    for match in _MONEY_RE.finditer(text):
        _add(match.group().strip(), EntityType.MONEY.value)

    # Phone numbers (only if they look like real phone numbers — 7+ digits).
    for match in _PHONE_RE.finditer(text):
        phone = match.group().strip()
        digit_count = sum(c.isdigit() for c in phone)
        if digit_count >= 7:
            _add(phone, EntityType.PHONE.value)

    # Capitalised phrases (potential names / organisations).
    if include_capitalized:
        for match in _CAPITALIZED_RE.finditer(text):
            name = match.group().strip()
            # Skip single-word stop words.
            if name in _CAPITALIZED_STOPWORDS:
                continue
            # Skip if it's a single common word.
            if " " not in name and name in _CAPITALIZED_STOPWORDS:
                continue
            # Heuristic: multi-word capitalised phrases are more likely
            # to be organisations; single capitalised words could be
            # people or concepts.
            if " " in name:
                _add(name, EntityType.ORGANIZATION.value)
            else:
                _add(name, EntityType.PERSON.value)

    return entities


def extract_relations(
    text: str,
    entities: list[Entity] | None = None,
    *,
    document_id: str = "",
) -> list[Relation]:
    """Extract relations from ``text`` using verb-based patterns.

    Scans for common sentence patterns (e.g. "X works at Y",
    "X is located in Y") and creates :class:`Relation` objects. The
    source and target entity IDs are left empty (the caller should
    resolve them against the graph). When ``entities`` is provided,
    co-occurrence-based ``related_to`` relations are also generated
    for entities that appear in the same sentence.

    Args:
        text: The text to extract from.
        entities: Optional list of pre-extracted entities for
            co-occurrence relation generation.
        document_id: Optional document ID to record as the source.

    Returns:
        List of :class:`Relation` objects. Entity IDs are populated
        when ``entities`` is provided and the entity name matches.
    """
    relations: list[Relation] = []
    source_docs = [document_id] if document_id else []

    # Build a name-to-entity lookup.
    entity_lookup: dict[str, Entity] = {}
    if entities:
        for ent in entities:
            entity_lookup[ent.name.lower()] = ent
            for alias in ent.aliases:
                entity_lookup[alias.lower()] = ent

    def _resolve(name: str) -> Entity | None:
        return entity_lookup.get(name.lower().strip())

    # Pattern-based relation extraction.
    for pattern, rel_type in _RELATION_PATTERNS:
        for match in pattern.finditer(text):
            source_name = match.group(1).strip()
            target_name = match.group(2).strip()
            source_ent = _resolve(source_name)
            target_ent = _resolve(target_name)
            relations.append(
                Relation(
                    source_entity_id=source_ent.id if source_ent else "",
                    target_entity_id=target_ent.id if target_ent else "",
                    relation_type=rel_type,
                    weight=0.8,
                    metadata={
                        "source_name": source_name,
                        "target_name": target_name,
                    },
                    source_documents=source_docs,
                )
            )

    # Co-occurrence-based relations: entities appearing in the same
    # sentence are connected with a ``related_to`` relation.
    if entities and len(entities) >= 2:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            found: list[Entity] = []
            for ent in entities:
                if ent.name.lower() in sentence.lower():
                    found.append(ent)
            # Create pairwise relations for all entities in this sentence.
            for i in range(len(found)):
                for j in range(i + 1, len(found)):
                    if found[i].id == found[j].id:
                        continue
                    relations.append(
                        Relation(
                            source_entity_id=found[i].id,
                            target_entity_id=found[j].id,
                            relation_type="related_to",
                            weight=0.3,
                            source_documents=source_docs,
                        )
                    )

    return relations


# ---------------------------------------------------------------------------
# Knowledge graph
# ---------------------------------------------------------------------------


class KnowledgeGraph:
    """An in-memory directed knowledge graph of entities and relations.

    Supports adding / removing entities and relations, querying by
    entity type or name, querying relations by type or entity, finding
    neighbours, and extracting entities/relations from text.

    The graph maintains several indexes for efficient lookup:

    * ``_entities`` — ``entity_id -> Entity``.
    * ``_relations`` — ``relation_id -> Relation``.
    * ``_name_index`` — ``lowercase_name -> set(entity_ids)``.
    * ``_type_index`` — ``entity_type -> set(entity_ids)``.
    * ``_outgoing`` — ``entity_id -> set(relation_ids)`` (outgoing edges).
    * ``_incoming`` — ``entity_id -> set(relation_ids)`` (incoming edges).
    * ``_relation_type_index`` — ``relation_type -> set(relation_ids)``.

    Example::

        >>> graph = KnowledgeGraph()
        >>> alice = graph.add_entity(name="Alice", entity_type="person")
        >>> acme = graph.add_entity(name="Acme Corp", entity_type="organization")
        >>> graph.add_relation(alice.id, acme.id, "works_at")
        >>> neighbors = graph.neighbors(alice.id)
        >>> acme.id in {n.id for n in neighbors}
        True
    """

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._relations: dict[str, Relation] = {}
        self._name_index: dict[str, set[str]] = defaultdict(set)
        self._type_index: dict[str, set[str]] = defaultdict(set)
        self._outgoing: dict[str, set[str]] = defaultdict(set)
        self._incoming: dict[str, set[str]] = defaultdict(set)
        self._relation_type_index: dict[str, set[str]] = defaultdict(set)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def entity_count(self) -> int:
        """Number of entities in the graph."""
        return len(self._entities)

    @property
    def relation_count(self) -> int:
        """Number of relations in the graph."""
        return len(self._relations)

    def __len__(self) -> int:
        return len(self._entities)

    def __contains__(self, entity_id: str) -> bool:
        return entity_id in self._entities

    def __iter__(self) -> Iterator[Entity]:
        return iter(self._entities.values())

    # ------------------------------------------------------------------
    # Entity management
    # ------------------------------------------------------------------

    def add_entity(
        self,
        *,
        name: str,
        entity_type: str = EntityType.CONCEPT.value,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        source_documents: list[str] | None = None,
        entity_id: str | None = None,
    ) -> Entity:
        """Add an entity to the graph.

        If an entity with the same name (case-insensitive) and type
        already exists, it is returned (and aliases/metadata are merged)
        instead of creating a duplicate.

        Args:
            name: Canonical entity name.
            entity_type: Entity type string.
            aliases: Alternative names.
            metadata: Key-value metadata.
            source_documents: Document IDs this entity was found in.
            entity_id: Optional explicit entity ID.

        Returns:
            The created or existing :class:`Entity`.
        """
        # Check for existing entity by name+type.
        existing_id = self._find_entity(name, entity_type)
        if existing_id is not None:
            entity = self._entities[existing_id]
            if aliases:
                for alias in aliases:
                    if alias not in entity.aliases:
                        entity.aliases.append(alias)
                        self._name_index[alias.lower()].add(entity.id)
            if metadata:
                entity.metadata.update(metadata)
            if source_documents:
                for doc_id in source_documents:
                    if doc_id not in entity.source_documents:
                        entity.source_documents.append(doc_id)
            return entity

        entity = Entity(
            id=entity_id or uuid.uuid4().hex,
            name=name,
            entity_type=entity_type,
            aliases=list(aliases) if aliases else [],
            metadata=dict(metadata) if metadata else {},
            source_documents=list(source_documents) if source_documents else [],
        )
        self._entities[entity.id] = entity
        self._name_index[name.lower()].add(entity.id)
        for alias in entity.aliases:
            self._name_index[alias.lower()].add(entity.id)
        self._type_index[entity_type].add(entity.id)
        logger.debug("Added entity %s (%s: %s)", entity.id, entity_type, name)
        return entity

    def get_entity(self, entity_id: str) -> Entity | None:
        """Return an entity by ID, or None if not found."""
        return self._entities.get(entity_id)

    def find_entity(self, name: str, entity_type: str | None = None) -> Entity | None:
        """Find an entity by name (case-insensitive).

        If ``entity_type`` is given, only matches of that type are
        considered.
        """
        eid = self._find_entity(name, entity_type)
        return self._entities[eid] if eid else None

    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity and all its relations.

        Returns True if the entity was removed.
        """
        entity = self._entities.pop(entity_id, None)
        if entity is None:
            return False
        # Remove from indexes.
        self._name_index[entity.name.lower()].discard(entity_id)
        for alias in entity.aliases:
            self._name_index[alias.lower()].discard(entity_id)
        self._type_index[entity.entity_type].discard(entity_id)
        # Remove all relations involving this entity.
        rel_ids = self._outgoing[entity_id] | self._incoming[entity_id]
        for rid in rel_ids:
            self._remove_relation_internal(rid)
        self._outgoing.pop(entity_id, None)
        self._incoming.pop(entity_id, None)
        logger.debug("Removed entity %s and %d relations", entity_id, len(rel_ids))
        return True

    def query_entities(
        self,
        *,
        entity_type: str | None = None,
        name_contains: str | None = None,
        limit: int | None = None,
    ) -> list[Entity]:
        """Query entities by type and/or name substring.

        Args:
            entity_type: Filter by entity type. None = all types.
            name_contains: Case-insensitive substring filter on entity
                name or aliases. None = no name filter.
            limit: Maximum number of results.

        Returns:
            List of matching :class:`Entity` objects.
        """
        if entity_type is not None:
            candidates = [
                self._entities[eid]
                for eid in self._type_index.get(entity_type, set())
                if eid in self._entities
            ]
        else:
            candidates = list(self._entities.values())

        if name_contains is not None:
            needle = name_contains.lower()
            candidates = [
                e for e in candidates
                if needle in e.name.lower()
                or any(needle in a.lower() for a in e.aliases)
            ]

        result = sorted(candidates, key=lambda e: e.name.lower())
        if limit is not None:
            result = result[:limit]
        return result

    def list_entities(self) -> list[Entity]:
        """Return all entities in the graph."""
        return list(self._entities.values())

    # ------------------------------------------------------------------
    # Relation management
    # ------------------------------------------------------------------

    def add_relation(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relation_type: str = "related_to",
        *,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
        source_documents: list[str] | None = None,
        relation_id: str | None = None,
    ) -> Relation:
        """Add a directed relation between two entities.

        If an identical relation (same source, target, and type) already
        exists, its weight is incremented and metadata/documents are
        merged instead of creating a duplicate.

        Args:
            source_entity_id: ID of the source entity.
            target_entity_id: ID of the target entity.
            relation_type: Type of the relation.
            weight: Confidence / frequency weight.
            metadata: Key-value metadata.
            source_documents: Document IDs this relation was found in.
            relation_id: Optional explicit relation ID.

        Returns:
            The created or updated :class:`Relation`.

        Raises:
            KeyError: If either entity ID does not exist in the graph.
        """
        if source_entity_id not in self._entities:
            raise KeyError(f"Source entity not found: {source_entity_id}")
        if target_entity_id not in self._entities:
            raise KeyError(f"Target entity not found: {target_entity_id}")

        # Check for existing relation.
        for rid in self._outgoing[source_entity_id]:
            rel = self._relations.get(rid)
            if (
                rel is not None
                and rel.target_entity_id == target_entity_id
                and rel.relation_type == relation_type
            ):
                # Merge.
                rel.weight = min(1.0, rel.weight + weight * 0.3)
                if metadata:
                    rel.metadata.update(metadata)
                if source_documents:
                    for doc_id in source_documents:
                        if doc_id not in rel.source_documents:
                            rel.source_documents.append(doc_id)
                return rel

        relation = Relation(
            id=relation_id or uuid.uuid4().hex,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relation_type=relation_type,
            weight=weight,
            metadata=dict(metadata) if metadata else {},
            source_documents=list(source_documents) if source_documents else [],
        )
        self._relations[relation.id] = relation
        self._outgoing[source_entity_id].add(relation.id)
        self._incoming[target_entity_id].add(relation.id)
        self._relation_type_index[relation_type].add(relation.id)
        logger.debug(
            "Added relation %s: %s --%s--> %s",
            relation.id,
            source_entity_id,
            relation_type,
            target_entity_id,
        )
        return relation

    def get_relation(self, relation_id: str) -> Relation | None:
        """Return a relation by ID, or None if not found."""
        return self._relations.get(relation_id)

    def remove_relation(self, relation_id: str) -> bool:
        """Remove a relation by ID. Returns True if removed."""
        return self._remove_relation_internal(relation_id)

    def query_relations(
        self,
        *,
        relation_type: str | None = None,
        entity_id: str | None = None,
        direction: str = "both",
        min_weight: float = 0.0,
        limit: int | None = None,
    ) -> list[Relation]:
        """Query relations by type and/or entity.

        Args:
            relation_type: Filter by relation type. None = all types.
            entity_id: Filter to relations involving this entity.
            direction: ``"outgoing"``, ``"incoming"``, or ``"both"``.
                Only used when ``entity_id`` is provided.
            min_weight: Minimum weight threshold.
            limit: Maximum number of results.

        Returns:
            List of matching :class:`Relation` objects.
        """
        if direction not in ("both", "outgoing", "incoming"):
            raise ValueError(f"Invalid direction: {direction}")

        # Work with relation IDs (strings are hashable; Pydantic models are not).
        if relation_type is not None:
            candidate_ids = set(self._relation_type_index.get(relation_type, set()))
        else:
            candidate_ids = set(self._relations.keys())

        if entity_id is not None:
            filtered_ids: set[str] = set()
            if direction in ("outgoing", "both"):
                filtered_ids |= self._outgoing.get(entity_id, set())
            if direction in ("incoming", "both"):
                filtered_ids |= self._incoming.get(entity_id, set())
            candidate_ids &= filtered_ids

        result = [
            self._relations[rid]
            for rid in candidate_ids
            if rid in self._relations
            and self._relations[rid].weight >= min_weight
        ]
        result.sort(key=lambda r: r.weight, reverse=True)
        if limit is not None:
            result = result[:limit]
        return result

    def list_relations(self) -> list[Relation]:
        """Return all relations in the graph."""
        return list(self._relations.values())

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def neighbors(
        self,
        entity_id: str,
        *,
        direction: str = "both",
        relation_type: str | None = None,
    ) -> list[Entity]:
        """Return the neighbour entities of ``entity_id``.

        Args:
            entity_id: The entity to find neighbours for.
            direction: ``"outgoing"``, ``"incoming"``, or ``"both"``.
            relation_type: Optional filter on relation type.

        Returns:
            List of neighbour :class:`Entity` objects (deduplicated).
        """
        if direction not in ("both", "outgoing", "incoming"):
            raise ValueError(f"Invalid direction: {direction}")

        neighbour_ids: set[str] = set()

        if direction in ("outgoing", "both"):
            for rid in self._outgoing.get(entity_id, set()):
                rel = self._relations.get(rid)
                if rel is None:
                    continue
                if relation_type is not None and rel.relation_type != relation_type:
                    continue
                neighbour_ids.add(rel.target_entity_id)

        if direction in ("incoming", "both"):
            for rid in self._incoming.get(entity_id, set()):
                rel = self._relations.get(rid)
                if rel is None:
                    continue
                if relation_type is not None and rel.relation_type != relation_type:
                    continue
                neighbour_ids.add(rel.source_entity_id)

        return [
            self._entities[nid]
            for nid in neighbour_ids
            if nid in self._entities
        ]

    def degree(self, entity_id: str, *, direction: str = "both") -> int:
        """Return the degree (number of connections) of an entity."""
        if direction not in ("both", "outgoing", "incoming"):
            raise ValueError(f"Invalid direction: {direction}")
        count = 0
        if direction in ("outgoing", "both"):
            count += len(self._outgoing.get(entity_id, set()))
        if direction in ("incoming", "both"):
            count += len(self._incoming.get(entity_id, set()))
        return count

    def shortest_path(
        self, source_id: str, target_id: str, max_depth: int = 5
    ) -> list[str] | None:
        """Find the shortest path between two entities (BFS).

        Returns a list of entity IDs from source to target, or None if
        no path exists within ``max_depth`` hops.
        """
        if source_id not in self._entities or target_id not in self._entities:
            return None
        if source_id == target_id:
            return [source_id]

        from collections import deque

        queue: deque[tuple[str, list[str]]] = deque()
        queue.append((source_id, [source_id]))
        visited: set[str] = {source_id}

        while queue:
            current, path = queue.popleft()
            if len(path) - 1 >= max_depth:
                continue
            for neighbour in self.neighbors(current):
                if neighbour.id in visited:
                    continue
                new_path = path + [neighbour.id]
                if neighbour.id == target_id:
                    return new_path
                visited.add(neighbour.id)
                queue.append((neighbour.id, new_path))
        return None

    # ------------------------------------------------------------------
    # Text extraction & ingestion
    # ------------------------------------------------------------------

    def extract_from_text(
        self,
        text: str,
        *,
        document_id: str = "",
        include_capitalized: bool = True,
    ) -> tuple[list[Entity], list[Relation]]:
        """Extract entities and relations from text and add them to the graph.

        This is a convenience method that calls :func:`extract_entities`
        and :func:`extract_relations`, adds all entities to the graph
        (deduplicating), resolves entity IDs in relations, and adds
        all relations to the graph.

        Args:
            text: The text to extract from.
            document_id: Optional source document ID.
            include_capitalized: If True, include capitalised-phrase
                entities.

        Returns:
            A tuple of ``(entities, relations)`` that were added (or
            merged) to the graph.
        """
        raw_entities = extract_entities(
            text,
            document_id=document_id,
            include_capitalized=include_capitalized,
        )
        # Add entities to the graph (deduplicates by name+type).
        added_entities: list[Entity] = []
        for raw in raw_entities:
            entity = self.add_entity(
                name=raw.name,
                entity_type=raw.entity_type,
                aliases=raw.aliases,
                metadata=raw.metadata,
                source_documents=raw.source_documents,
            )
            added_entities.append(entity)

        # Extract relations using the resolved entities.
        raw_relations = extract_relations(
            text,
            entities=added_entities,
            document_id=document_id,
        )
        added_relations: list[Relation] = []
        for raw in raw_relations:
            if not raw.source_entity_id or not raw.target_entity_id:
                continue
            try:
                relation = self.add_relation(
                    raw.source_entity_id,
                    raw.target_entity_id,
                    raw.relation_type,
                    weight=raw.weight,
                    metadata=raw.metadata,
                    source_documents=raw.source_documents,
                )
                added_relations.append(relation)
            except KeyError as exc:
                logger.debug("Skipping relation with missing entity: %s", exc)

        logger.info(
            "Extracted %d entities and %d relations from text (doc=%s)",
            len(added_entities),
            len(added_relations),
            document_id or "(none)",
        )
        return added_entities, added_relations

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph to a plain dict."""
        return {
            "entities": [e.model_dump() for e in self._entities.values()],
            "relations": [r.model_dump() for r in self._relations.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeGraph:
        """Reconstruct a graph from a dict produced by :meth:`to_dict`."""
        graph = cls()
        for ent_data in data.get("entities", []):
            entity = Entity.model_validate(ent_data)
            graph._entities[entity.id] = entity
            graph._name_index[entity.name.lower()].add(entity.id)
            for alias in entity.aliases:
                graph._name_index[alias.lower()].add(entity.id)
            graph._type_index[entity.entity_type].add(entity.id)
        for rel_data in data.get("relations", []):
            relation = Relation.model_validate(rel_data)
            graph._relations[relation.id] = relation
            graph._outgoing[relation.source_entity_id].add(relation.id)
            graph._incoming[relation.target_entity_id].add(relation.id)
            graph._relation_type_index[relation.relation_type].add(relation.id)
        return graph

    def save(self, path: Path | str) -> None:
        """Save the graph to a JSON file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = file_path.with_suffix(file_path.suffix + ".tmp")
        try:
            tmp.write_text(
                json.dumps(self.to_dict(), ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(file_path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        logger.debug(
            "Saved knowledge graph (%d entities, %d relations) to %s",
            self.entity_count,
            self.relation_count,
            file_path,
        )

    @classmethod
    def load(cls, path: Path | str) -> KnowledgeGraph:
        """Load a graph from a JSON file.

        Returns an empty graph if the file does not exist or is invalid.
        """
        file_path = Path(path)
        if not file_path.exists():
            return cls()
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load knowledge graph from %s: %s", file_path, exc)
            return cls()
        return cls.from_dict(data)

    def clear(self) -> None:
        """Remove all entities and relations."""
        self._entities.clear()
        self._relations.clear()
        self._name_index.clear()
        self._type_index.clear()
        self._outgoing.clear()
        self._incoming.clear()
        self._relation_type_index.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_entity(
        self, name: str, entity_type: str | None = None
    ) -> str | None:
        """Return the entity ID for a name (case-insensitive), or None."""
        for eid in self._name_index.get(name.lower(), set()):
            entity = self._entities.get(eid)
            if entity is None:
                continue
            if entity_type is None or entity.entity_type == entity_type:
                return eid
        return None

    def _remove_relation_internal(self, relation_id: str) -> bool:
        """Remove a relation and clean up all indexes."""
        rel = self._relations.pop(relation_id, None)
        if rel is None:
            return False
        self._outgoing[rel.source_entity_id].discard(relation_id)
        self._incoming[rel.target_entity_id].discard(relation_id)
        self._relation_type_index[rel.relation_type].discard(relation_id)
        return True


__all__ = [
    "Entity",
    "EntityType",
    "KnowledgeGraph",
    "Relation",
    "extract_entities",
    "extract_relations",
]
