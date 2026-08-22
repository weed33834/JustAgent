"""Tests for the multi-agent meeting lifecycle service."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from justagent.communication.meeting import (
    AgendaItem,
    AgendaItemStatus,
    Attendee,
    Meeting,
    MeetingError,
    MeetingService,
    MeetingStatus,
    RSVPStatus,
)


@pytest.fixture
def svc() -> MeetingService:
    return MeetingService()


async def _make_meeting(svc: MeetingService, **kw):
    return await svc.create_meeting(
        title=kw.pop("title", "Standup"),
        organizer=kw.pop("organizer", "alice"),
        attendees=kw.pop(
            "attendees",
            [
                Attendee(user_id="alice", name="Alice"),
                Attendee(user_id="bob", name="Bob"),
            ],
        ),
        agenda=kw.pop(
            "agenda",
            [AgendaItem(title="Blockers"), AgendaItem(title="Demos")],
        ),
        start_time=datetime.now(UTC) + timedelta(days=1),
        end_time=datetime.now(UTC) + timedelta(days=1, hours=1),
        **kw,
    )


class TestModels:
    def test_attendee_respond_and_is_attending(self) -> None:
        a = Attendee(user_id="u1", name="U")
        assert a.is_attending is False
        a.respond(RSVPStatus.TENTATIVE)
        assert a.is_attending is True
        a.respond(RSVPStatus.DECLINED)
        assert a.is_attending is False

    def test_agenda_item_lifecycle(self) -> None:
        item = AgendaItem(title="Topic")
        assert item.status is AgendaItemStatus.PENDING
        item.mark_discussed("went well")
        assert item.status is AgendaItemStatus.DISCUSSED and item.notes == "went well"
        item2 = AgendaItem(title="Later")
        item2.defer("no time")
        assert item2.status is AgendaItemStatus.DEFERRED
        item3 = AgendaItem(title="Skip me")
        item3.skip("obsolete")
        assert item3.status is AgendaItemStatus.SKIPPED

    def test_meeting_counts_and_upcoming(self) -> None:
        start = datetime.now(UTC) + timedelta(days=2)
        m = Meeting(
            title="Retro",
            organizer="a",
            attendees=[
                Attendee(user_id="a", name="A", rsvp=RSVPStatus.ACCEPTED),
                Attendee(user_id="b", name="B", rsvp=RSVPStatus.DECLINED),
                Attendee(user_id="c", name="C"),
            ],
            start_time=start,
            end_time=start + timedelta(minutes=45),
        )
        assert m.accepted_count == 1
        assert m.declined_count == 1
        assert m.pending_count == 1
        assert m.duration_minutes == 45
        assert m.is_upcoming is False  # DRAFT does not count as upcoming
        m.status = MeetingStatus.SCHEDULED
        assert m.is_upcoming is True


@pytest.mark.asyncio
class TestMeetingServiceLifecycle:
    async def test_create_adds_organizer_and_sorts_agenda(self, svc: MeetingService) -> None:
        m = await _make_meeting(
            svc,
            attendees=[Attendee(user_id="bob", name="Bob")],
            agenda=[AgendaItem(title="Second", order=2), AgendaItem(title="First", order=1)],
        )
        assert m.status is MeetingStatus.DRAFT
        assert m.attendees[0].user_id == "alice"  # organizer auto-added first
        assert [i.title for i in m.agenda] == ["First", "Second"]

    async def test_create_rejects_end_before_start(self, svc: MeetingService) -> None:
        now = datetime.now(UTC)
        with pytest.raises(MeetingError, match="after start"):
            await svc.create_meeting(
                title="bad",
                organizer="a",
                start_time=now,
                end_time=now - timedelta(minutes=1),
            )

    async def test_update_and_cancel_rules(self, svc: MeetingService) -> None:
        m = await _make_meeting(svc)
        updated = await svc.update_meeting(m.id, location="Room 2")
        assert updated.location == "Room 2"
        cancelled = await svc.cancel_meeting(m.id, reason="conflict")
        assert cancelled.status is MeetingStatus.CANCELLED
        with pytest.raises(MeetingError, match="Cannot update"):
            await svc.update_meeting(m.id, title="x")

    async def test_rsvp_auto_confirms_when_required_all_answered(self, svc: MeetingService) -> None:
        m = await _make_meeting(
            svc,
            attendees=[
                Attendee(user_id="alice", name="A", required=True),
                Attendee(user_id="bob", name="B", required=True),
            ],
        )
        # mark it SCHEDULED as an external step would
        meeting = await svc.get_meeting(m.id)
        assert meeting is not None
        meeting.status = MeetingStatus.SCHEDULED

        await svc.respond_rsvp(m.id, "alice", RSVPStatus.ACCEPTED)
        still = await svc.get_meeting(m.id)
        assert still is not None and still.status is MeetingStatus.SCHEDULED
        await svc.respond_rsvp(m.id, "bob", RSVPStatus.ACCEPTED)
        final = await svc.get_meeting(m.id)
        assert final is not None and final.status is MeetingStatus.CONFIRMED

    async def test_rsvp_unknown_user_raises(self, svc: MeetingService) -> None:
        m = await _make_meeting(svc)
        with pytest.raises(MeetingError, match="not an attendee"):
            await svc.respond_rsvp(m.id, "ghost", RSVPStatus.ACCEPTED)

    async def test_add_remove_attendee_and_duplicate_guard(self, svc: MeetingService) -> None:
        m = await _make_meeting(svc)
        await svc.add_attendee(m.id, Attendee(user_id="carol", name="C"))
        got = await svc.get_meeting(m.id)
        assert got is not None and got.get_attendee("carol") is not None
        with pytest.raises(MeetingError, match="already in meeting"):
            await svc.add_attendee(m.id, Attendee(user_id="carol", name="C"))
        await svc.remove_attendee(m.id, "carol")
        got2 = await svc.get_meeting(m.id)
        assert got2 is not None and got2.get_attendee("carol") is None

    async def test_agenda_update_status_paths(self, svc: MeetingService) -> None:
        m = await _make_meeting(svc)
        item_id = m.agenda[0].id
        upd = await svc.update_agenda_item(
            m.id, item_id, status=AgendaItemStatus.DISCUSSED, notes="ok"
        )
        assert upd.status is AgendaItemStatus.DISCUSSED and upd.notes == "ok"
        with pytest.raises(MeetingError, match="Agenda item not found"):
            await svc.update_agenda_item(m.id, "ghost-id")

    async def test_list_filters(self, svc: MeetingService) -> None:
        await _make_meeting(svc, title="M1", organizer="alice")
        m2 = await _make_meeting(svc, title="M2", organizer="bob")
        await svc.cancel_meeting(m2.id)
        by_org = await svc.list_meetings(organizer="alice")
        assert [x.title for x in by_org] == ["M1"]
        by_status = await svc.list_meetings(status=MeetingStatus.CANCELLED)
        assert [x.title for x in by_status] == ["M2"]
        by_participant = await svc.list_meetings(participant="bob")
        assert {x.title for x in by_participant} == {"M1", "M2"}

    async def test_minutes_generation_and_text(self, svc: MeetingService) -> None:
        m = await _make_meeting(svc)
        await svc.respond_rsvp(m.id, "alice", RSVPStatus.ACCEPTED)
        await svc.respond_rsvp(m.id, "bob", RSVPStatus.DECLINED)
        minutes = await svc.generate_minutes(m.id, generated_by="alice", decisions=["ship it"])
        assert minutes.attendees_present == ["alice"]
        assert "bob" in minutes.attendees_absent
        text = minutes.to_text()
        assert "Standup" in text and "ship it" in text
        again = await svc.get_minutes(m.id)
        assert again is not None and again.generated_by == "alice"
