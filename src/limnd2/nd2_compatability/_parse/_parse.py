from __future__ import annotations

from typing import Any

from .. import nd2file_types as structures

LITE_EVENT_KEYS = {"T", "T2", "M", "D", "A", "I", "S"}


def _is_lite_events(events: dict[str, dict[str, Any]]) -> bool:
    if not events:
        return False
    event_keys = set().union(*(set(x) for x in events.values()))
    return event_keys.issubset(LITE_EVENT_KEYS)


def _load_lite_event(event: dict[str, Any]) -> structures.ExperimentEvent:
    stim_event = event.get("S", {})
    if stim_event:
        stim_struct = structures.StimulationEvent(
            type=structures.StimulationType(stim_event.get("T", 0)),
            loop_index=stim_event.get("L", 0),
            position=stim_event.get("P", 0),
            description=stim_event.get("D", ""),
        )
    else:
        stim_struct = None

    meaning = structures.EventMeaning(event.get("M", 0))
    description = event.get("D", "") or meaning.description()
    if stim_struct:
        description += f" Phase {stim_struct.type.name}"
        if stim_struct.description:
            description += f" - {stim_struct.description}"

    return structures.ExperimentEvent(
        id=event.get("I", 0),
        time=event.get("T", 0.0),
        time2=event.get("T2", 0.0),
        meaning=meaning,
        description=description,
        data=event.get("A", ""),
        stimulation=stim_struct,
    )


def load_events(events: dict[str, Any]) -> list[structures.ExperimentEvent]:
    count = events.get("uiCount", 0)
    if count == 0:
        return []
    p_events = events.get("pEvents", {})
    if _is_lite_events(p_events):
        return [_load_lite_event(x[1]) for x in sorted(p_events.items())]
    return []
