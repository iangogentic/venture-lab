"""Identifier helpers: prefixing, and the ordering the registry relies on."""

import time

from app.utils.ids import new_id


def test_ids_carry_their_prefix() -> None:
    assert new_id("ev").startswith("ev_")


def test_ids_are_unique_within_a_millisecond() -> None:
    assert len({new_id("ev") for _ in range(200)}) == 200


def test_ids_sort_in_creation_order() -> None:
    """`find_by_type(limit=1)` and filename listings assume this holds."""
    first = new_id("ev")
    time.sleep(0.002)  # the timestamp is millisecond-grained
    second = new_id("ev")

    assert first < second
