"""
Unit tests for `lstm.serialization_codec`.

Snapshots are built from numpy arrays only; no torch and no BMI instance.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from lstm import serialization_codec as codec


def _member(hidden_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    hidden = rng.standard_normal(hidden_size).astype(np.float32)
    cell = rng.standard_normal(hidden_size).astype(np.float32)
    return hidden, cell


def _expected_length(snapshot: codec.Snapshot) -> int:
    """Total byte length derived directly from the layout table in the module docstring."""
    fixed = codec.HEADER_SIZE + len(snapshot.fingerprint)
    members = sum(
        codec.MEMBER_HEADER_SIZE + 2 * hidden.size * np.dtype(np.float32).itemsize
        for hidden, _ in snapshot.members
    )
    outputs = snapshot.outputs.size * np.dtype(np.float64).itemsize
    return fixed + members + outputs


@pytest.fixture
def single_member_snapshot() -> codec.Snapshot:
    return codec.Snapshot(
        timestep=12,
        fingerprint=b"members=1;0:hidden=8,inputs=a|b,run=demo,epoch=3",
        members=[_member(8, seed=1)],
        outputs=np.array([0.25, 1.5e-3], dtype=np.float64),
    )


@pytest.fixture
def multi_member_snapshot() -> codec.Snapshot:
    return codec.Snapshot(
        timestep=-7,
        fingerprint="members=3;énsemble".encode("utf-8"),
        members=[_member(4, seed=10), _member(64, seed=11), _member(9, seed=12)],
        outputs=np.array([1.0, 2.0, 3.0], dtype=np.float64),
    )


def _assert_snapshots_equal(actual: codec.Snapshot, expected: codec.Snapshot) -> None:
    assert actual.timestep == expected.timestep
    assert actual.fingerprint == expected.fingerprint
    assert len(actual.members) == len(expected.members)
    for (a_h, a_c), (e_h, e_c) in zip(actual.members, expected.members):
        assert np.array_equal(a_h, np.ravel(e_h))
        assert np.array_equal(a_c, np.ravel(e_c))
    assert np.array_equal(actual.outputs, np.ravel(expected.outputs))


def test_constants_match_layout_table():
    assert codec.MAGIC == b"LSTMBMI\0"
    assert len(codec.MAGIC) == 8
    assert codec.FORMAT_VERSION == 1
    assert codec.HEADER_SIZE == 32
    assert codec.MEMBER_HEADER_SIZE == 4
    assert codec.STATE_DTYPE == np.dtype("<f4")
    assert codec.OUTPUT_DTYPE == np.dtype("<f8")


def test_round_trip_single_member(single_member_snapshot: codec.Snapshot):
    payload = codec.pack(single_member_snapshot)
    assert isinstance(payload, bytes)
    _assert_snapshots_equal(codec.unpack(payload), single_member_snapshot)


def test_round_trip_multiple_members_with_differing_hidden_sizes(
    multi_member_snapshot: codec.Snapshot,
):
    payload = codec.pack(multi_member_snapshot)
    restored = codec.unpack(payload)
    _assert_snapshots_equal(restored, multi_member_snapshot)
    assert [h.size for h, _ in restored.members] == [4, 64, 9]


@pytest.mark.parametrize("fixture_name", ["single_member_snapshot", "multi_member_snapshot"])
def test_packed_length_matches_layout(fixture_name: str, request: pytest.FixtureRequest):
    snapshot = request.getfixturevalue(fixture_name)
    assert len(codec.pack(snapshot)) == _expected_length(snapshot)


def test_header_fields_are_readable_in_place(multi_member_snapshot: codec.Snapshot):
    payload = codec.pack(multi_member_snapshot)
    assert payload[:8] == codec.MAGIC
    magic, version, timestep, member_count, output_count, fingerprint_len = struct.unpack_from(
        "<8sIqIII", payload, 0
    )
    assert magic == codec.MAGIC
    assert version == codec.FORMAT_VERSION
    assert timestep == -7
    assert member_count == 3
    assert output_count == 3
    assert fingerprint_len == len(multi_member_snapshot.fingerprint)
    assert payload[32 : 32 + fingerprint_len] == multi_member_snapshot.fingerprint
    # First member header follows the fingerprint and carries its hidden size.
    (first_hidden_size,) = struct.unpack_from("<I", payload, 32 + fingerprint_len)
    assert first_hidden_size == 4


def test_dtype_and_shape_preserved_on_unpack(single_member_snapshot: codec.Snapshot):
    restored = codec.unpack(codec.pack(single_member_snapshot))
    hidden, cell = restored.members[0]
    assert hidden.dtype == np.float32
    assert cell.dtype == np.float32
    assert hidden.shape == (8,)
    assert cell.shape == (8,)
    assert restored.outputs.dtype == np.float64
    assert restored.outputs.shape == (2,)
    assert isinstance(restored.timestep, int)
    assert isinstance(restored.fingerprint, bytes)


def test_pack_flattens_and_casts_member_arrays():
    """Torch-shaped (1, 1, H) float64 inputs are stored as flat float32."""
    hidden = np.arange(6, dtype=np.float64).reshape(1, 1, 6)
    cell = (np.arange(6, dtype=np.float64) * 0.5).reshape(1, 1, 6)
    snapshot = codec.Snapshot(timestep=0, fingerprint=b"", members=[(hidden, cell)], outputs=np.array([]))
    restored = codec.unpack(codec.pack(snapshot))
    assert restored.members[0][0].shape == (6,)
    assert np.array_equal(restored.members[0][0], hidden.ravel().astype(np.float32))
    assert np.array_equal(restored.members[0][1], cell.ravel().astype(np.float32))
    assert restored.outputs.size == 0


def test_empty_snapshot_round_trips():
    snapshot = codec.Snapshot(timestep=0, fingerprint=b"", members=[], outputs=np.array([], dtype=np.float64))
    payload = codec.pack(snapshot)
    assert len(payload) == codec.HEADER_SIZE
    restored = codec.unpack(payload)
    assert restored.members == []
    assert restored.outputs.size == 0
    assert restored.fingerprint == b""


def test_unpack_accepts_uint8_array_and_returns_independent_copies(
    single_member_snapshot: codec.Snapshot,
):
    payload = np.frombuffer(codec.pack(single_member_snapshot), dtype=np.uint8)
    restored = codec.unpack(payload)
    _assert_snapshots_equal(restored, single_member_snapshot)
    hidden, _ = restored.members[0]
    assert hidden.flags.writeable
    hidden[0] = 123.0
    assert codec.unpack(payload).members[0][0][0] != 123.0


def test_pack_rejects_mismatched_hidden_and_cell_sizes():
    snapshot = codec.Snapshot(
        timestep=1,
        fingerprint=b"x",
        members=[(np.zeros(4, dtype=np.float32), np.zeros(5, dtype=np.float32))],
        outputs=np.zeros(1),
    )
    with pytest.raises(ValueError, match="member 0"):
        codec.pack(snapshot)
