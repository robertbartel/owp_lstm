"""
Binary codec for the LSTM BMI serialization payload.

This module packs a :class:`Snapshot` of the module's computed state into a
versioned, explicit, little-endian byte layout and unpacks such bytes back into
a :class:`Snapshot`. It depends only on the standard library ``struct`` module
and numpy: no pickle, no torch, and no reference to the BMI class, so a payload
can be decoded without executing arbitrary code.

Payload layout (format version 1)
=================================

All multi-byte integers and floats are little-endian with no padding.

+---------------------------+------------------+-------------------------------------------+
| Section                   | Size (bytes)     | Content                                   |
+===========================+==================+===========================================+
| magic                     | 8                | ``b"LSTMBMI\\0"`` (:data:`MAGIC`)          |
+---------------------------+------------------+-------------------------------------------+
| version                   | 4 (uint32)       | format version (:data:`FORMAT_VERSION`)   |
+---------------------------+------------------+-------------------------------------------+
| timestep                  | 8 (int64)        | module timestep counter when captured     |
+---------------------------+------------------+-------------------------------------------+
| member_count              | 4 (uint32)       | number of ensemble members, ``M``         |
+---------------------------+------------------+-------------------------------------------+
| output_count              | 4 (uint32)       | number of output values, ``N``            |
+---------------------------+------------------+-------------------------------------------+
| fingerprint_len           | 4 (uint32)       | byte length of the fingerprint, ``F``     |
+---------------------------+------------------+-------------------------------------------+
| fingerprint               | ``F``            | UTF-8 model fingerprint text              |
+---------------------------+------------------+-------------------------------------------+
| member[i].hidden_size     | 4 (uint32)       | hidden size ``H_i`` of member ``i``       |
+---------------------------+------------------+-------------------------------------------+
| member[i].hidden          | ``4 * H_i``      | hidden state, raw float32                 |
+---------------------------+------------------+-------------------------------------------+
| member[i].cell            | ``4 * H_i``      | cell state, raw float32                   |
+---------------------------+------------------+-------------------------------------------+
| outputs                   | ``8 * N``        | output values, raw float64                |
+---------------------------+------------------+-------------------------------------------+

The three ``member[i]`` rows repeat ``M`` times, in ensemble order. The fixed
header (magic through ``fingerprint_len``) is :data:`HEADER_SIZE` bytes. The
total payload length is therefore::

    HEADER_SIZE + F + sum(MEMBER_HEADER_SIZE + 8 * H_i for i in range(M)) + 8 * N

Any change to this layout must increment :data:`FORMAT_VERSION`.

Validation
==========

:func:`unpack` validates a payload before returning anything, in this order:
minimum length for the fixed header, magic bytes, known format version,
non-negative counts, total length exactly equal to the length implied by the
header and the per-member hidden sizes, and (when an expected fingerprint is
given) an exact byte comparison of the fingerprint. Every failure raises
:class:`PayloadError`, a ``ValueError`` subclass, with a message naming the
cause. :func:`check_fingerprint` exposes the fingerprint comparison on its own
so a caller can verify identity before applying a decoded snapshot.
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

if sys.version_info < (3, 10):
    import typing_extensions as typing
else:
    import typing

# `slots` feature added to `dataclass` in 3.10
if sys.version_info < (3, 10):
    _dataclass_kwargs = {}
else:
    _dataclass_kwargs = {"slots": True}

__all__ = [
    "MAGIC",
    "FORMAT_VERSION",
    "HEADER_STRUCT",
    "HEADER_SIZE",
    "MEMBER_HEADER_STRUCT",
    "MEMBER_HEADER_SIZE",
    "STATE_DTYPE",
    "OUTPUT_DTYPE",
    "Snapshot",
    "PayloadError",
    "pack",
    "unpack",
    "check_fingerprint",
]

MAGIC: typing.Final[bytes] = b"LSTMBMI\0"
"""Eight magic bytes that open every payload."""

FORMAT_VERSION: typing.Final[int] = 1
"""Payload format version written by :func:`pack`."""

HEADER_STRUCT: typing.Final[struct.Struct] = struct.Struct("<8sIqIII")
"""
Fixed header: magic, version, timestep (int64), member_count, output_count,
fingerprint_len.
"""

HEADER_SIZE: typing.Final[int] = HEADER_STRUCT.size
"""Byte length of the fixed header (32)."""

MEMBER_HEADER_STRUCT: typing.Final[struct.Struct] = struct.Struct("<I")
"""Per-member header: hidden size as uint32."""

MEMBER_HEADER_SIZE: typing.Final[int] = MEMBER_HEADER_STRUCT.size
"""Byte length of the per-member header (4)."""

STATE_DTYPE: typing.Final[np.dtype] = np.dtype("<f4")
"""Element type of the hidden and cell state sections (little-endian float32)."""

OUTPUT_DTYPE: typing.Final[np.dtype] = np.dtype("<f8")
"""Element type of the outputs section (little-endian float64)."""


class PayloadError(ValueError):
    """
    Raised by :func:`unpack` and :func:`check_fingerprint` when a payload is
    malformed, has an unknown format, does not match its own header, or does
    not carry the expected fingerprint.
    """


@dataclass(**_dataclass_kwargs)
class Snapshot:
    """
    The module's computed state as plain numpy data.

    Attributes:
        timestep: the module's integer timestep counter when captured.
        fingerprint: UTF-8 bytes identifying the model configuration.
        members: per ensemble member, in order, a ``(hidden, cell)`` pair of
            float32 arrays. :func:`pack` accepts any array shape and flattens
            it; :func:`unpack` always returns one-dimensional arrays of length
            equal to the member's hidden size.
        outputs: float64 array of output values in output variable order.
    """

    timestep: int
    fingerprint: bytes
    members: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]
    outputs: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=OUTPUT_DTYPE)
    )


def _as_state_array(arr: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Return ``arr`` as a flat, contiguous little-endian float32 array."""
    return np.ascontiguousarray(arr, dtype=STATE_DTYPE).ravel()


def pack(snapshot: Snapshot) -> bytes:
    """
    Serialize a :class:`Snapshot` into payload bytes following the module
    layout table.

    Hidden and cell arrays are cast to float32 and flattened; outputs are cast
    to float64 and flattened. Raises ``ValueError`` if a member's hidden and
    cell arrays do not have the same number of elements.
    """
    fingerprint = bytes(snapshot.fingerprint)
    outputs = np.ascontiguousarray(snapshot.outputs, dtype=OUTPUT_DTYPE).ravel()

    chunks: list[bytes] = [
        HEADER_STRUCT.pack(
            MAGIC,
            FORMAT_VERSION,
            int(snapshot.timestep),
            len(snapshot.members),
            outputs.size,
            len(fingerprint),
        ),
        fingerprint,
    ]

    for index, (hidden, cell) in enumerate(snapshot.members):
        hidden_flat = _as_state_array(hidden)
        cell_flat = _as_state_array(cell)
        if hidden_flat.size != cell_flat.size:
            raise ValueError(
                f"member {index}: hidden state has {hidden_flat.size} elements "
                f"but cell state has {cell_flat.size}"
            )
        chunks.append(MEMBER_HEADER_STRUCT.pack(hidden_flat.size))
        chunks.append(hidden_flat.tobytes())
        chunks.append(cell_flat.tobytes())

    chunks.append(outputs.tobytes())
    return b"".join(chunks)


def _coerce_fingerprint(fingerprint: typing.Union[bytes, bytearray, memoryview, str]) -> bytes:
    """Return ``fingerprint`` as bytes; ``str`` is encoded as UTF-8."""
    if isinstance(fingerprint, str):
        return fingerprint.encode("utf-8")
    return bytes(fingerprint)


def check_fingerprint(
    actual: typing.Union[Snapshot, bytes, bytearray, memoryview, str],
    expected: typing.Union[bytes, bytearray, memoryview, str],
) -> None:
    """
    Compare a fingerprint against the expected one byte for byte.

    ``actual`` may be a :class:`Snapshot` (its ``fingerprint`` is used) or raw
    fingerprint bytes/text. Raises :class:`PayloadError` on mismatch and
    returns ``None`` otherwise.
    """
    actual_bytes = _coerce_fingerprint(actual.fingerprint if isinstance(actual, Snapshot) else actual)
    expected_bytes = _coerce_fingerprint(expected)
    if actual_bytes != expected_bytes:
        raise PayloadError(
            "fingerprint mismatch: payload carries "
            f"{actual_bytes!r} but this module expects {expected_bytes!r}"
        )


def _validated_layout(buf: memoryview) -> tuple[int, int, int, int, bytes, list[int]]:
    """
    Read and validate the header and per-member hidden sizes of ``buf``.

    Returns ``(timestep, member_count, output_count, fingerprint_len,
    fingerprint, hidden_sizes)``. Raises :class:`PayloadError` if the payload
    is too short for the header, opens with the wrong magic, has an unknown
    format version, carries a negative count, or has a total length that
    differs from the one implied by the header and hidden sizes.
    """
    total = len(buf)
    if total < HEADER_SIZE:
        raise PayloadError(
            f"payload is {total} bytes, shorter than the {HEADER_SIZE}-byte header"
        )

    magic, version, timestep, member_count, output_count, fingerprint_len = (
        HEADER_STRUCT.unpack_from(buf, 0)
    )
    if magic != MAGIC:
        raise PayloadError(f"bad magic bytes: expected {MAGIC!r}, found {bytes(magic)!r}")
    if version != FORMAT_VERSION:
        raise PayloadError(
            f"unsupported payload format version {version}; this codec reads version {FORMAT_VERSION}"
        )
    # The header fields are unsigned on the wire, so these can only trip if the
    # header struct is ever changed to signed fields; they keep the documented
    # validation order explicit.
    for name, value in (
        ("member_count", member_count),
        ("output_count", output_count),
        ("fingerprint_len", fingerprint_len),
    ):
        if value < 0:
            raise PayloadError(f"negative {name} in header: {value}")

    offset = HEADER_SIZE + fingerprint_len
    if total < offset:
        raise PayloadError(
            f"payload is {total} bytes but the header implies at least {offset} "
            f"bytes (truncated inside the {fingerprint_len}-byte fingerprint)"
        )
    fingerprint = bytes(buf[HEADER_SIZE:offset])

    hidden_sizes: list[int] = []
    for index in range(member_count):
        if total < offset + MEMBER_HEADER_SIZE:
            raise PayloadError(
                f"payload is {total} bytes but the header implies at least "
                f"{offset + MEMBER_HEADER_SIZE} bytes (truncated at member {index} of "
                f"{member_count} header)"
            )
        (hidden_size,) = MEMBER_HEADER_STRUCT.unpack_from(buf, offset)
        if hidden_size < 0:
            raise PayloadError(f"negative hidden size for member {index}: {hidden_size}")
        hidden_sizes.append(hidden_size)
        offset += MEMBER_HEADER_SIZE + 2 * hidden_size * STATE_DTYPE.itemsize
        if total < offset:
            raise PayloadError(
                f"payload is {total} bytes but the header implies at least {offset} "
                f"bytes (truncated inside member {index} of {member_count}, hidden size "
                f"{hidden_size})"
            )

    implied = offset + output_count * OUTPUT_DTYPE.itemsize
    if total < implied:
        raise PayloadError(
            f"payload is {total} bytes but the header implies {implied} bytes "
            f"(truncated inside the {output_count}-element outputs section)"
        )
    if total > implied:
        raise PayloadError(
            f"payload is {total} bytes but the header implies {implied} bytes "
            f"({total - implied} extra trailing bytes)"
        )

    return timestep, member_count, output_count, fingerprint_len, fingerprint, hidden_sizes


def unpack(
    payload: typing.Union[bytes, bytearray, memoryview, npt.NDArray[np.uint8]],
    expected_fingerprint: typing.Optional[typing.Union[bytes, bytearray, memoryview, str]] = None,
) -> Snapshot:
    """
    Deserialize payload bytes produced by :func:`pack` into a :class:`Snapshot`.

    The payload is fully validated (see the module docstring) before any
    snapshot is built, so a :class:`PayloadError` is raised and nothing is
    returned for a malformed, unknown-version, truncated, or over-long
    payload. If ``expected_fingerprint`` is given, the payload's fingerprint
    must match it exactly, byte for byte, or :class:`PayloadError` is raised.

    Sections are read in layout order. Returned arrays are independent copies,
    so mutating them does not alias ``payload``.
    """
    buf = memoryview(payload).cast("B")

    timestep, _member_count, output_count, fingerprint_len, fingerprint, hidden_sizes = (
        _validated_layout(buf)
    )
    if expected_fingerprint is not None:
        check_fingerprint(fingerprint, expected_fingerprint)

    offset = HEADER_SIZE + fingerprint_len
    members: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]] = []
    for hidden_size in hidden_sizes:
        offset += MEMBER_HEADER_SIZE
        section = hidden_size * STATE_DTYPE.itemsize
        hidden = np.frombuffer(buf, dtype=STATE_DTYPE, count=hidden_size, offset=offset).copy()
        offset += section
        cell = np.frombuffer(buf, dtype=STATE_DTYPE, count=hidden_size, offset=offset).copy()
        offset += section
        members.append((hidden, cell))

    outputs = np.frombuffer(buf, dtype=OUTPUT_DTYPE, count=output_count, offset=offset).copy()

    return Snapshot(
        timestep=int(timestep),
        fingerprint=fingerprint,
        members=members,
        outputs=outputs,
    )
