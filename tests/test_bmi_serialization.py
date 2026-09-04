"""
Tests for the ngen BMI Serialization Protocol support on `lstm.bmi_lstm.bmi_LSTM`.

The bundled trained model and the golden single-member config are loaded from
the repository root, matching the config files' repo-root-relative paths.
"""

from __future__ import annotations

import textwrap
import types
from pathlib import Path

import numpy as np
import pytest

from lstm import bmi_lstm

REPO_ROOT = Path(__file__).parent.parent
SINGLE_MEMBER_CONFIG = REPO_ROOT / "configs/02064000_nh_NLDAS_hourly.yml"
BUNDLED_MODEL_CONFIG = (
    "./trained_neuralhydrology_models/"
    "nh_AORC_hourly_slope_elev_precip_temp_seq999_seed101_2801_191806/config.yml"
)


@pytest.fixture(autouse=True)
def _repo_root_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config files on this branch reference paths relative to the repository root."""
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture
def two_member_config(tmp_path: Path) -> Path:
    """
    A two-member ensemble config listing the bundled trained model twice, with
    the golden config's static attributes and area.
    """
    cfg = tmp_path / "two_member.yml"
    cfg.write_text(
        textwrap.dedent(
            f"""\
            train_cfg_file:
              - {BUNDLED_MODEL_CONFIG}
              - {BUNDLED_MODEL_CONFIG}
            basin_id: '02064000'
            basin_name: FALLING RIVER NEAR NARUNA, VA
            initial_state: zero
            time_step: 1 hour
            verbose: 0
            static_attributes:
              slope_mean: 9.95686
              elev_mean: 192.21
            area_sqkm: 427.77
            """
        )
    )
    return cfg


def _initialized(config: Path) -> bmi_lstm.bmi_LSTM:
    model = bmi_lstm.bmi_LSTM()
    model.initialize(str(config))
    return model


# ---------------  fingerprint  -----------------------------


def _fake_member(hidden_size: int, inputs: list[str], run_dir: str, epochs: int):
    """Stand-in with the attributes `member_fingerprint` reads; no torch model."""
    cfg = {"hidden_size": hidden_size, "run_dir": Path(run_dir), "epochs": epochs}
    return types.SimpleNamespace(cfg=cfg, input_names=list(inputs))


def test_compute_fingerprint_is_readable_text():
    members = [
        _fake_member(8, ["a", "b", "elev_mean"], "./runs/demo_run", 3),
        _fake_member(4, ["a"], "/abs/other_run", 12),
    ]
    fp = bmi_lstm.compute_fingerprint(members)
    assert isinstance(fp, bytes)
    assert fp == (
        b"lstm-bmi;members=2;"
        b"0:hidden=8,inputs=a|b|elev_mean,run=demo_run,epoch=3;"
        b"1:hidden=4,inputs=a,run=other_run,epoch=12"
    )
    # decodes cleanly as UTF-8 and is readable
    assert fp.decode("utf-8").startswith(bmi_lstm.FINGERPRINT_PREFIX)


def test_compute_fingerprint_no_members():
    assert bmi_lstm.compute_fingerprint([]) == b"lstm-bmi;members=0"


@pytest.mark.parametrize(
    "change",
    [
        dict(hidden_size=9),
        dict(inputs=["b", "a"]),
        dict(inputs=["a"]),
        dict(run_dir="./runs/demo_run_2"),
        dict(epochs=4),
    ],
)
def test_compute_fingerprint_sensitive_to_each_field(change: dict):
    base = dict(hidden_size=8, inputs=["a", "b"], run_dir="./runs/demo_run", epochs=3)
    reference = bmi_lstm.compute_fingerprint([_fake_member(**base)])
    changed = bmi_lstm.compute_fingerprint([_fake_member(**{**base, **change})])
    assert reference != changed


def test_compute_fingerprint_ignores_run_dir_parent():
    """Only the run directory's name participates, not where it is checked out."""
    here = _fake_member(8, ["a"], "./trained/demo_run", 3)
    elsewhere = _fake_member(8, ["a"], "/somewhere/else/demo_run", 3)
    assert bmi_lstm.compute_fingerprint([here]) == bmi_lstm.compute_fingerprint(
        [elsewhere]
    )


def test_fingerprint_set_at_initialize():
    model = bmi_lstm.bmi_LSTM()
    assert not hasattr(model, "_fingerprint")
    model.initialize(str(SINGLE_MEMBER_CONFIG))
    assert isinstance(model._fingerprint, bytes)
    assert model._fingerprint == (
        b"lstm-bmi;members=1;"
        b"0:hidden=126,inputs=APCP_surface|TMP_2maboveground|elev_mean|slope_mean,"
        b"run=nh_AORC_hourly_slope_elev_precip_temp_seq999_seed101_2801_191806,"
        b"epoch=9"
    )
    assert model._fingerprint == bmi_lstm.compute_fingerprint(model.ensemble_members)


def test_fingerprint_identical_for_same_config():
    a = _initialized(SINGLE_MEMBER_CONFIG)
    b = _initialized(SINGLE_MEMBER_CONFIG)
    assert a._fingerprint == b._fingerprint


def test_fingerprint_differs_between_single_and_two_member_config(
    two_member_config: Path,
):
    single = _initialized(SINGLE_MEMBER_CONFIG)
    double = _initialized(two_member_config)
    assert len(double.ensemble_members) == 2
    assert single._fingerprint != double._fingerprint
    assert double._fingerprint.startswith(b"lstm-bmi;members=2;")
    # both members are the bundled model, so their sections differ only by index
    sections = double._fingerprint.decode("utf-8").split(";")[2:]
    assert [s.split(":", 1)[0] for s in sections] == ["0", "1"]
    assert sections[0].split(":", 1)[1] == sections[1].split(":", 1)[1]


def test_fingerprint_stable_across_updates():
    model = _initialized(SINGLE_MEMBER_CONFIG)
    before = bytes(model._fingerprint)
    for name in model.get_input_var_names():
        model.set_value(name, np.array([1.0], dtype="float64"))
    for _ in range(3):
        model.update()
    assert model.get_current_time() == 3 * model.get_time_step()
    assert model._fingerprint == before


# ---------------  protocol surface (reserved variables)  -----------------------------

RESERVED = {
    bmi_lstm.SERIALIZATION_CREATE: ("ngen::trigger", "int32", 4),
    bmi_lstm.SERIALIZATION_FREE: ("ngen::trigger", "int32", 4),
    bmi_lstm.SERIALIZATION_SIZE: ("bytes", "int64", 8),
    bmi_lstm.SERIALIZATION_STATE: ("ngen::opaque", "uint8", 1),
}
"""expected (unit, type, itemsize) per reserved name, per the ngen protocol"""


def test_reserved_name_constants_are_exact():
    assert bmi_lstm.SERIALIZATION_CREATE == "ngen::serialization_create"
    assert bmi_lstm.SERIALIZATION_FREE == "ngen::serialization_free"
    assert bmi_lstm.SERIALIZATION_SIZE == "ngen::serialization_size"
    assert bmi_lstm.SERIALIZATION_STATE == "ngen::serialization_state"
    assert bmi_lstm.SERIALIZATION_VAR_NAMES == tuple(RESERVED)
    assert bmi_lstm.SERIALIZATION_TRIGGER_UNIT == "ngen::trigger"
    assert bmi_lstm.SERIALIZATION_SIZE_UNIT == "bytes"
    assert bmi_lstm.SERIALIZATION_OPAQUE_UNIT == "ngen::opaque"


@pytest.fixture(params=["uninitialized", "initialized"])
def module(request: pytest.FixtureRequest) -> bmi_lstm.bmi_LSTM:
    """The reserved names must resolve both before and after `initialize()`."""
    if request.param == "initialized":
        return _initialized(SINGLE_MEMBER_CONFIG)
    return bmi_lstm.bmi_LSTM()


@pytest.mark.parametrize("name", list(RESERVED))
def test_reserved_units_exact(module: bmi_lstm.bmi_LSTM, name: str):
    unit, _, _ = RESERVED[name]
    # ngen's support probe compares this string exactly
    assert module.get_var_units(name) == unit


@pytest.mark.parametrize("name", list(RESERVED))
def test_reserved_type_and_itemsize(module: bmi_lstm.bmi_LSTM, name: str):
    _, dtype, itemsize = RESERVED[name]
    assert module.get_var_type(name) == dtype
    assert module.get_var_itemsize(name) == itemsize
    assert module.get_value_ptr(name).dtype == np.dtype(dtype)


@pytest.mark.parametrize("name", list(RESERVED))
def test_reserved_nbytes_matches_ptr(module: bmi_lstm.bmi_LSTM, name: str):
    ptr = module.get_value_ptr(name)
    assert module.get_var_nbytes(name) == ptr.nbytes
    # ngen's Python adapter sizes arrays from nbytes / itemsize
    assert module.get_var_nbytes(name) // module.get_var_itemsize(name) == len(ptr)


def test_reserved_names_absent_from_var_lists_and_counts(module: bmi_lstm.bmi_LSTM):
    inputs = module.get_input_var_names()
    outputs = module.get_output_var_names()
    for name in RESERVED:
        assert name not in inputs
        assert name not in outputs
    assert module.get_input_item_count() == len(inputs)
    assert module.get_output_item_count() == len(outputs)
    assert not any(n.startswith("ngen::") for n in (*inputs, *outputs))


def test_public_var_lists_unchanged_by_protocol():
    """Adding the reserved names must not alter the pre-existing BMI surface."""
    model = _initialized(SINGLE_MEMBER_CONFIG)
    assert model.get_input_var_names() == tuple(
        name for name, _ in bmi_lstm._dynamic_input_vars
    )
    assert model.get_output_var_names() == tuple(
        name for name, _ in bmi_lstm._output_vars
    )
    assert model.get_input_item_count() == len(bmi_lstm._dynamic_input_vars)
    assert model.get_output_item_count() == len(bmi_lstm._output_vars)


@pytest.mark.parametrize("name", list(RESERVED))
def test_reserved_names_have_no_grid_or_location(
    module: bmi_lstm.bmi_LSTM, name: str
):
    with pytest.raises(KeyError):
        module.get_var_grid(name)
    with pytest.raises(KeyError):
        module.get_var_location(name)


def test_unknown_name_still_raises_same_as_reserved(module: bmi_lstm.bmi_LSTM):
    """Reserved names get the standard unknown-variable signal from grid/location."""
    with pytest.raises(KeyError):
        module.get_var_grid("ngen::not_a_variable")
    with pytest.raises(KeyError):
        module.get_var_location("ngen::not_a_variable")
    with pytest.raises(KeyError):
        module.get_var_units("ngen::not_a_variable")


def test_size_reads_zero_and_state_reads_empty_when_fresh(module: bmi_lstm.bmi_LSTM):
    size_ptr = module.get_value_ptr(bmi_lstm.SERIALIZATION_SIZE)
    assert size_ptr.shape == (1,)
    assert size_ptr[0] == 0

    size = module.get_value(bmi_lstm.SERIALIZATION_SIZE, np.empty(1, dtype="int64"))
    assert size.dtype == np.dtype("int64")
    assert size[0] == 0

    state_ptr = module.get_value_ptr(bmi_lstm.SERIALIZATION_STATE)
    assert state_ptr.dtype == np.dtype("uint8")
    assert state_ptr.shape == (0,)
    assert module.get_var_nbytes(bmi_lstm.SERIALIZATION_STATE) == 0

    state = module.get_value(bmi_lstm.SERIALIZATION_STATE, np.empty(0, dtype="uint8"))
    assert state.shape == (0,)


@pytest.mark.parametrize(
    "name", [bmi_lstm.SERIALIZATION_CREATE, bmi_lstm.SERIALIZATION_FREE]
)
def test_trigger_arrays_are_single_int32(module: bmi_lstm.bmi_LSTM, name: str):
    ptr = module.get_value_ptr(name)
    assert ptr.shape == (1,)
    assert ptr.dtype == np.dtype("int32")
    assert module.get_var_nbytes(name) == 4


@pytest.mark.parametrize("name", list(RESERVED))
def test_reserved_reads_are_non_mutating(module: bmi_lstm.bmi_LSTM, name: str):
    first = module.get_value_ptr(name)
    before = first.copy()
    for _ in range(3):
        again = module.get_value_ptr(name)
        assert again is first
        np.testing.assert_array_equal(again, before)
        copied = module.get_value(name, np.empty_like(before))
        np.testing.assert_array_equal(copied, before)
    assert module.get_var_units(name) == RESERVED[name][0]
    assert module.get_var_nbytes(name) == before.nbytes


def test_reserved_state_is_per_instance():
    a = bmi_lstm.bmi_LSTM()
    b = bmi_lstm.bmi_LSTM()
    for name in RESERVED:
        assert a.get_value_ptr(name) is not b.get_value_ptr(name)


def test_build_serialization_state_standalone():
    state = bmi_lstm.build_serialization_state()
    assert tuple(state.names()) == bmi_lstm.SERIALIZATION_VAR_NAMES
    for name, (unit, dtype, itemsize) in RESERVED.items():
        assert state.unit(name) == unit
        assert state.value(name).dtype == np.dtype(dtype)
        assert state.value(name).itemsize == itemsize
