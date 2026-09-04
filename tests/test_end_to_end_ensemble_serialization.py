"""
End-to-end equivalence tests for the ngen BMI Serialization Protocol with a
multi-member ensemble on real forcing.

The repository's example ensemble config references a second trained model
that is not in the tree, so these tests write a temporary two-member config
listing the bundled trained model twice with the golden config's static
attributes and area. A run that is interrupted, captured, restored into a
fresh two-member module, and continued must reproduce an uninterrupted run
bitwise, for both outputs and for every member's hidden and cell tensors.
Payloads must also never cross between single- and two-member modules.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest

from lstm import bmi_lstm
from lstm import serialization_protocol as protocol
from lstm import serialization_codec as codec

REPO_ROOT = Path(__file__).parent.parent
BASIN_ID = "02064000"
GOLDEN_CONFIG = REPO_ROOT / f"configs/{BASIN_ID}_nh_NLDAS_hourly.yml"
FORCING_FILE = REPO_ROOT / "data/usgs-streamflow-nldas_hourly.nc"

# identical to the mapping used by tests/integration_test.py
FORCING_VARIABLE_NAME_MAPPING = {
    "total_precipitation": "atmosphere_water__liquid_equivalent_precipitation_rate",
    "temperature": "land_surface_air__temperature",
    "longwave_radiation": "land_surface_radiation~incoming~longwave__energy_flux",
    "shortwave_radiation": "land_surface_radiation~incoming~shortwave__energy_flux",
    "pressure": "land_surface_air__pressure",
    "specific_humidity": "atmosphere_air_water~vapor__relative_saturation",
    "wind_u": "land_surface_wind__x_component_of_velocity",
    "wind_v": "land_surface_wind__y_component_of_velocity",
}

TOTAL_STEPS = 24
SPLIT_STEP = 12

TRIGGER = np.array([1], dtype="int32")

BUNDLED_MODEL_CONFIG = (
    "./trained_neuralhydrology_models/"
    "nh_AORC_hourly_slope_elev_precip_temp_seq999_seed101_2801_191806/config.yml"
)


@pytest.fixture(autouse=True)
def _repo_root_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config files on this branch reference paths relative to the repository root."""
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture(scope="module")
def nldas_forcing() -> list[dict[str, np.ndarray]]:
    """
    The first 24 hours of NLDAS forcing for the golden basin, as one dict per
    step keyed by BMI input name. Read once so every module sees identical values.
    """
    with nc.Dataset(FORCING_FILE, "r") as ds:
        basins = ds.variables["basin"][:]
        (basin_idxs,) = np.where(basins == BASIN_ID)
        assert len(basin_idxs) == 1
        basin_idx = basin_idxs[0]
        steps = []
        for ts in range(TOTAL_STEPS):
            steps.append(
                {
                    bmi_name: np.array(ds.variables[nc_name][basin_idx, ts], dtype="float64")
                    for nc_name, bmi_name in FORCING_VARIABLE_NAME_MAPPING.items()
                }
            )
    return steps


@pytest.fixture
def two_member_config(tmp_path: Path) -> Path:
    """
    A two-member ensemble config listing the bundled trained model twice, with
    the golden config's static attributes and area.
    """
    cfg = tmp_path / "two_member_e2e.yml"
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


def _initialized(config: Path = GOLDEN_CONFIG) -> bmi_lstm.bmi_LSTM:
    model = bmi_lstm.bmi_LSTM()
    model.initialize(str(config))
    return model


def _step(model: bmi_lstm.bmi_LSTM, inputs: dict[str, np.ndarray]) -> dict[str, float]:
    """Set every forcing input, update once, and return both outputs by name."""
    for name, value in inputs.items():
        model.set_value(name, value)
    model.update()
    return {name: float(model.get_value_ptr(name)[0]) for name in model.get_output_var_names()}


def _run(model: bmi_lstm.bmi_LSTM, forcing: list[dict[str, np.ndarray]]) -> dict[str, list[float]]:
    """Run `model` over `forcing`, returning per-output lists of recorded values."""
    recorded: dict[str, list[float]] = {name: [] for name in model.get_output_var_names()}
    for inputs in forcing:
        for name, value in _step(model, inputs).items():
            recorded[name].append(value)
    return recorded


def _capture_and_free(model: bmi_lstm.bmi_LSTM) -> tuple[int, bytes]:
    """The ngen save sequence: create, read size and state, free."""
    model.set_value(protocol.SERIALIZATION_CREATE, TRIGGER)
    size = int(model.get_value_ptr(protocol.SERIALIZATION_SIZE)[0])
    state = model.get_value_ptr(protocol.SERIALIZATION_STATE).tobytes()
    model.set_value(protocol.SERIALIZATION_FREE, TRIGGER)
    return size, state


def _announce_and_deliver(model: bmi_lstm.bmi_LSTM, size: int, state: bytes) -> None:
    """The ngen restore sequence: announce the byte count, then deliver the bytes."""
    model.set_value(protocol.SERIALIZATION_SIZE, np.array([size], dtype="int64"))
    model.set_value(protocol.SERIALIZATION_STATE, np.frombuffer(state, dtype="uint8"))


def _member_arrays(model: bmi_lstm.bmi_LSTM) -> list[tuple[np.ndarray, np.ndarray]]:
    """Flat float32 copies of every member's (hidden, cell) state via the member accessor."""
    return [m.state_arrays() for m in model.ensemble_members]


def _assert_members_bitwise_equal(actual: bmi_lstm.bmi_LSTM, expected: bmi_lstm.bmi_LSTM) -> None:
    """Every member's hidden and cell tensors match bitwise, with dtype and shape."""
    actual_arrays = _member_arrays(actual)
    expected_arrays = _member_arrays(expected)
    assert len(actual_arrays) == len(expected_arrays)
    for (h_a, c_a), (h_e, c_e) in zip(actual_arrays, expected_arrays):
        assert h_a.dtype == h_e.dtype == np.float32
        assert h_a.shape == h_e.shape
        assert np.array_equal(h_a, h_e)
        assert np.array_equal(c_a, c_e)
    for restored, original in zip(actual.ensemble_members, expected.ensemble_members):
        assert restored.h_t.shape == original.h_t.shape
        assert restored.c_t.shape == original.c_t.shape


def _observe(model: bmi_lstm.bmi_LSTM) -> dict:
    """Everything a rejected restore must leave alone, copied for later comparison."""
    return {
        "members": _member_arrays(model),
        "outputs": {
            name: float(model.get_value_ptr(name)[0]) for name in model.get_output_var_names()
        },
        "time": model.get_current_time(),
    }


def _assert_untouched(model: bmi_lstm.bmi_LSTM, before: dict) -> None:
    after = _observe(model)
    assert len(after["members"]) == len(before["members"])
    for (h_a, c_a), (h_b, c_b) in zip(after["members"], before["members"]):
        assert np.array_equal(h_a, h_b)
        assert np.array_equal(c_a, c_b)
    assert after["outputs"] == before["outputs"]
    assert after["time"] == before["time"]
    # the protocol invariant survives a rejected delivery: size == buffer length
    assert int(model.get_value_ptr(protocol.SERIALIZATION_SIZE)[0]) == len(
        model.get_value_ptr(protocol.SERIALIZATION_STATE)
    )


def test_two_member_config_builds_a_two_member_ensemble(two_member_config: Path):
    model = _initialized(two_member_config)
    assert len(model.ensemble_members) == 2
    assert len(_initialized().ensemble_members) == 1


def test_two_member_split_and_restore_matches_uninterrupted_run(
    two_member_config: Path,
    nldas_forcing: list[dict[str, np.ndarray]],
):
    """
    24 hours uninterrupted versus 12 hours, capture, restore into a fresh
    two-member module, and 12 more hours: both output variables and every
    member's hidden and cell tensors must match bitwise.
    """
    assert len(nldas_forcing) == TOTAL_STEPS

    uninterrupted = _initialized(two_member_config)
    assert len(uninterrupted.ensemble_members) == 2
    expected = _run(uninterrupted, nldas_forcing)
    assert set(expected) == {name for name, _ in bmi_lstm._output_vars}
    assert all(len(values) == TOTAL_STEPS for values in expected.values())
    assert any(value != 0.0 for value in expected["land_surface_water__runoff_depth"])

    # first half: run 12 steps, then perform the ngen save sequence
    first_half = _initialized(two_member_config)
    observed = _run(first_half, nldas_forcing[:SPLIT_STEP])
    size, state = _capture_and_free(first_half)
    assert size == len(state) > 0
    assert first_half.get_value_ptr(protocol.SERIALIZATION_STATE).size == 0

    # the payload really carries two members (and is larger than a single-member one)
    snapshot = codec.unpack(state)
    assert len(snapshot.members) == 2
    single = _initialized()
    _run(single, nldas_forcing[:SPLIT_STEP])
    single_size, _ = _capture_and_free(single)
    assert size > single_size

    # second half: fresh two-member module, ngen restore sequence, remaining 12 steps
    second_half = _initialized(two_member_config)
    assert len(second_half.ensemble_members) == 2
    _announce_and_deliver(second_half, size, state)
    continued = _run(second_half, nldas_forcing[SPLIT_STEP:])
    for name in observed:
        observed[name].extend(continued[name])

    # bitwise equality of all 24 recorded values of each output variable
    assert observed.keys() == expected.keys()
    for name in expected:
        assert len(observed[name]) == TOTAL_STEPS
        assert observed[name] == expected[name], name
        assert np.array_equal(
            np.array(observed[name], dtype="float64"),
            np.array(expected[name], dtype="float64"),
        )

    # exact equality of every member's hidden and cell tensors after step 24
    assert len(second_half.ensemble_members) == len(uninterrupted.ensemble_members) == 2
    _assert_members_bitwise_equal(second_half, uninterrupted)


def test_two_member_restored_module_matches_source_at_split(
    two_member_config: Path,
    nldas_forcing: list[dict[str, np.ndarray]],
):
    """Immediately after restore the fresh two-member module equals the source at step 12."""
    source = _initialized(two_member_config)
    at_split = _run(source, nldas_forcing[:SPLIT_STEP])
    size, state = _capture_and_free(source)

    target = _initialized(two_member_config)
    _announce_and_deliver(target, size, state)

    _assert_members_bitwise_equal(target, source)
    assert len(target.ensemble_members) == 2
    for name, values in at_split.items():
        assert float(target.get_value_ptr(name)[0]) == values[-1]
    # the timestep is carried in the payload but not applied on restore
    assert target.get_current_time() == target.get_start_time()
    assert source.get_current_time() == SPLIT_STEP * source.get_time_step()


@pytest.mark.parametrize(
    "source_kind, target_kind",
    [("two", "single"), ("single", "two")],
    ids=["two_member_payload_into_single_member", "single_member_payload_into_two_member"],
)
def test_cross_member_count_restore_raises_and_leaves_target_untouched(
    source_kind: str,
    target_kind: str,
    two_member_config: Path,
    nldas_forcing: list[dict[str, np.ndarray]],
):
    """
    Restoring a two-member payload into a single-member module, and vice
    versa, is rejected on fingerprint mismatch before anything is mutated,
    whether the target is freshly initialized or has already stepped.
    """
    configs = {"single": GOLDEN_CONFIG, "two": two_member_config}

    source = _initialized(configs[source_kind])
    _run(source, nldas_forcing[:SPLIT_STEP])
    size, state = _capture_and_free(source)

    # freshly initialized target: zeros stay zeros
    fresh = _initialized(configs[target_kind])
    before = _observe(fresh)
    assert all(not h.any() and not c.any() for h, c in before["members"])
    assert all(v == 0.0 for v in before["outputs"].values())
    fresh.set_value(protocol.SERIALIZATION_SIZE, np.array([size], dtype="int64"))
    with pytest.raises(codec.PayloadError, match="fingerprint"):
        fresh.set_value(protocol.SERIALIZATION_STATE, np.frombuffer(state, dtype="uint8"))
    _assert_untouched(fresh, before)
    assert fresh.get_current_time() == fresh.get_start_time()

    # a target that has already stepped keeps its non-trivial state
    stepped = _initialized(configs[target_kind])
    _run(stepped, nldas_forcing[:3])
    before = _observe(stepped)
    assert any(h.any() for h, _ in before["members"])
    stepped.set_value(protocol.SERIALIZATION_SIZE, np.array([size], dtype="int64"))
    with pytest.raises(codec.PayloadError, match="fingerprint"):
        stepped.set_value(protocol.SERIALIZATION_STATE, np.frombuffer(state, dtype="uint8"))
    _assert_untouched(stepped, before)

    # and the rejected target still runs normally afterwards, matching an untouched twin
    twin = _initialized(configs[target_kind])
    _run(twin, nldas_forcing[:3])
    assert _run(stepped, nldas_forcing[3:6]) == _run(twin, nldas_forcing[3:6])
    _assert_members_bitwise_equal(stepped, twin)
