"""
End-to-end equivalence tests for the ngen BMI Serialization Protocol on real
forcing: a run that is interrupted, captured, restored into a fresh module, and
continued must reproduce an uninterrupted run bitwise.

These reuse the golden integration test's basin config and 24 hours of the
bundled NLDAS forcing, driven through the same forcing-name mapping.
"""

from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest

from lstm import bmi_lstm

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
    model.set_value(bmi_lstm.SERIALIZATION_CREATE, TRIGGER)
    size = int(model.get_value_ptr(bmi_lstm.SERIALIZATION_SIZE)[0])
    state = model.get_value_ptr(bmi_lstm.SERIALIZATION_STATE).tobytes()
    model.set_value(bmi_lstm.SERIALIZATION_FREE, TRIGGER)
    return size, state


def _announce_and_deliver(model: bmi_lstm.bmi_LSTM, size: int, state: bytes) -> None:
    """The ngen restore sequence: announce the byte count, then deliver the bytes."""
    model.set_value(bmi_lstm.SERIALIZATION_SIZE, np.array([size], dtype="int64"))
    model.set_value(bmi_lstm.SERIALIZATION_STATE, np.frombuffer(state, dtype="uint8"))


def _member_arrays(model: bmi_lstm.bmi_LSTM) -> list[tuple[np.ndarray, np.ndarray]]:
    """Flat float32 copies of every member's (hidden, cell) state via the member accessor."""
    return [m.state_arrays() for m in model.ensemble_members]


def _assert_members_bitwise_equal(actual: bmi_lstm.bmi_LSTM, expected: bmi_lstm.bmi_LSTM) -> None:
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


def test_single_member_split_and_restore_matches_uninterrupted_run(
    nldas_forcing: list[dict[str, np.ndarray]],
):
    """
    24 hours uninterrupted versus 12 hours, capture, restore into a fresh
    module, and 12 more hours: both output variables and the final hidden and
    cell tensors must match bitwise.
    """
    assert len(nldas_forcing) == TOTAL_STEPS

    # an uninterrupted 24-step run recording both outputs every step
    uninterrupted = _initialized()
    expected = _run(uninterrupted, nldas_forcing)
    assert set(expected) == {name for name, _ in bmi_lstm._output_vars}
    assert all(len(values) == TOTAL_STEPS for values in expected.values())
    # the golden dataset actually exercises the model (not an all-zero run)
    assert any(value != 0.0 for value in expected["land_surface_water__runoff_depth"])

    # first half: run 12 steps, then perform the ngen save sequence
    first_half = _initialized()
    observed = _run(first_half, nldas_forcing[:SPLIT_STEP])
    size, state = _capture_and_free(first_half)
    assert size == len(state) > 0
    assert first_half.get_value_ptr(bmi_lstm.SERIALIZATION_STATE).size == 0

    # second half: fresh module, ngen restore sequence, remaining 12 steps
    second_half = _initialized()
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

    # exact equality of the hidden and cell tensors after step 24
    _assert_members_bitwise_equal(second_half, uninterrupted)


def test_single_member_restored_module_matches_source_at_split(
    nldas_forcing: list[dict[str, np.ndarray]],
):
    """Immediately after restore the fresh module equals the source at step 12."""
    source = _initialized()
    at_split = _run(source, nldas_forcing[:SPLIT_STEP])
    size, state = _capture_and_free(source)

    target = _initialized()
    _announce_and_deliver(target, size, state)

    _assert_members_bitwise_equal(target, source)
    for name, values in at_split.items():
        assert float(target.get_value_ptr(name)[0]) == values[-1]
    # the timestep is carried in the payload but not applied on restore
    assert target.get_current_time() == target.get_start_time()
    assert source.get_current_time() == SPLIT_STEP * source.get_time_step()


def test_golden_run_matches_regenerated_reference_values(
    nldas_forcing: list[dict[str, np.ndarray]],
):
    """
    Guard the forcing read used here against the golden test's driving loop:
    reading the 24 forcing values once up front and passing them as float64
    arrays yields the same runoff series as the golden test's per-step reads.
    """
    upfront = _initialized()
    recorded = _run(upfront, nldas_forcing)["land_surface_water__runoff_depth"]

    with nc.Dataset(FORCING_FILE, "r") as ds:
        (basin_idxs,) = np.where(ds.variables["basin"][:] == BASIN_ID)
        basin_idx = basin_idxs[0]
        per_step = _initialized()
        golden_style = np.zeros(TOTAL_STEPS)
        for ts in range(TOTAL_STEPS):
            for nc_name, bmi_name in FORCING_VARIABLE_NAME_MAPPING.items():
                per_step.set_value(bmi_name, ds.variables[nc_name][basin_idx, ts])
            per_step.update()
            per_step.get_value("land_surface_water__runoff_depth", golden_style[ts : ts + 1])

    assert np.array_equal(np.array(recorded, dtype="float64"), golden_style)
