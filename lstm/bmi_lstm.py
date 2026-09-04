#
# Copyright (C) 2025 Austin Raney, Lynker
#
# Author:
#   Austin Raney <araney@lynker.com>
#   Jonathan Frame <jmframe@ua.edu>
#
#  In hopes to avoid confusion, there are five main abstractions of note:
#
#  bmi_LSTM:
#       Class that implements the BMI interface and interoperates with the
#       NextGen Framework. Its roles are to (1) manage variables passed to and
#       from the framework, (2) orchestrate running one or more LSTM models
#       (each contained in an `EnsembleMember`), and (3) ensemble output from
#       aforementioned `EnsembleMember`s model for consumption by NextGen
#       Framework.
#
# State:
#       Represents a collection of variables (`Var`s), and provides methods to
#       access and mutate individual `Vars`. Not to be confused with an LSTM
#       model's 'hidden states' or 'cell states', or the dynamic modeling
#       typical taxonomy of 'state-space'. This is simply a container type for
#       passing data to and from the framework. An `EnsembleMember`, which
#       contains an LSTM model, queries (see `EnsembleMember.update()`) a
#       `State` container to receive its input (e.g. precipitation).
#
# Var:
#       Representation of a model variable with a name, unit, and value
#       (`np.NDArray`) that may or may not be exposed via BMI (e.g. static LSTM
#       attributes). `Var`s exposed over BMI are queried and mutated inplace
#       using their `value` property (`np.NDArray`).
#
# EnsembleMember:
#       Class that encapsulates a _single_ LSTM model and all necessary
#       subcomponents. Its roles are to (1) manage an LSTM model's internal
#       states (tensors) and (2) run an LSTM model by querying a `State`
#       container (see: return its output. `EnsembleMember.update()`) for
#       input.
#
# Nextgen_CudaLSTM:
#       Represents a PyTorch-based LSTM model used to make predictions. This
#       class is responsible for (1) defining the LSTM architecture, (2)
#       managing hidden and cell states, and (3) performing forward inference
#       (i.e. takes input data, processes it through the LSTM, and returns
#       predicted outputs along with updated hidden and cell states) based on
#       the information collected in the BMI. An `Nextgen_CudaLSTM` contains a
#       single instance of an LSTM model that performs stepwise predictions
#       without handling model orchestration or input-output management.
#
from __future__ import annotations

import collections
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import yaml
try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

from . import nextgen_cuda_lstm
from . import serialization_codec
from .base import BmiBase
from .logger import configure_logging, logger
from .model_state import State, StateFacade, Var

# --------------   Dynamic Attributes -----------------------------
_dynamic_input_vars = [
    ("land_surface_radiation~incoming~longwave__energy_flux", "W m-2"),
    ("land_surface_air__pressure", "Pa"),
    ("atmosphere_air_water~vapor__relative_saturation", "kg kg-1"),
    ("atmosphere_water__liquid_equivalent_precipitation_rate", "mm h-1"),
    ("land_surface_radiation~incoming~shortwave__energy_flux", "W m-2"),
    ("land_surface_air__temperature", "degK"),
    ("land_surface_wind__x_component_of_velocity", "m s-1"),
    ("land_surface_wind__y_component_of_velocity", "m s-1"),
]

# --------------    Name Mappings    -----------------------------
DYNAMIC_INPUT_NAME_CROSSWALK = {
    "DLWRF_surface": "land_surface_radiation~incoming~longwave__energy_flux",
    "PRES_surface": "land_surface_air__pressure",
    "SPFH_2maboveground": "atmosphere_air_water~vapor__relative_saturation",
    "APCP_surface": "atmosphere_water__liquid_equivalent_precipitation_rate",
    "DSWRF_surface": "land_surface_radiation~incoming~shortwave__energy_flux",
    "TMP_2maboveground": "land_surface_air__temperature",
    "UGRD_10maboveground": "land_surface_wind__x_component_of_velocity",
    "VGRD_10maboveground": "land_surface_wind__y_component_of_velocity",
}

# --------------   Static Attributes -----------------------------

_output_vars = [
    ("land_surface_water__runoff_volume_flux", "m3 s-1"),
    ("land_surface_water__runoff_depth", "m"),
]


def crosswalk_to_external(name: str):
    """Return the external name (the name exposed via BMI) for a given internal name."""
    return DYNAMIC_INPUT_NAME_CROSSWALK.get(name, name)

# ---------------  Ensemble Member -----------------------------


class EnsembleMember:
    """
    An `EnsembleMember` is responsible for initializing and maintaining an LSTM model,
    handling input scaling, managing hidden and cell states, and performing
    inference using the trained model.
    """

    def __init__(self, cfg: dict[str, typing.Any], output_scaling_factor_cms: float):
        self.cfg = cfg
        # NOTE: aaraney: not sure if this *should* go here. leaving it for now.
        self.output_scaling_factor_cms = output_scaling_factor_cms

        # load training feature scales
        scaler_file = cfg["run_dir"] / "train_data/train_data_scaler.yml"
        with scaler_file.open("r") as fp:
            train_data_scaler = yaml.load(fp, Loader=SafeLoader)
        self.scalars = load_training_scalars(cfg, train_data_scaler)

        # initialize torch lstm object
        self.lstm = initialize_lstm(cfg)

        # TODO: aaraney: how to handle input mapping conceptually?
        # NOTE: this is the expected order of variables in the model input
        # tensor, which is required to match the training order when used
        self.input_names = cfg["dynamic_inputs"] + cfg["static_attributes"]

        # WARNING: This implementation of the LSTM can only handle a batch size of 1
        # No need to included different batch sizes
        batch_size = 1
        hidden_layer_size = cfg["hidden_size"]
        # if init_config['initial_state'] == 'zero':
        # NOTE: aaraney: assume initial state is always zero (ask jframe about this. no other option now)
        self.h_t = torch.zeros(1, batch_size, hidden_layer_size).float()
        self.c_t = torch.zeros(1, batch_size, hidden_layer_size).float()

    def update(self, state: Valuer) -> typing.Iterable[Var]:
        """
        Run a single model timestep and return the model inference values.

        `state` contains the input variable names, units, and values for the
        current iteration.
        """
        with torch.no_grad():
            inputs = gather_inputs(state, self.input_names)
            scaled = scale_inputs(
                inputs, self.scalars.input_mean, self.scalars.input_std
            )
            input_tensor = torch.tensor(scaled)
            lstm_output, self.h_t, self.c_t = self.lstm.forward(
                input_tensor, self.h_t, self.c_t
            )
            # TODO: aaraney, there is gap here between mapping 'internal'
            # output names to 'external' output names. Right now this is
            # hard-coded and handled in `scale_outputs`. Introduce semantics
            # for more generally handling outputs.
            yield from scale_outputs(
                self.cfg,
                lstm_output,
                self.scalars.output_mean,
                self.scalars.output_std,
                self.output_scaling_factor_cms,
            )


def bmi_array(arr: list[float]) -> npt.NDArray:
    """Trivial wrapper function to ensure the expected numpy array datatype is used."""
    return np.array(arr, dtype="float64")


class Valuer(typing.Protocol):
    """Thin interface with the same signature as `State.value`."""

    def value(self, name: str) -> npt.NDArray: ...


@dataclass
class TrainingScalars:
    input_mean: npt.NDArray
    input_std: npt.NDArray
    output_mean: npt.NDArray
    output_std: npt.NDArray


def load_training_scalars(
    cfg: dict[str, typing.Any], train_data_scalar: dict[str, typing.Any]
) -> TrainingScalars:
    out_mean = train_data_scalar["xarray_feature_center"]["data_vars"][
        cfg["target_variables"][0]
    ]["data"]
    out_std = train_data_scalar["xarray_feature_scale"]["data_vars"][
        cfg["target_variables"][0]
    ]["data"]

    input_mean = bmi_array(
        [
            train_data_scalar["xarray_feature_center"]["data_vars"][x]["data"]
            for x in cfg["dynamic_inputs"]
        ]
        + [train_data_scalar["attribute_means"][x] for x in cfg["static_attributes"]]
    )

    input_std = bmi_array(
        [
            train_data_scalar["xarray_feature_scale"]["data_vars"][x]["data"]
            for x in cfg["dynamic_inputs"]
        ]
        + [train_data_scalar["attribute_stds"][x] for x in cfg["static_attributes"]]
    )
    return TrainingScalars(
        input_mean=input_mean,
        input_std=input_std,
        output_mean=out_mean,
        output_std=out_std,
    )


def initialize_lstm(cfg: dict[str, typing.Any]) -> nextgen_cuda_lstm.Nextgen_CudaLSTM:
    # Collect the LSTM model architecture details from the configuration file
    input_size = len(cfg["dynamic_inputs"]) + len(cfg["static_attributes"])
    # TODO: aaraney: verify there is a mapping from internal names to external names
    hidden_layer_size = cfg["hidden_size"]
    output_size = len(cfg["target_variables"])
    lstm = nextgen_cuda_lstm.Nextgen_CudaLSTM(
        input_size=input_size,
        hidden_layer_size=hidden_layer_size,
        output_size=output_size,
        batch_size=1,
        seq_length=1,
    )
    # ------------ Load in the trained weights ----------------------------#
    # Save the default model weights. We need to make sure we have the same keys.
    default_state_dict = lstm.state_dict()

    trained_model_file = cfg["run_dir"] / "model_epoch{}.pt".format(
        str(cfg["epochs"]).zfill(3)
    )
    trained_state_dict = torch.load(
        trained_model_file, map_location=torch.device("cpu")
    )

    # Changing the name of the head weights, since different in NH
    trained_state_dict["head.weight"] = trained_state_dict.pop("head.net.0.weight")
    trained_state_dict["head.bias"] = trained_state_dict.pop("head.net.0.bias")
    trained_state_dict = {x: trained_state_dict[x] for x in default_state_dict.keys()}

    # Load in the trained weights.
    lstm.load_state_dict(trained_state_dict)
    return lstm


def gather_inputs(
    state: Valuer, train_input_names: typing.Iterable[str]
) -> npt.NDArray:
    logger.debug("Collecting LSTM inputs ...")

    input_list = []
    for name in train_input_names:
        bmi_name = crosswalk_to_external(name)
        value = state.value(bmi_name)
        assert value.size == 1, "`value` should a single scalar in a 1d array"
        input_list.append(value[0])
        logger.debug("  var_name=%s", bmi_name)
        logger.debug("  type(value)=%s", type(value))
        logger.debug("  value=%s", value)

    collected = bmi_array(input_list)
    logger.debug("Collected inputs: %s",collected)
    return collected


def scale_inputs(
    input: npt.NDArray, mean: npt.NDArray, std: npt.NDArray
) -> npt.NDArray:
    logger.debug("Normalizing the tensor...")
    logger.debug("  input_mean =", mean)
    logger.debug("  input_std  =", std)

    # Center and scale the input values for use in torch
    input_array_scaled = (input - mean) / std
    logger.debug("### input_array =%s", input)
    logger.debug("### dtype(input_array) =%s", input.dtype)
    logger.debug("### type(input_array_scaled) =%s", type(input_array_scaled))
    logger.debug("### dtype(input_array_scaled) =%s", input_array_scaled.dtype)
    return input_array_scaled


def scale_outputs(
    cfg: dict[str, typing.Any],
    output: torch.tensor,
    output_mean: npt.NDArray,
    output_std: npt.NDArray,
    output_scale_factor_cms: float,
):
    logger.debug("model output: %s", output[0, 0, 0].numpy().tolist())

    if cfg["target_variables"][0] in ["qobs_mm_per_hour", "QObs(mm/hr)", "QObs(mm/h)"]:
        surface_runoff_mm = output[0, 0, 0].numpy() * output_std + output_mean
    elif cfg["target_variables"][0] in ["QObs(mm/d)"]:
        # daily to hourly
        surface_runoff_mm = (output[0, 0, 0].numpy() * output_std + output_mean) * (
            1 / 24
        )
    else:
        raise RuntimeError("unreachable")

    # clamp
    surface_runoff_mm = max(surface_runoff_mm, 0.0)
    # mm -> m
    surface_runoff_m = surface_runoff_mm / 1000.0

    # TODO: aaraney, this is kind of gross. think of a better way to do this.
    # The output is area normalized, this is needed to un-normalize it
    # mm->m                             km2 -> m2          hour->s
    # (1/1000) * (self.cfg_bmi['area_sqkm'] * 1000*1000) * (1/3600)
    surface_runoff_volume_m3_s = surface_runoff_mm * output_scale_factor_cms

    # TODO: aaraney: consider making this into a class or closure to avoid so
    # many small allocations.
    yield from (
        Var(
            name="land_surface_water__runoff_depth",
            unit="m",
            value=bmi_array([surface_runoff_m]),
        ),
        Var(
            name="land_surface_water__runoff_volume_flux",
            unit="m3 s-1",
            value=bmi_array([surface_runoff_volume_m3_s]),
        ),
    )


# ---------------  Model Fingerprint  -----------------------------

FINGERPRINT_PREFIX: typing.Final[str] = "lstm-bmi"
"""Leading token of every fingerprint, so the text is recognizable in a hex dump."""


def member_fingerprint(index: int, member: EnsembleMember) -> str:
    """
    Return the deterministic textual identity of a single ensemble member.

    The text records, in order, the member's hidden size, its ordered input
    names (dynamic inputs followed by static attributes, exactly as fed to the
    model), the name of the trained-model run directory, and the epoch number
    of the loaded weights. It is meant to be readable in a hex dump and is
    compared as bytes, never parsed.
    """
    cfg = member.cfg
    run_dir = Path(cfg["run_dir"]).name
    return (
        f"{index}:hidden={int(cfg['hidden_size'])},"
        f"inputs={'|'.join(member.input_names)},"
        f"run={run_dir},"
        f"epoch={int(cfg['epochs'])}"
    )


def compute_fingerprint(members: typing.Sequence[EnsembleMember]) -> bytes:
    """
    Derive the UTF-8 fingerprint identifying an ensemble of members.

    The fingerprint is ``lstm-bmi;members=<count>`` followed by one
    :func:`member_fingerprint` section per member, in ensemble order, all
    joined by ``;``. Two modules initialized from equivalent configurations
    produce identical bytes; a change in member count, order, hidden size,
    input names, trained-model run directory, or epoch changes the bytes.
    """
    sections = [FINGERPRINT_PREFIX, f"members={len(members)}"]
    sections.extend(
        member_fingerprint(index, member) for index, member in enumerate(members)
    )
    return ";".join(sections).encode("utf-8")


# ---------------  ngen BMI Serialization Protocol  -----------------------------
#
# The four reserved variable names below are the surface of the ngen BMI
# Serialization Protocol. They are discovered by name, never enumerated: they
# do not appear in `get_input_var_names` / `get_output_var_names` and have no
# spatial semantics, so `get_var_grid` and `get_var_location` keep raising for
# them as for any unknown name. ngen decides whether a model conforms by an
# exact string comparison of `get_var_units` on each name.
#
# Capture is driven through `set_value`: the create trigger packs the computed
# state (see `bmi_LSTM.snapshot`) into the state buffer and records its byte
# length in the size variable; the free trigger releases the buffer. ngen
# guarantees create and free are paired, but free is safe at any time, and
# `finalize()` releases the buffer as well. Neither trigger alters the
# computed state, so a capture mid-run cannot perturb later timesteps.

SERIALIZATION_CREATE: typing.Final[str] = "ngen::serialization_create"
"""trigger: capture a snapshot of the computed state into the payload buffer"""
SERIALIZATION_FREE: typing.Final[str] = "ngen::serialization_free"
"""trigger: release the payload buffer"""
SERIALIZATION_SIZE: typing.Final[str] = "ngen::serialization_size"
"""byte count of the payload buffer (read after create; set before restore)"""
SERIALIZATION_STATE: typing.Final[str] = "ngen::serialization_state"
"""the opaque payload bytes (read to serialize; set to restore)"""

SERIALIZATION_TRIGGER_UNIT: typing.Final[str] = "ngen::trigger"
SERIALIZATION_SIZE_UNIT: typing.Final[str] = "bytes"
SERIALIZATION_OPAQUE_UNIT: typing.Final[str] = "ngen::opaque"

SERIALIZATION_VAR_NAMES: typing.Final[tuple[str, ...]] = (
    SERIALIZATION_CREATE,
    SERIALIZATION_FREE,
    SERIALIZATION_SIZE,
    SERIALIZATION_STATE,
)
"""all reserved protocol names, in protocol-document order"""


def tensor_to_state_array(tensor: torch.Tensor) -> npt.NDArray[np.float32]:
    """
    Return an independent, flat float32 numpy copy of a member state tensor.

    Member tensors are shaped (1, batch size 1, hidden size); the copy is
    flattened to the hidden size, which is the layout the codec stores.
    """
    return np.array(tensor.detach().cpu().numpy(), dtype="float32").ravel()


def build_serialization_state() -> State:
    """
    Create the `State` backing the four reserved protocol variables.

    The two triggers are one-element int32 arrays, the size is a one-element
    int64 array, and the state is an empty uint8 array (its `Var.value` is
    replaced, not resized, whenever a payload is captured or released). The
    arrays exist from construction so the names resolve for introspection
    before `initialize()` is called.
    """
    return State(
        vars=(
            Var(
                name=SERIALIZATION_CREATE,
                unit=SERIALIZATION_TRIGGER_UNIT,
                value=np.zeros(1, dtype="int32"),
            ),
            Var(
                name=SERIALIZATION_FREE,
                unit=SERIALIZATION_TRIGGER_UNIT,
                value=np.zeros(1, dtype="int32"),
            ),
            Var(
                name=SERIALIZATION_SIZE,
                unit=SERIALIZATION_SIZE_UNIT,
                value=np.zeros(1, dtype="int64"),
            ),
            Var(
                name=SERIALIZATION_STATE,
                unit=SERIALIZATION_OPAQUE_UNIT,
                value=np.empty(0, dtype="uint8"),
            ),
        )
    )


# ---------------  LSTM BMI Wrapper  -----------------------------


def build_state(vars: typing.Iterable[tuple[str, str]]) -> State:
    """
    Create a `State` object from a collection of (name: str, unit: str) tuples.
    Each `Var`'s `value` array is initialized to an `np.array([0.0], dtype="float64"))`.
    """
    g = (Var(name=name, unit=unit, value=bmi_array([0.0])) for (name, unit) in vars)
    return State(vars=g)


def load_static_attributes(cfg_static_attrs: dict[str, typing.Any], state: State):
    for name in state.names():
        value = cfg_static_attrs[name]
        state.set_value(name, bmi_array([value]))


class bmi_LSTM(BmiBase):
    _timestep_size_s: typing.Final[int] = 3600
    """model timestep size in seconds"""

    def __init__(self) -> None:
        # _bmi_ variable state; this is separate from lstm ensemble member state.
        self._dynamic_inputs = build_state(_dynamic_input_vars)
        self._outputs = build_state(_output_vars)
        # reserved ngen serialization protocol variables; resolved by name only
        # and deliberately kept out of the input / output name lists.
        self._serialization = build_serialization_state()

        # current model timestep.
        # e.g. current time = self._timestep * self._timestep_size_s
        self._timestep: int = 0

        ### type hints ###
        # for clarify and type checking, the following type hints are defined
        # here, however the names are bound and initialized in `initialize`.
        self.cfg_bmi: dict[str, typing.Any]
        self.ensemble_members: list[EnsembleMember]
        self._fingerprint: bytes
        """model identity derived from the ensemble members; see `compute_fingerprint`"""

    def initialize(self, config_file: str) -> None:
        # read and setup main configuration file
        with open(config_file, "r") as fp:
            self.cfg_bmi = yaml.load(fp, Loader=SafeLoader)

        _static_input_vars = [
            (key, '1') 
            for key in self.cfg_bmi["static_attributes"].keys()
        ]
        
        self._static_inputs = build_state(_static_input_vars)
        
        coerce_config(self.cfg_bmi)

        # TODO: aaraney: config logging levels to python logging levels
        # setup logging
        # self.cfg_bmi["verbose"]
        configure_logging()

        # ----------- The output is area normalized, this is needed to un-normalize it
        #                         mm->m                             km2 -> m2          hour->s
        output_factor_cms = (
            (1 / 1000) * (self.cfg_bmi["area_sqkm"] * 1000 * 1000) * (1 / 3600)
        )

        # initialize ensemble members
        self.ensemble_members = []
        for member_cfg_file in self.cfg_bmi["train_cfg_file"]:
            cfg = yaml.load(member_cfg_file.read_text(), Loader=SafeLoader)
            coerce_config(cfg)
            member = EnsembleMember(cfg, output_factor_cms)
            self.ensemble_members.append(member)

            provided_inputs = {v[0] for v in _static_input_vars} | {v for v in member.input_names if v in DYNAMIC_INPUT_NAME_CROSSWALK}
            required_inputs = set(member.input_names)

            if not required_inputs.issubset(provided_inputs):
                missing = required_inputs - provided_inputs
                raise ValueError(
                    f"Missing required inputs: {missing}.\n"
                    f"Provided in the config: {provided_inputs}\n"
                    f"Expected by the lstm: {sorted(required_inputs)}\n"
                )

        # load static variables from config into state
        load_static_attributes(self.cfg_bmi["static_attributes"], self._static_inputs)

        # identity of the fully constructed ensemble, used to reject state
        # payloads produced by a differently configured module.
        self._fingerprint = compute_fingerprint(self.ensemble_members)

    def update(self) -> None:
        """update a single timestep."""

        # wrap dynamic and static inputs in a container so an `EnsembleMember`
        # can access both like they are from the same `State` object.
        #
        # each ensemble member will query the `state` object for its required inputs.
        # this could ensemble members with a different number of required features in the future.
        state = StateFacade(self._dynamic_inputs, self._static_inputs)

        outputs: dict[str, list[float]] = collections.defaultdict(list)
        for member in self.ensemble_members:
            for output in member.update(state):
                assert len(output.value) == 1, (
                    f"expected output of length 1, got {len(output.value)}"
                )
                outputs[output.name].append(output.value[0])

        # ensemble output and set output variables
        for name, values in outputs.items():
            self._outputs.set_value(name, np.mean(values, dtype="float64"))

        # increment model timestep
        self._timestep += 1

    def update_until(self, time: float) -> None:
        if time <= self.get_current_time():
            current_time = self.get_current_time()
            logger.warning("no update performed: time=%s <= current_time=%s", time, current_time)
            return None

        n_steps, remainder = divmod(
            time - self.get_current_time(), self.get_time_step()
        )

        if remainder != 0:
            logger.warning(
                "time is not multiple of time step size. updating until: %s", (time - remainder)
            )

        for _ in range(int(n_steps)):
            self.update()

    def finalize(self) -> None:
        # release any captured serialization payload; nothing else is held.
        self._serialization_free()

    def get_component_name(self) -> str:
        return "LSTM"

    def get_input_item_count(self) -> int:
        return len(self._dynamic_inputs)

    def get_output_item_count(self) -> int:
        return len(self._outputs)

    def get_input_var_names(self) -> tuple[str, ...]:  # type: ignore
        return tuple(self._dynamic_inputs.names())

    def get_output_var_names(self) -> tuple[str, ...]:  # type: ignore
        return tuple(self._outputs.names())

    def get_var_grid(self, name: str) -> int:
        # Note: all vars have grid 0 but check if its in names list first
        # raises KeyError on failure
        first_containing(name, self._outputs, self._dynamic_inputs)
        return 0

    def get_var_type(self, name: str) -> str:
        return self.get_value_ptr(name).dtype.name

    def get_var_units(self, name: str) -> str:
        return first_containing(
            name, self._outputs, self._dynamic_inputs, self._serialization
        ).unit(name)

    def get_var_itemsize(self, name: str) -> int:
        return self.get_value_ptr(name).itemsize

    def get_var_nbytes(self, name: str) -> int:
        return self.get_var_itemsize(name) * len(self.get_value_ptr(name))

    def get_var_location(self, name: str) -> str:
        # raises KeyError on failure
        first_containing(name, self._outputs, self._dynamic_inputs)
        return "node"

    def get_current_time(self) -> float:
        return self._timestep * self._timestep_size_s

    def get_start_time(self) -> float:
        return 0

    def get_end_time(self) -> float:
        return np.finfo("d").max  # type: ignore

    def get_time_units(self) -> str:
        return "s"

    def get_time_step(self) -> float:
        return self._timestep_size_s

    def get_value(self, name: str, dest: np.ndarray) -> np.ndarray:
        """_Copies_ a variable's np.NDArray into `dest` and returns `dest`."""
        dest[:] = self.get_value_ptr(name)
        return dest

    def get_value_ptr(self, name: str) -> np.ndarray:
        """Returns a _reference_ to a variable's np.NDArray."""
        return first_containing(
            name, self._outputs, self._dynamic_inputs, self._serialization
        ).value(name)

    def get_value_at_indices(
        self, name: str, dest: np.ndarray, inds: np.ndarray
    ) -> np.ndarray:
        return first_containing(
            name, self._outputs, self._dynamic_inputs
        ).value_at_indices(name, dest, inds)

    def set_value(self, name: str, src: np.ndarray) -> None:
        # the protocol triggers are commands, not values: `src` is ignored.
        if name == SERIALIZATION_CREATE:
            self._serialization_create()
            return None
        if name == SERIALIZATION_FREE:
            self._serialization_free()
            return None
        return first_containing(name, self._outputs, self._dynamic_inputs).set_value(
            name, src
        )

    def set_value_at_indices(
        self, name: str, inds: np.ndarray, src: np.ndarray
    ) -> None:
        return first_containing(
            name, self._outputs, self._dynamic_inputs
        ).set_value_at_indices(name, inds, src)

    # Grid information
    def get_grid_rank(self, grid: int) -> int:
        # 0 is the only id we have
        if grid == 0:
            return 1
        raise RuntimeError(f"unsupported grid rank: {grid!s}. only support 0")

    def get_grid_size(self, grid: int) -> int:
        # 0 is the only id we have
        if grid == 0:
            return 1
        raise RuntimeError(f"unsupported grid size: {grid!s}. only support 0")

    def get_grid_type(self, grid: int) -> str:
        # 0 is the only id we have
        if grid == 0:
            return "scalar"
        raise RuntimeError(f"unsupported grid type: {grid!s}. only support 0")

    # ngen BMI Serialization Protocol: capture and release

    def snapshot(self) -> serialization_codec.Snapshot:
        """
        Gather the module's computed state into a codec `Snapshot`.

        The snapshot holds the timestep counter, the fingerprint, each ensemble
        member's hidden and cell state as flat float32 copies (in ensemble
        order), and the output values as float64 (in output variable order).
        Nothing on the module is modified, and the returned arrays do not alias
        the member tensors. Raises `RuntimeError` before `initialize()`.
        """
        if not hasattr(self, "_fingerprint"):
            raise RuntimeError(
                "cannot capture serialization state before initialize() is called"
            )
        members = [
            (tensor_to_state_array(member.h_t), tensor_to_state_array(member.c_t))
            for member in self.ensemble_members
        ]
        outputs = np.array(
            [self._outputs.value(name)[0] for name in self._outputs.names()],
            dtype=serialization_codec.OUTPUT_DTYPE,
        )
        return serialization_codec.Snapshot(
            timestep=self._timestep,
            fingerprint=self._fingerprint,
            members=members,
            outputs=outputs,
        )

    def capture_state(self) -> bytes:
        """Pack `snapshot()` into payload bytes; see `serialization_codec`."""
        return serialization_codec.pack(self.snapshot())

    def _serialization_var(self, name: str) -> Var:
        """Return the `Var` backing a reserved protocol name."""
        for var in self._serialization:
            if var.name == name:
                return var
        raise KeyError(f"unknown serialization name: {name!s}")

    def _serialization_create(self) -> None:
        """
        Handle the create trigger: capture the state into the payload buffer.

        The buffer and size are only replaced once packing has succeeded, so a
        failed capture (e.g. before `initialize()`) leaves both untouched.
        """
        payload = self.capture_state()
        # an owned, writable uint8 copy; `len()` equals the size reported below.
        self._serialization_var(SERIALIZATION_STATE).value = np.frombuffer(
            payload, dtype="uint8"
        ).copy()
        self._serialization.set_value(SERIALIZATION_SIZE, len(payload))

    def _serialization_free(self) -> None:
        """Handle the free trigger: release the payload buffer. Safe at any time."""
        self._serialization_var(SERIALIZATION_STATE).value = np.empty(0, dtype="uint8")
        self._serialization.set_value(SERIALIZATION_SIZE, 0)


def coerce_config(cfg: dict[str, typing.Any]):
    for key, val in cfg.items():
        # Handle 'train_cfg_file' specifically to ensure it is always a list
        if key == "train_cfg_file":
            if val is not None and val != "None":
                if isinstance(val, list):
                    cfg[key] = [Path(element) for element in val]
                else:
                    cfg[key] = [Path(val)]
            else:
                cfg[key] = []

        # Convert all path strings to PosixPath objects for other keys
        elif any([key.endswith(x) for x in ["_dir", "_path", "_file", "_files"]]):
            if val is not None and val != "None":
                if isinstance(val, list):
                    temp_list = []
                    for element in val:
                        temp_list.append(Path(element))
                    cfg[key] = temp_list
                else:
                    cfg[key] = Path(val)
            else:
                cfg[key] = None

        # Convert Dates to pandas Datetime indexs
        elif key.endswith("_date"):
            if isinstance(val, list):
                temp_list = []
                for elem in val:
                    temp_list.append(pd.to_datetime(elem, format="%d/%m/%Y"))
                cfg[key] = temp_list
            else:
                cfg[key] = pd.to_datetime(val, format="%d/%m/%Y")


def first_containing(name: str, *states: State) -> State:
    """
    Return the first `State` object containing `name` in `states`.
    Otherwise, raise `KeyError`.
    """
    for state in states:
        if name in state:
            return state
    raise KeyError(f"unknown name: {name!s}")
