All notable changes to this project will be documented in this file.
We follow the [Semantic Versioning 2.0.0](http://semver.org/) format.


## Unreleased

### Added

- Support for the ngen BMI Serialization Protocol (v0.2), so NextGen can
  checkpoint and restore the module's computed state. The four reserved
  variables `ngen::serialization_create`, `ngen::serialization_free`,
  `ngen::serialization_size`, and `ngen::serialization_state` resolve through
  BMI introspection with the units `ngen::trigger`, `ngen::trigger`, `bytes`,
  and `ngen::opaque`, and are absent from the input and output variable lists.
  No new keys are needed in the module's own configuration file; checkpointing
  is enabled from the realization config's `serialization` block.
- `lstm.serialization_codec`: a pickle-free, torch-free, versioned binary
  payload format (magic `LSTMBMI`, format version 1) carrying the timestep,
  a model fingerprint, every ensemble member's hidden and cell state, and the
  output values. Decoding validates the header, section lengths, and
  fingerprint before returning, and raises `PayloadError` otherwise.
- `lstm.serialization_protocol`: the four reserved variables, their payload
  buffer, and the trigger, size-announcement, and delivery dispatch live in
  this module, independent of the LSTM; `bmi_lstm` wires it to the codec.
  The size announcement allocates the buffer, so `ngen::serialization_size`
  always equals the buffer length, and a delivered payload whose length
  differs from the announcement is rejected with `ValueError`.
- A model fingerprint computed at `initialize()` from the ensemble members
  (member count, hidden size, input names, trained-model run directory name,
  and epoch). A restore whose fingerprint does not match is rejected before
  any state is mutated.
- Codec unit tests, BMI protocol tests, and end-to-end tests that assert a
  split-and-restore run matches an uninterrupted run bitwise for a
  single-member and a two-member ensemble configuration.
- README section "NextGen Serialization (Checkpoint and Restore)" documenting
  the reserved variables and the payload layout, and a checkpoint/restore
  recipe in `doc/How_to_Run_LSTM_in_NextGen.txt`.

### Changed

- `finalize()` now releases any outstanding serialization buffer.
- Documentation paths updated from the removed `bmi_config_files/` directory
  to `configs/`.

## x.y.z - YYYY-MM-DD

### Added

- Lorem ipsum dolor sit amet

### Deprecated

- Nothing.

### Removed

- Nothing.

### Fixed

- Nothing.
