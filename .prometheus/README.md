# Prometheus training adapter

This directory is the hardware-free boundary between HumanEgo's native
`python -m training.FlowMatchingTrainer` entry point and PrometheusV4's policy
training runner.

The adapter accepts only `humanego_preprocessed_v1` datasets with a bimanual
20-value hand target in this modality-major order:

1. left xyz (3), right xyz (3);
2. left rotation-6D (6), right rotation-6D (6);
3. left grasp (1), right grasp (1).

These values are hand-pose targets. They are **not** ARX joint commands, and
this repository contains no approved 20D-to-14D ARX decoder. Robot conversion,
serving, and rollout therefore remain outside this training source. The adapter
sets `hardware_rollout_authorized` to `false` and exposes no prepare, standalone
evaluation, export, or serve stage.

The dataset contract adds a `humanego` mapping with `task`, `source_type`,
`recipe`, and `image_name`. `dataset.uri` points to an external data root whose
layout is `<root>/<task>/<source_type>/<session>/preprocess/all_data`. At least
one held-out and one training session must be present. The run directory must
also be outside the source checkout. The adapter writes only a resolved YAML
and native training artifacts to that external run directory.

HumanEgo's documented environment is Conda with Python 3.11. The dependency
file is a versioned environment specification, not a fully locked environment.

The native `latest.pt` checkpoint restores model weights, AdamW state, epoch,
global step, and best-metric metadata. It does not restore EMA shadow weights,
AMP scaler, RNG, or dataloader state. Resume is consequently declared
`partial_state_non_bit_exact`, not full-state or bit-exact.

Run the dependency-free source probe:

```bash
python .prometheus/adapter.py doctor
```

Print the native argv without importing the model stack or running training:

```bash
python .prometheus/adapter.py train \
  --dataset-contract /managed/contracts/humanego.yaml \
  --run-dir /managed/runs/humanego-001 \
  --plan
```
