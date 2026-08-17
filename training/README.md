# HumanEgo — Training

Train a HumanEgo **flow-matching policy** from preprocessed Aria data. The policy
predicts a short horizon of future 6-DoF hand (and object) motion from an egocentric
image plus a set of *Interaction-Centric Tokens (ICTs)* — the hands and objects in
the scene. This doc
covers (1) how to train, (2) the data it expects, (3) the files in `training/`,
(4) what a run produces, and (5) every config parameter + how to add your own task.

---

## 1. Quick start

Install the environment (repo root `bash setup.sh`), get some **preprocessed** data —
download a released task, or run [preprocessing](../preprocess/README.md) on your own
recordings — then:

```bash
# serve_bread, the released HumanEgo recipe
python -m training.FlowMatchingTrainer --task serve_bread --use_cfg --job HumanEgo
```

`--task` picks the data + config folder, `--job` picks the YAML
(`cfg/training/serve_bread/HumanEgo.yaml`). Outputs go to `runs/serve_bread/HumanEgo/`.

---

## 2. Data it expects

Training consumes the **preprocessing output** (the
[preprocessing](../preprocess/README.md) step), one folder per recording:

```
data/<task>/aria/
└── mps_<task>_<id>_vrs/
    └── preprocess/
        └── all_data/
            ├── 00000/   training_data.json + rgb*.png / mask*.png
            ├── 00001/   ...
            └── ...
```

Each `training_data.json` is the per-frame target written by preprocessing
(camera / hand / object SE(3) poses, grasp, and image paths — see the
[preprocess output reference](../preprocess/README.md#44-training_datajson-schema)).
The dataloader collects every `all_data/<idx>/training_data.json` across all training
sessions, plus the image variant named by `img_name`.

**Train / eval split (`data_sources` mode).** With `data_sources: {aria: N}` the
trainer auto-discovers `data/<task>/aria/mps_<task>_*_vrs`, **holds out recording
`000` for evaluation**, and trains on the next `N`. So you need **at least 2
recordings** (one eval + one train); `eval_source` chooses which source supplies the
held-out session. (You can also pass explicit `--train_data`/`--eval_data` session
lists instead.)

---

## 3. Files in `training/`

| File | What it is |
|------|-----------|
| `FlowMatchingTrainer.py` | **Entry point** — CLI, config resolution, the train/eval loop, checkpointing. Run with `python -m training.FlowMatchingTrainer`. |
| `FlowMatchingModel.py` | The **policy network** — a flow-matching decoder over ICTs, with optional region-aware attention, point-cloud injection, and the auxiliary co-training heads. |
| `FlowMatchingDataloader.py` | Builds per-frame **samples** from `training_data.json`: the image(s), ICTs (hands + objects), and future-horizon targets. Implements the paradigm ablations (frame / centric / action modes, dual-hand, augmentations). |
| `FlowMatchingEvaluator.py` | Teacher-forced **visual evaluator** — renders GT-vs-prediction trajectory videos during training. |

---

## 4. What a run produces

Everything lands in `runs/<task>/<job>/` (or `runs/<task>/<exp>/<job>/` with `--exp`):

| File | Meaning |
|------|---------|
| `latest.pt` | Checkpoint (model + optimizer + epoch). Training **auto-resumes** from it if present. |
| `dataset_stats.json` | Normalization statistics — computed once and cached. |
| `config.json` | The fully-resolved config used for the run. |
| `train_curve.png`, `eval_curve.png` | Training / evaluation loss curves over epochs. |
| `eval_snapshots/eval_ep_*.json` | Per-epoch eval metrics. |
| `eval_render/epoch_*/` | GT-vs-pred visualization videos (every `vis_eval_every` epochs). |

---

## 5. Configuration

### 5.1 How a config is resolved

The trainer starts from the defaults in `TrainConfig` (top of `FlowMatchingTrainer.py`),
then applies, in order:

1. **YAML** — with `--use_cfg` it loads `cfg/training/<task>/<job>.yaml` (or
   `cfg/training/<task>/<exp>/<job>.yaml` when `--exp` is given). An explicit
   `--config /absolute/path/to/config.yaml` takes precedence, which lets an
   external orchestrator keep its resolved config outside this checkout.
2. **CLI flags** — anything you pass (e.g. `--epochs 200 --lr 5e-5`) overrides the
   YAML. Most `TrainConfig` fields have a matching flag; the data-source fields
   (`data_sources`, `data_root`, `eval_source`) are set in the YAML, not on the CLI.

The default run directory is `runs/<task>/<job>/`, and `--task` selects which data
to load (`data/<task>/...`). Use `--data_root /absolute/data/root` and
`--out_dir /absolute/run/path` to keep datasets and artifacts outside this
source checkout.

```bash
python -m training.FlowMatchingTrainer --task <task> --use_cfg --job <job> [--exp <group>] \
    [--epochs N] [--lr 1e-4] [--data_num K] ...
```

### 5.2 Parameter reference

> The defaults below are the `TrainConfig` fallbacks. The released
> `HumanEgo.yaml` overrides several of them (loss weights, paradigm flags, AMP, …) —
> **that file is the canonical recipe**; copy it rather than starting from scratch.

**Data & split**

| Key | Default | Meaning |
|-----|---------|---------|
| `data_sources` | `null` | `{aria: N}` — auto-discover sessions per source; hold out `000` for eval, train on the next N. |
| `eval_source` | `aria` | Which source supplies the held-out eval session. |
| `data_num` | `null` | Hard cap on the number of training sessions (applied after the split). |
| `task` | `serve_bread` | Data + config folder; set by `--task`. |
| `data_root` | `./data` | Dataset root directory. |

**Optimization & schedule**

| Key | Default | Meaning |
|-----|---------|---------|
| `epochs` | 400 | Training epochs. |
| `batch_size` | 32 | Batch size. |
| `lr` | 1e-4 | AdamW learning rate. |
| `weight_decay` | 0.01 | AdamW weight decay. |
| `grad_clip` | 1.0 | Gradient-norm clip. |
| `use_lr_schedule` | False | Cosine LR schedule with warmup. |
| `warmup_steps` / `min_lr_ratio` | 200 / 0.05 | Warmup length; LR floor as a fraction of `lr`. |
| `use_amp` | False | Mixed-precision (AMP) training. |
| `use_ema` / `ema_decay` | True / 0.999 | Keep an exponential-moving-average copy of the weights. |

**Policy I/O & horizon**

| Key | Default | Meaning |
|-----|---------|---------|
| `pred_horizon` | 50 | Number of future steps the policy predicts (the action-chunk length). |
| `image_size` | [240, 320] | Input image size (H, W). |
| `img_name` | `rgb_WoArm_WArmObjKpts.png` | Which preprocessed image variant to feed; set to `None` for state-only (no vision). |
| `single_hand` / `single_hand_side` | False / "right" | One-handed vs bimanual; which hand when single. |
| `max_ict` | 8 | Max number of ICTs (hands + objects). |
| `hand_tracking_method` | `aria_mps` | Which hand source to read from `training_data.json`. |

**Paradigm (model inductive biases)**

| Key | Default | Meaning |
|-----|---------|---------|
| `centric_mode` | `object_centric` | Reference-frame origin: object-centric vs `ego_centric`. |
| `frame_mode` | `anchor_frame` | Predict relative to an anchor object (`anchor_frame`) vs the `camera_frame`. |
| `action_mode` | `absolute` | Predict absolute poses vs `delta` steps. |
| `use_region_attn` | False | Learnable region-aware ("spotlight") attention bias. |
| `use_pcd_features` | False | Inject explicit 3D point-cloud features. |
| `use_ot_cfm` | False | Optimal-transport conditional flow matching (straighter flows). |

**Auxiliary co-training** (each adds a head + a loss term)

| Key | Default | Meaning |
|-----|---------|---------|
| `use_aux_obj_dynamics` | False | Jointly model object dynamics. |
| `use_aux_visual_foresight` | False | Predict future 2D spatial heatmaps. |
| `use_aux_temporal_contrastive` | False | Predict future ICTs in latent space. |

**Loss weights**

| Key | Default | Meaning |
|-----|---------|---------|
| `w_flow` | 3.0 | Flow-matching velocity loss. |
| `w_pos` / `w_rot` | 2.0 / 1.0 | Hand position / rotation. |
| `w_g` | 10.0 | Grasp. |
| `w_done` | 5.0 | Done/finished flag. |
| `w_foresight` / `w_contrastive` | 1.0 / 1.0 | Weights for the two aux heads above. |

**Model architecture**

| Key | Default | Meaning |
|-----|---------|---------|
| `patch_size` | 16 | Vision patch size. |
| `vision_embed_dim` | 384 | Vision / token embedding dim. |
| `num_decoder_layers` / `num_heads` | 6 / 8 | Transformer decoder depth / attention heads. |
| `mlp_ratio` / `dropout` | 4.0 / 0.05 | MLP expansion ratio; dropout. |

**Flow-matching inference & eval**

| Key | Default | Meaning |
|-----|---------|---------|
| `num_inference_steps` | 10 | Flow-integration steps at sampling time. |
| `eval_every` / `vis_eval_every` | 1 / 50 | Run eval / render eval videos every N epochs. |

**Augmentations** — `enable_augmentation` is the master toggle, with per-type switches
`enable_aug_img`, `enable_aug_rrc` (random-resized-crop), `enable_aug_target_jittering`,
`enable_aug_cutout`, `enable_aug_temporal_stride`, `enable_aug_interpolation`.

**Legacy-compat** (match the released recipe; leave as set in `HumanEgo.yaml` unless you
know you want to change them): `use_pre_norm`, `use_ctx_norm`, `use_done_in_flow`,
`use_legacy_image_loading`, `use_legacy_rng`.

For the exact defaults, read `TrainConfig` in `FlowMatchingTrainer.py`.

### 5.3 Adding your own config

To train a policy on **your own task**:

1. **Preprocess** your recordings (see [preprocessing](../preprocess/README.md)) so you
   have `data/<your_task>/aria/mps_<your_task>_*_vrs/preprocess/all_data/…`. You need
   **≥2 recordings** (one is held out for eval).
2. **Create** `cfg/training/<your_task>/HumanEgo.yaml` — easiest is to copy
   `cfg/training/serve_bread/HumanEgo.yaml` and adjust:
   - `data_sources: {aria: N}` — set `N` to how many training recordings you have.
   - `single_hand` / `single_hand_side` — `True` / `"right"` for one-handed tasks,
     `False` for bimanual.
   - loss weights / paradigm flags only if your task needs them; otherwise keep the
     released defaults.
3. **Train:**
   ```bash
   python -m training.FlowMatchingTrainer --task <your_task> --use_cfg --job HumanEgo
   ```
4. **Watch** `runs/<your_task>/HumanEgo/` — `eval_curve.png` and the
   `eval_render/epoch_*/` videos show progress; training auto-resumes from `latest.pt`
   if interrupted.

> Smoke test: add `--epochs 5 --data_num 1` to confirm the data loads and a step runs
> before committing to a full run.
