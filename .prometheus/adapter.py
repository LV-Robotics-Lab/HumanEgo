#!/usr/bin/env python3
"""Hardware-free Prometheus adapter for HumanEgo policy training.

Planning imports no model or hardware package. Native execution uses an argv
array with ``shell=False`` and directs every generated artifact outside this
source checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

import yaml


ROOT = Path(__file__).resolve().parents[1]
CAPABILITIES_PATH = Path(__file__).with_name("capabilities.json")
REQUIRED_PATHS = (
    Path("requirements.txt"),
    Path("setup.sh"),
    Path("training/FlowMatchingTrainer.py"),
    Path("training/FlowMatchingDataloader.py"),
    Path("training/FlowMatchingModel.py"),
)
ACTION_SPACES = {
    "abs_dual_hand_pose_rot6d_grasp": "absolute",
    "delta_dual_hand_pose_rot6d_grasp": "delta",
}
ACTION_FRAMES = {"camera": "camera_frame", "anchor": "anchor_frame"}
ACTION_LAYOUT = (
    ("left_position_xyz", 3),
    ("right_position_xyz", 3),
    ("left_rotation_6d", 6),
    ("right_rotation_6d", 6),
    ("left_grasp", 1),
    ("right_grasp", 1),
)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _identifier(value: Any, label: str) -> str:
    selected = _string(value, label)
    if not _IDENTIFIER_RE.fullmatch(selected):
        raise ValueError(f"{label} must be a path-free identifier")
    return selected


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _digest(value: Any, label: str) -> str:
    selected = _string(value, label).lower()
    if not _DIGEST_RE.fullmatch(selected):
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    return selected


def _feature_size(value: Any, label: str) -> int:
    feature = _mapping(value, label)
    shape = _list(feature.get("shape"), f"{label}.shape")
    if not shape or any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in shape
    ):
        raise ValueError(f"{label}.shape must contain positive integers")
    size = 1
    for item in shape:
        size *= item
    return size


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(_mapping(payload, label))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _external_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == ROOT or resolved.is_relative_to(ROOT):
        raise ValueError(f"{label} must be outside the immutable source checkout")
    return resolved


def _file_uri_path(uri: str, label: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValueError(f"{label} must be an absolute local file:// URI")
    path = Path(unquote(parsed.path)).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must resolve to an absolute path")
    resolved = _external_path(path, label)
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {resolved}")
    return resolved


def capabilities() -> dict[str, Any]:
    payload = json.loads(CAPABILITIES_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != "prometheus_source_adapter_v1":
        raise RuntimeError("unsupported Prometheus source-adapter schema")
    declared = _mapping(payload.get("capabilities"), "capabilities")
    if declared.get("hardware_rollout_authorized") is not False:
        raise RuntimeError("HumanEgo training must not authorize hardware rollout")
    if declared.get("resume") != "partial_state_non_bit_exact":
        raise RuntimeError("HumanEgo resume must remain conservatively declared")
    return payload


def doctor() -> dict[str, Any]:
    declared = capabilities()
    missing = [path.as_posix() for path in REQUIRED_PATHS if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"missing required HumanEgo paths: {missing}")
    trainer = (ROOT / "training/FlowMatchingTrainer.py").read_text(encoding="utf-8")
    for flag in ("--config", "--data_root", "--out_dir"):
        if flag not in trainer:
            raise RuntimeError(f"native trainer is missing external-path flag {flag}")
    return {
        "ok": True,
        "policy_id": declared["policy_id"],
        "checked_paths": [path.as_posix() for path in REQUIRED_PATHS],
        "python": "3.11",
        "environment_reproducible": False,
        "environment_note": "requirements.txt is a dependency specification, not a lockfile",
        "imports_model_stack": False,
        "hardware_touched": False,
    }


def _validate_action(action: Mapping[str, Any]) -> tuple[str, str, int]:
    action_space = _string(action.get("space"), "action.space").lower()
    if action_space not in ACTION_SPACES:
        raise ValueError(
            f"unsupported action.space={action_space!r}; expected one of {sorted(ACTION_SPACES)}"
        )
    frame = _string(action.get("frame"), "action.frame").lower()
    if frame not in ACTION_FRAMES:
        raise ValueError("HumanEgo action.frame must be camera or anchor")
    action_dim = _positive_int(action.get("dim"), "action.dim")
    if action_dim != 20:
        raise ValueError("HumanEgo bimanual hand-pose training requires action.dim=20")
    features = _list(action.get("features"), "action.features")
    measured: list[tuple[str, int]] = []
    for index, item in enumerate(features):
        feature = _mapping(item, f"action.features[{index}]")
        name = _string(feature.get("name"), f"action.features[{index}].name").rsplit(".", 1)[-1]
        measured.append((name, _feature_size(feature, f"action.features[{index}]")))
    if tuple(measured) != ACTION_LAYOUT:
        raise ValueError(
            "HumanEgo requires the explicit modality-major 20D action layout "
            f"{list(ACTION_LAYOUT)}, got {measured}"
        )
    horizon = _positive_int(action.get("horizon"), "action.horizon")
    return ACTION_SPACES[action_space], ACTION_FRAMES[frame], horizon


def validate_dataset_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "prometheus_training_dataset_v1":
        raise ValueError("unsupported training dataset schema")
    dataset = _mapping(payload.get("dataset"), "dataset")
    if _string(dataset.get("format"), "dataset.format").lower() != "humanego_preprocessed_v1":
        raise ValueError("HumanEgo accepts only humanego_preprocessed_v1")
    dataset_root = _file_uri_path(_string(dataset.get("uri"), "dataset.uri"), "dataset.uri")
    dataset_digest = _digest(dataset.get("digest"), "dataset.digest")

    robot = _mapping(payload.get("robot"), "robot")
    schema_sources = _list(robot.get("schema_sources"), "robot.schema_sources")
    if not schema_sources:
        raise ValueError("robot.schema_sources must be non-empty")
    schema_digests = [
        _digest(
            _mapping(item, f"robot.schema_sources[{index}]").get("digest"),
            f"robot.schema_sources[{index}].digest",
        )
        for index, item in enumerate(schema_sources)
    ]

    observation = _mapping(payload.get("observation"), "observation")
    if _string(observation.get("color_order"), "observation.color_order").upper() != "RGB":
        raise ValueError("HumanEgo training requires RGB images")
    if not _list(observation.get("state"), "observation.state"):
        raise ValueError("HumanEgo requires ICT state features")
    images = _list(observation.get("images"), "observation.images")
    if len(images) != 1:
        raise ValueError("the current HumanEgo trainer consumes exactly one selected RGB image")
    if _list(observation.get("tactile"), "observation.tactile"):
        raise ValueError("the current HumanEgo trainer does not consume tactile observations")

    action_mode, frame_mode, horizon = _validate_action(
        _mapping(payload.get("action"), "action")
    )
    language = _mapping(payload.get("language"), "language")
    if _string(language.get("mode"), "language.mode").lower() != "none":
        raise ValueError("the current HumanEgo trainer does not consume language")
    normalization = _mapping(payload.get("normalization"), "normalization")
    if _string(normalization.get("owner"), "normalization.owner").lower() != "trainer":
        raise ValueError("HumanEgo owns and caches dataset normalization statistics")
    if _string(normalization.get("method"), "normalization.method").lower() != "none":
        raise ValueError("the dataset must be raw at this boundary; normalization is trainer-owned")
    rate_hz = _mapping(payload.get("sampling"), "sampling").get("rate_hz")
    if not isinstance(rate_hz, (int, float)) or isinstance(rate_hz, bool) or rate_hz <= 0:
        raise ValueError("sampling.rate_hz must be positive")

    options = _mapping(payload.get("humanego"), "humanego")
    task = _identifier(options.get("task"), "humanego.task")
    source_type = _identifier(options.get("source_type"), "humanego.source_type")
    image_name = _identifier(options.get("image_name"), "humanego.image_name")
    image_contract_name = _string(
        _mapping(images[0], "observation.images[0]").get("name"),
        "observation.images[0].name",
    )
    image_contract_name = image_contract_name.removeprefix("observation.images.")
    if image_contract_name != image_name:
        raise ValueError(
            "humanego.image_name must match the selected observation.images feature name"
        )
    recipe_rel = Path(_string(options.get("recipe"), "humanego.recipe"))
    if recipe_rel.is_absolute() or ".." in recipe_rel.parts:
        raise ValueError("humanego.recipe must be a relative path below cfg/training")
    recipe = (ROOT / "cfg" / "training" / recipe_rel).resolve()
    training_cfg_root = (ROOT / "cfg" / "training").resolve()
    if not recipe.is_relative_to(training_cfg_root) or not recipe.is_file():
        raise ValueError("humanego.recipe must name an existing source training YAML")

    source_root = dataset_root / task / source_type
    sessions = sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir() and (path / "preprocess" / "all_data").is_dir()
    ) if source_root.is_dir() else []
    if len(sessions) < 2:
        raise ValueError(
            "HumanEgo requires at least two valid sessions: one held out and one for training"
        )
    return {
        "dataset_root": dataset_root,
        "dataset_digest": dataset_digest,
        "robot_schema_digests": schema_digests,
        "task": task,
        "source_type": source_type,
        "recipe": recipe,
        "image_name": image_name,
        "action_mode": action_mode,
        "frame_mode": frame_mode,
        "action_horizon": horizon,
        "session_count": len(sessions),
        "rate_hz": float(rate_hz),
    }


def build_resolved_config(
    dataset_contract: Path, run_dir: Path
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    output = _external_path(run_dir, "run directory")
    contract_path = dataset_contract.expanduser().resolve()
    contract = _load_yaml(contract_path, "dataset contract")
    selected = validate_dataset_contract(contract)
    config = deepcopy(_load_yaml(selected["recipe"], "HumanEgo recipe"))
    config.update(
        {
            "task": selected["task"],
            "data_root": str(selected["dataset_root"]),
            "data_sources": {
                selected["source_type"]: selected["session_count"] - 1,
            },
            "eval_source": selected["source_type"],
            "single_hand": False,
            "img_name": selected["image_name"],
            "action_mode": selected["action_mode"],
            "frame_mode": selected["frame_mode"],
            "pred_horizon": selected["action_horizon"],
        }
    )
    config["prometheus_contract"] = {
        "schema": "prometheus_humanego_resolved_v1",
        "dataset_contract": str(contract_path),
        "dataset_contract_digest": _sha256_file(contract_path),
        "dataset_digest": selected["dataset_digest"],
        "robot_schema_digests": selected["robot_schema_digests"],
        "action_dim": 20,
        "output_semantics": "dual_hand_pose_not_robot_joint_command",
        "arx_20d_to_14d_decoder": None,
        "hardware_rollout_authorized": False,
    }
    return config, output / "prometheus_humanego_config.yaml", selected


def write_resolved_config(config: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def build_native_argv(
    config_path: Path, run_dir: Path, selected: Mapping[str, Any]
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "training.FlowMatchingTrainer",
        "--config",
        str(config_path),
        "--task",
        str(selected["task"]),
        "--job",
        "prometheus",
        "--data_root",
        str(selected["dataset_root"]),
        "--out_dir",
        str(run_dir),
    ]


def _native_environment(run_dir: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HOME": str(run_dir / "cache" / "huggingface"),
            "MPLCONFIGDIR": str(run_dir / "cache" / "matplotlib"),
            "WANDB_DIR": str(run_dir / "wandb"),
            "XDG_CACHE_HOME": str(run_dir / "cache" / "xdg"),
        }
    )
    return environment


def dispatch(stage: str, dataset_contract: Path, run_dir: Path, plan: bool) -> int:
    config, config_path, selected = build_resolved_config(dataset_contract, run_dir)
    output = config_path.parent
    checkpoint = output / "latest.pt"
    if stage == "train" and checkpoint.exists():
        raise ValueError("train refuses an existing latest.pt; use the resume stage")
    if stage == "resume" and not checkpoint.is_file():
        raise FileNotFoundError(f"resume requires the native checkpoint: {checkpoint}")
    write_resolved_config(config, config_path)
    argv = build_native_argv(config_path, output, selected)
    if plan:
        print(
            json.dumps(
                {
                    "stage": stage,
                    "argv": argv,
                    "cwd": str(ROOT),
                    "resolved_config": str(config_path),
                    "resume_semantics": "partial_state_non_bit_exact",
                    "hardware_rollout_authorized": False,
                    "shell": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    subprocess.run(
        argv,
        cwd=ROOT,
        env=_native_environment(output),
        check=True,
        shell=False,
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("doctor", "train", "resume"))
    parser.add_argument("--dataset-contract", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    if args.stage != "doctor" and (args.dataset_contract is None or args.run_dir is None):
        parser.error("train/resume require --dataset-contract and --run-dir")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "doctor":
        print(json.dumps(doctor(), indent=2, sort_keys=True))
        return 0
    return dispatch(args.stage, args.dataset_contract, args.run_dir, args.plan)


if __name__ == "__main__":
    raise SystemExit(main())
