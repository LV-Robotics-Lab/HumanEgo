from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / ".prometheus" / "adapter.py"
SPEC = importlib.util.spec_from_file_location("humanego_prometheus_adapter", ADAPTER_PATH)
assert SPEC is not None and SPEC.loader is not None
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


def _contract(dataset_root: Path) -> dict:
    return {
        "schema": "prometheus_training_dataset_v1",
        "dataset": {
            "id": "humanego-dual-hand-test",
            "format": "humanego_preprocessed_v1",
            "uri": dataset_root.as_uri(),
            "digest": "a" * 64,
        },
        "robot": {
            "id": "unmapped-dual-hand-target",
            "schema_sources": [
                {"name": "test", "uri": "file:///contract.yaml", "digest": "b" * 64}
            ],
        },
        "sampling": {"rate_hz": 30.0},
        "observation": {
            "color_order": "RGB",
            "state": [
                {"name": "observation.state.ict", "shape": [8, 29], "dtype": "float32", "unit": "mixed"}
            ],
            "images": [
                {"name": "observation.images.rgb_WoArm_WArmObjKpts.png", "shape": [240, 320, 3], "dtype": "uint8"}
            ],
            "tactile": [],
        },
        "action": {
            "space": "abs_dual_hand_pose_rot6d_grasp",
            "frame": "camera",
            "dim": 20,
            "horizon": 50,
            "features": [
                {"name": "action.left_position_xyz", "shape": [3], "dtype": "float32", "unit": "m"},
                {"name": "action.right_position_xyz", "shape": [3], "dtype": "float32", "unit": "m"},
                {"name": "action.left_rotation_6d", "shape": [6], "dtype": "float32", "unit": "unitless"},
                {"name": "action.right_rotation_6d", "shape": [6], "dtype": "float32", "unit": "unitless"},
                {"name": "action.left_grasp", "shape": [1], "dtype": "float32", "unit": "normalized"},
                {"name": "action.right_grasp", "shape": [1], "dtype": "float32", "unit": "normalized"},
            ],
        },
        "language": {"mode": "none"},
        "normalization": {"method": "none", "owner": "trainer"},
        "humanego": {
            "task": "serve_bread",
            "source_type": "aria",
            "recipe": "serve_bread/HumanEgo.yaml",
            "image_name": "rgb_WoArm_WArmObjKpts.png",
        },
    }


class PrometheusAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.dataset = self.base / "dataset"
        for name in ("session_000", "session_001"):
            (self.dataset / "serve_bread" / "aria" / name / "preprocess" / "all_data").mkdir(parents=True)
        self.contract_path = self.base / "contract.yaml"
        self.contract_path.write_text(yaml.safe_dump(_contract(self.dataset)), encoding="utf-8")
        self.run_dir = self.base / "run"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_doctor_is_hardware_free_and_conservative(self) -> None:
        report = ADAPTER.doctor()
        self.assertTrue(report["ok"])
        self.assertFalse(report["hardware_touched"])
        self.assertFalse(report["imports_model_stack"])
        self.assertFalse(report["environment_reproducible"])
        caps = ADAPTER.capabilities()
        self.assertEqual(caps["capabilities"]["resume"], "partial_state_non_bit_exact")
        self.assertFalse(caps["capabilities"]["hardware_rollout_authorized"])
        self.assertFalse(caps["stages"]["serve"])
        self.assertFalse(caps["stages"]["export"])

    def test_plan_maps_to_native_trainer_and_external_paths(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ADAPTER_PATH),
                "train",
                "--dataset-contract",
                str(self.contract_path),
                "--run-dir",
                str(self.run_dir),
                "--plan",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        plan = json.loads(result.stdout)
        self.assertEqual(plan["argv"][1:3], ["-m", "training.FlowMatchingTrainer"])
        self.assertIn(str(self.dataset.resolve()), plan["argv"])
        self.assertIn(str(self.run_dir.resolve()), plan["argv"])
        self.assertFalse(plan["shell"])
        self.assertFalse(plan["hardware_rollout_authorized"])
        resolved = yaml.safe_load((self.run_dir / "prometheus_humanego_config.yaml").read_text())
        self.assertFalse(resolved["single_hand"])
        self.assertEqual(resolved["pred_horizon"], 50)
        self.assertEqual(resolved["prometheus_contract"]["action_dim"], 20)
        self.assertIsNone(resolved["prometheus_contract"]["arx_20d_to_14d_decoder"])

    def test_rejects_non_20d_or_reordered_action(self) -> None:
        payload = _contract(self.dataset)
        payload["action"]["features"].reverse()
        with self.assertRaisesRegex(ValueError, "modality-major 20D action layout"):
            ADAPTER.validate_dataset_contract(payload)
        payload = _contract(self.dataset)
        payload["action"]["dim"] = 14
        with self.assertRaisesRegex(ValueError, "action.dim=20"):
            ADAPTER.validate_dataset_contract(payload)

    def test_rejects_source_owned_data_or_outputs(self) -> None:
        payload = _contract(ROOT)
        with self.assertRaisesRegex(ValueError, "outside the immutable source checkout"):
            ADAPTER.validate_dataset_contract(payload)
        with self.assertRaisesRegex(ValueError, "outside the immutable source checkout"):
            ADAPTER.build_resolved_config(self.contract_path, ROOT / "runs" / "bad")

    def test_resume_requires_native_checkpoint(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "latest.pt"):
            ADAPTER.dispatch("resume", self.contract_path, self.run_dir, plan=True)


if __name__ == "__main__":
    unittest.main()
