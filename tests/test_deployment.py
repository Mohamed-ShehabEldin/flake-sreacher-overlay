import hashlib
import io
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import deploy_flake_searcher as deploy


class _InterruptedResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, _size):
        raise OSError("connection dropped")


class DeploymentTests(unittest.TestCase):
    def test_platform_detection_normalizes_windows_amd64(self):
        system, machine, asset = deploy.detect_platform("Windows", "AMD64")
        self.assertEqual((system, machine), ("Windows", "x86_64"))
        self.assertEqual(asset.executable_name, "uv.exe")

    def test_unsupported_platform_has_clear_failure(self):
        with self.assertRaisesRegex(deploy.DeploymentError, "Windows ARM"):
            deploy.detect_platform("Windows", "ARM64")

    def test_intel_mac_fails_before_attempting_source_builds(self):
        asset = deploy.UV_ASSETS[("Darwin", "x86_64")]
        with patch.object(deploy, "detect_platform", return_value=("Darwin", "x86_64", asset)):
            with self.assertRaisesRegex(deploy.DeploymentError, "do not publish Intel macOS wheels"):
                deploy.preflight("runtime", preview=True)

    def test_profile_selects_the_correct_locked_extra(self):
        uv = Path("uv")
        self.assertEqual(deploy.sync_command(uv, "runtime")[-1], "detector")
        self.assertEqual(deploy.sync_command(uv, "full")[-1], "full")
        self.assertIn("--locked", deploy.sync_command(uv, "runtime"))

    def test_sam2_build_uses_the_locked_environment(self):
        configuration = tomllib.loads((deploy.PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(configuration["tool"]["uv"]["no-build-isolation-package"], ["sam-2"])
        self.assertIn("setuptools==84.0.0", configuration["project"]["optional-dependencies"]["full"])
        self.assertEqual(
            configuration["tool"]["uv"]["extra-build-dependencies"]["sam-2"],
            ["setuptools==84.0.0", "torch==2.10.0"],
        )

    def test_correct_download_is_reused_without_network(self):
        data = b"verified"
        digest = hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "asset.bin"
            destination.write_bytes(data)

            def should_not_open(*_args, **_kwargs):
                raise AssertionError("network should not be used")

            result = deploy.download_verified("https://invalid", destination, digest, opener=should_not_open)
            self.assertEqual(result.read_bytes(), data)

    def test_hash_failure_preserves_invalid_download(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "asset.bin"
            with self.assertRaisesRegex(deploy.DeploymentError, "integrity"):
                deploy.download_verified(
                    "https://example.test/asset",
                    destination,
                    hashlib.sha256(b"expected").hexdigest(),
                    opener=lambda *_args, **_kwargs: io.BytesIO(b"wrong"),
                )
            self.assertFalse(destination.exists())
            self.assertEqual((Path(folder) / "asset.bin.invalid").read_bytes(), b"wrong")

    def test_timeout_fails_without_partial_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "asset.bin"

            def timeout(*_args, **_kwargs):
                raise TimeoutError("timed out")

            with self.assertRaisesRegex(deploy.DeploymentError, "Download failed"):
                deploy.download_verified("https://example.test/asset", destination, "0" * 64, opener=timeout)
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(folder).glob("*.part")), [])

    def test_interrupted_download_leaves_no_partial_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "asset.bin"
            with self.assertRaisesRegex(deploy.DeploymentError, "connection dropped"):
                deploy.download_verified(
                    "https://example.test/asset",
                    destination,
                    "0" * 64,
                    opener=lambda *_args, **_kwargs: _InterruptedResponse(),
                )
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(folder).glob("*.part")), [])

    def test_checkpoint_reuses_valid_legacy_file_and_then_target(self):
        data = b"checkpoint"
        metadata = {
            "sam2": {
                "checkpoint": {
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size": len(data),
                    "url": "https://example.test/checkpoint",
                }
            }
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            destination = root / "assets" / "checkpoint.pt"
            legacy = root / "legacy.pt"
            legacy.write_bytes(data)
            with (
                patch.object(deploy, "CHECKPOINT_PATH", destination),
                patch.object(deploy, "LEGACY_CHECKPOINT_PATH", legacy),
                patch.object(deploy, "load_manifest", return_value=metadata),
            ):
                self.assertEqual(deploy.ensure_checkpoint(), destination)
                legacy.unlink()
                self.assertEqual(deploy.ensure_checkpoint(), destination)
            self.assertEqual(destination.read_bytes(), data)

    def test_preview_constructs_commands_without_running_them(self):
        asset = deploy.UV_ASSETS[("Darwin", "arm64")]
        with (
            patch.object(deploy, "preflight", return_value=("Darwin", "arm64", asset)),
            patch.object(deploy, "ensure_uv", return_value=Path("/managed/uv")),
            patch.object(deploy, "ensure_checkpoint") as checkpoint,
            patch.object(deploy, "verify_installation") as verify,
            patch.object(deploy, "run_checked") as run,
        ):
            deploy.setup("full", preview=True)
        checkpoint.assert_called_once_with(preview=True)
        verify.assert_called_once_with("full", preview=True)
        run.assert_not_called()

    def test_managed_python_is_platform_specific(self):
        self.assertEqual(deploy.managed_python("Windows").parts[-2:], ("Scripts", "python.exe"))
        self.assertEqual(deploy.managed_python("Darwin").parts[-2:], ("bin", "python"))

    def test_launch_uses_managed_interpreter(self):
        with tempfile.TemporaryDirectory() as folder:
            interpreter = Path(folder) / "python"
            interpreter.touch()
            with (
                patch.object(deploy, "managed_python", return_value=interpreter),
                patch.object(deploy, "read_state", return_value={"profile": "full"}),
                patch.object(deploy.subprocess, "run") as run,
            ):
                deploy.launch()
        command = run.call_args.args[0]
        self.assertEqual(command, [str(interpreter), "-m", "flake_searcher"])
        self.assertEqual(run.call_args.kwargs["cwd"], deploy.PROJECT_ROOT)

    def test_launch_rejects_partial_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            interpreter = Path(folder) / "python"
            interpreter.touch()
            with (
                patch.object(deploy, "managed_python", return_value=interpreter),
                patch.object(deploy, "read_state", return_value={}),
                patch.object(deploy.subprocess, "run") as run,
            ):
                with self.assertRaisesRegex(deploy.DeploymentError, "Setup is incomplete"):
                    deploy.launch()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
