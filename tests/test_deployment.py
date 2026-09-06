import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import deploy_flake_searcher as deploy


class _InterruptedResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size):
        raise OSError("connection dropped")


class DeploymentTests(unittest.TestCase):
    def test_platform_detection_normalizes_windows_amd64(self):
        self.assertEqual(deploy.detect_platform("Windows", "AMD64"), ("Windows", "x86_64"))

    def test_unsupported_platform_has_clear_failure(self):
        with self.assertRaisesRegex(deploy.DeploymentError, "Windows ARM"):
            deploy.detect_platform("Windows", "ARM64")

    def test_finds_conda_next_to_selected_miniconda_base_python(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "miniconda3"
            python = root / "python.exe"
            conda = root / "Scripts" / "conda.exe"
            conda.parent.mkdir(parents=True)
            python.touch()
            conda.touch()
            found = deploy.find_conda_executable(
                environ={"PATH": ""}, host_python=python, home=Path(folder) / "user"
            )
        self.assertEqual(found, conda.resolve())

    def test_missing_conda_error_contains_install_link(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(deploy.DeploymentError, "anaconda.com"):
                deploy.find_conda_executable(
                    environ={"PATH": ""},
                    host_python=Path(folder) / "python",
                    home=Path(folder) / "empty-home",
                )

    def test_conda_environment_commands_create_or_update_named_environment(self):
        conda = Path("conda")
        create = deploy.conda_environment_command(conda, False)
        update = deploy.conda_environment_command(conda, True)
        self.assertEqual(create[1:3], ["env", "create"])
        self.assertEqual(update[1:3], ["env", "update"])
        self.assertIn(deploy.CONDA_ENVIRONMENT, create)
        self.assertIn("--prune", update)

    def test_locked_dependencies_install_before_sam2_without_build_isolation(self):
        python = Path("environment-python")
        dependencies = deploy.dependency_command(python, "full")
        sam2 = deploy.sam2_install_command(python, Path("sam2.tar.gz"))
        self.assertIn("--require-hashes", dependencies)
        self.assertIn(str(deploy.requirements_path("full")), dependencies)
        self.assertIn("--no-build-isolation", sam2)
        self.assertIn("--no-deps", sam2)
        full_lock = deploy.requirements_path("full").read_text(encoding="utf-8")
        self.assertIn("setuptools==84.0.0", full_lock)
        self.assertIn("torch==2.10.0", full_lock)
        self.assertNotIn("sam-2 @", full_lock)

    def test_pyqt_qt_bundle_has_hash_locked_platform_pins(self):
        project = (deploy.PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        mac_requirement = (
            "pyqt5-qt5==5.15.19; sys_platform == 'darwin' or sys_platform == 'linux'"
        )
        windows_requirement = "pyqt5-qt5==5.15.2; sys_platform == 'win32'"
        self.assertIn('"pyqt5==5.15.11"', project)
        self.assertIn(f'"{mac_requirement}"', project)
        self.assertIn(f'"{windows_requirement}"', project)

        for profile in ("runtime", "full"):
            lock = deploy.requirements_path(profile).read_text(encoding="utf-8")
            self.assertIn("pyqt5==5.15.11 \\", lock)
            self.assertIn(
                "pyqt5-qt5==5.15.19 ; "
                "sys_platform == 'darwin' or sys_platform == 'linux' \\",
                lock,
            )
            self.assertIn(
                "pyqt5-qt5==5.15.2 ; sys_platform == 'win32' \\", lock
            )
            self.assertIn(
                "--hash=sha256:750b78e4dba6bdf1607febedc08738e318ea09e9b10aea9ff0d73073f11f6962",
                lock,
            )
            self.assertNotIn("\npyqt5-qt5==5.15.19 \\", lock)

    def test_correct_download_is_reused_without_network(self):
        data = b"verified"
        digest = hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "asset.bin"
            destination.write_bytes(data)

            def should_not_open(*_args, **_kwargs):
                raise AssertionError("network should not be used")

            result = deploy.download_verified(
                "https://invalid", destination, digest, opener=should_not_open
            )
            self.assertEqual(result.read_bytes(), data)

    def test_hash_failure_preserves_invalid_download_and_shows_manual_steps(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "asset.bin"
            with self.assertRaises(deploy.DeploymentError) as raised:
                deploy.download_verified(
                    "https://example.test/asset",
                    destination,
                    hashlib.sha256(b"expected").hexdigest(),
                    opener=lambda *_args, **_kwargs: io.BytesIO(b"wrong"),
                )
            message = str(raised.exception)
            self.assertIn("Open this link in a browser", message)
            self.assertIn(str(destination), message)
            self.assertFalse(destination.exists())
            self.assertEqual((Path(folder) / "asset.bin.invalid").read_bytes(), b"wrong")

    def test_timeout_fails_without_partial_destination_and_shows_manual_steps(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "asset.bin"

            def timeout(*_args, **_kwargs):
                raise TimeoutError("timed out")

            with self.assertRaises(deploy.DeploymentError) as raised:
                deploy.download_verified(
                    "https://example.test/asset", destination, "0" * 64, opener=timeout
                )
            self.assertIn("Save the file exactly here", str(raised.exception))
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

    def test_preview_constructs_conda_workflow_without_running_commands(self):
        conda = Path("/miniconda/bin/conda")
        with (
            patch.object(
                deploy,
                "preflight",
                return_value=("Darwin", "arm64", conda, None),
            ),
            patch.object(deploy, "ensure_sam2_source") as source,
            patch.object(deploy, "ensure_checkpoint") as checkpoint,
            patch.object(deploy, "run_checked") as run,
        ):
            deploy.setup("full", preview=True)
        source.assert_called_once_with(preview=True)
        checkpoint.assert_called_once_with(preview=True)
        run.assert_not_called()

    def test_full_setup_runs_dependencies_before_sam2(self):
        conda = Path("/miniconda/bin/conda")
        commands = []
        with tempfile.TemporaryDirectory() as folder:
            prefix = Path(folder) / "flake-searcher"
            python = prefix / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
            with (
                patch.object(
                    deploy,
                    "preflight",
                    return_value=("Darwin", "arm64", conda, None),
                ),
                patch.object(deploy, "ensure_sam2_source", return_value=deploy.SAM2_SOURCE_PATH),
                patch.object(deploy, "ensure_checkpoint"),
                patch.object(deploy, "verify_installation"),
                patch.object(deploy, "conda_environment_prefixes", return_value=(prefix,)),
                patch.object(deploy, "write_state"),
                patch.object(deploy, "run_checked", side_effect=lambda command, **_kwargs: commands.append(command)),
            ):
                deploy.setup("full")
        dependency_index = next(index for index, command in enumerate(commands) if "--require-hashes" in command)
        sam2_index = next(index for index, command in enumerate(commands) if str(deploy.SAM2_SOURCE_PATH) in command)
        self.assertLess(dependency_index, sam2_index)

    def test_windows_environment_python_is_conda_prefix_python(self):
        prefix = Path("C:/Users/QMLab/miniconda3/envs/flake-searcher")
        self.assertEqual(deploy.environment_python(prefix, "Windows"), prefix / "python.exe")

    def test_launch_uses_conda_environment_python_directly(self):
        conda = Path("/miniconda/bin/conda")
        state = {
            "environment": deploy.CONDA_ENVIRONMENT,
            "profile": "full",
            "conda": str(conda),
        }
        with tempfile.TemporaryDirectory() as folder:
            prefix = Path(folder) / "flake-searcher"
            python = prefix / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
            with (
                patch.object(deploy, "read_state", return_value=state),
                patch.object(deploy, "find_conda_executable", return_value=conda),
                patch.object(deploy, "conda_environment_prefixes", return_value=(prefix,)),
                patch.object(deploy, "run_checked") as run,
            ):
                deploy.launch()
        command = run.call_args.args[0]
        self.assertEqual(command, [str(python), "-m", "flake_searcher"])

    def test_launch_rejects_old_or_partial_installation_state(self):
        with patch.object(deploy, "read_state", return_value={"profile": "full", "uv": "0.12.7"}):
            with self.assertRaisesRegex(deploy.DeploymentError, "Conda setup is incomplete"):
                deploy.launch()

    def test_conda_environment_list_parses_json(self):
        output = json.dumps({"envs": ["C:/miniconda3", "C:/miniconda3/envs/flake-searcher"]})
        result = type("Result", (), {"stdout": output})()
        with patch.object(deploy.subprocess, "run", return_value=result):
            prefixes = deploy.conda_environment_prefixes(Path("conda.exe"))
        self.assertEqual(prefixes[-1].name, "flake-searcher")

    def test_conda_commands_use_classic_solver_for_compatibility(self):
        self.assertEqual(deploy.command_environment()["CONDA_SOLVER"], "classic")

    def test_compatible_environment_python_is_reused(self):
        result = type("Result", (), {"returncode": 0})()
        with tempfile.TemporaryDirectory() as folder:
            python = Path(folder) / "python"
            python.touch()
            with patch.object(deploy.subprocess, "run", return_value=result) as run:
                self.assertTrue(deploy.environment_python_is_compatible(python))
        self.assertIn("sys.version_info", run.call_args.args[0][-1])


if __name__ == "__main__":
    unittest.main()
