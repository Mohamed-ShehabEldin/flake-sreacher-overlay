#!/usr/bin/env python3
"""Local, repeatable setup and launcher for Flake Searcher Overlay.

This file intentionally uses only the Python standard library so it can create
the managed application environment before project dependencies are installed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parent
MANAGED_ROOT = PROJECT_ROOT / ".flake-searcher"
TOOLS_ROOT = MANAGED_ROOT / "tools"
PYTHON_ROOT = MANAGED_ROOT / "python"
VENV_ROOT = MANAGED_ROOT / "venv"
CACHE_ROOT = MANAGED_ROOT / "cache"
STATE_PATH = MANAGED_ROOT / "install-state.json"
LOCK_PATH = PROJECT_ROOT / "uv.lock"
MANIFEST_PATH = PROJECT_ROOT / "assets" / "manifest.json"
CHECKPOINT_PATH = PROJECT_ROOT / "assets" / "checkpoints" / "sam2.1_hiera_small.pt"
LEGACY_CHECKPOINT_PATH = PROJECT_ROOT / "ai" / "auto_scan_v1" / "sam2.1_hiera_small.pt"

UV_VERSION = "0.12.7"
UV_RELEASE_ROOT = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}"
PYTHON_VERSION = "3.12"
MIN_HOST_PYTHON = (3, 10)


class DeploymentError(RuntimeError):
    """A setup or verification step failed with a user-actionable message."""


@dataclass(frozen=True)
class UvAsset:
    filename: str
    sha256: str
    executable_name: str

    @property
    def url(self) -> str:
        return f"{UV_RELEASE_ROOT}/{self.filename}"


UV_ASSETS = {
    ("Darwin", "arm64"): UvAsset(
        "uv-aarch64-apple-darwin.tar.gz",
        "127ebdda7ad953cdf198e964b570ea5771b85467ea93eb7cb6d6f8e6f55408f3",
        "uv",
    ),
    ("Darwin", "x86_64"): UvAsset(
        "uv-x86_64-apple-darwin.tar.gz",
        "06b8ae1da8c2661c5434507a66f8c2b0b835933bf955b5958a9ac357a37d1959",
        "uv",
    ),
    ("Windows", "x86_64"): UvAsset(
        "uv-x86_64-pc-windows-msvc.zip",
        "bf1518af459a3915511a11fdc6e2f43ef9a2afa138b9d498eeb9642fe9d85218",
        "uv.exe",
    ),
}


def normalize_machine(machine: str) -> str:
    value = machine.lower()
    if value in {"amd64", "x64", "x86_64"}:
        return "x86_64"
    if value in {"arm64", "aarch64"}:
        return "arm64"
    return value


def detect_platform(system: str | None = None, machine: str | None = None) -> tuple[str, str, UvAsset]:
    system = system or platform.system()
    machine = normalize_machine(machine or platform.machine())
    try:
        asset = UV_ASSETS[(system, machine)]
    except KeyError as error:
        if system == "Windows":
            detail = "Only 64-bit x86 Windows 10/11 is supported. Windows ARM and 32-bit Windows are not supported."
        else:
            detail = "Supported systems are macOS (Apple Silicon or Intel) and Windows x86-64."
        raise DeploymentError(f"Unsupported platform: {system} {machine}. {detail}") from error
    return system, machine, asset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_is_valid(path: Path, expected_sha256: str, expected_size: int | None = None) -> bool:
    if not path.is_file():
        return False
    if expected_size is not None and path.stat().st_size != expected_size:
        return False
    return sha256_file(path) == expected_sha256


def invalid_path(path: Path) -> Path:
    candidate = path.with_name(path.name + ".invalid")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(path.name + f".invalid.{counter}")
        counter += 1
    return candidate


def preserve_invalid(path: Path) -> Path | None:
    if not path.exists():
        return None
    destination = invalid_path(path)
    path.replace(destination)
    print(f"Preserved invalid file as: {destination}")
    return destination


def download_verified(
    url: str,
    destination: Path,
    expected_sha256: str,
    expected_size: int | None = None,
    *,
    timeout: int = 120,
    opener=urllib.request.urlopen,
) -> Path:
    if asset_is_valid(destination, expected_sha256, expected_size):
        return destination
    preserve_invalid(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".part",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        print(f"Downloading {destination.name} …")
        with opener(url, timeout=timeout) as response, temporary_path.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)

        if not asset_is_valid(temporary_path, expected_sha256, expected_size):
            failed_path = invalid_path(destination)
            temporary_path.replace(failed_path)
            raise DeploymentError(f"Downloaded file failed integrity validation: {failed_path}")
        os.replace(temporary_path, destination)
        return destination
    except DeploymentError:
        raise
    except (OSError, TimeoutError, urllib.error.URLError) as error:
        temporary_path.unlink(missing_ok=True)
        raise DeploymentError(f"Download failed for {url}: {error}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def uv_executable(system: str | None = None) -> Path:
    current_system = system or platform.system()
    return TOOLS_ROOT / ("uv.exe" if current_system == "Windows" else "uv")


def managed_python(system: str | None = None) -> Path:
    current_system = system or platform.system()
    if current_system == "Windows":
        return VENV_ROOT / "Scripts" / "python.exe"
    return VENV_ROOT / "bin" / "python"


def command_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "UV_PROJECT_ENVIRONMENT": str(VENV_ROOT),
            "UV_PYTHON_INSTALL_DIR": str(PYTHON_ROOT),
            "UV_CACHE_DIR": str(CACHE_ROOT),
            "UV_PYTHON_INSTALL_BIN": "0",
            "SAM2_BUILD_CUDA": "0",
        }
    )
    return environment


def sync_command(uv_path: Path, profile: str) -> list[str]:
    if profile not in {"runtime", "full"}:
        raise DeploymentError(f"Unknown setup profile: {profile}")
    extra = "detector" if profile == "runtime" else "full"
    return [str(uv_path), "sync", "--locked", "--no-dev", "--python", PYTHON_VERSION, "--extra", extra]


def run_checked(command: list[str], *, timeout: int = 3600, cwd: Path = PROJECT_ROOT) -> None:
    print("Running:", " ".join(command))
    try:
        subprocess.run(
            command,
            cwd=cwd,
            env=command_environment(),
            check=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise DeploymentError(f"Command timed out after {timeout} seconds: {' '.join(command)}") from error
    except (OSError, subprocess.CalledProcessError) as error:
        raise DeploymentError(f"Command failed: {' '.join(command)}") from error


def current_uv_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        result = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    version_output = result.stdout.strip()
    return version_output == f"uv {UV_VERSION}" or version_output.startswith(f"uv {UV_VERSION} ")


def extract_uv(archive: Path, asset: UvAsset, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix="uv.", dir=destination.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        if archive.name.endswith(".zip"):
            with zipfile.ZipFile(archive) as bundle:
                members = [name for name in bundle.namelist() if Path(name).name == asset.executable_name]
                if len(members) != 1:
                    raise DeploymentError("The uv archive does not contain exactly one expected executable.")
                with bundle.open(members[0]) as source, temporary_path.open("wb") as output:
                    shutil.copyfileobj(source, output)
        else:
            with tarfile.open(archive, "r:gz") as bundle:
                members = [member for member in bundle.getmembers() if Path(member.name).name == asset.executable_name and member.isfile()]
                if len(members) != 1:
                    raise DeploymentError("The uv archive does not contain exactly one expected executable.")
                source = bundle.extractfile(members[0])
                if source is None:
                    raise DeploymentError("Unable to read the uv executable from its archive.")
                with source, temporary_path.open("wb") as output:
                    shutil.copyfileobj(source, output)

        temporary_path.chmod(temporary_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def ensure_uv(asset: UvAsset, *, preview: bool = False) -> Path:
    destination = uv_executable()
    if current_uv_is_valid(destination):
        print(f"Reusing uv {UV_VERSION}: {destination}")
        return destination
    if preview:
        print(f"Would download and verify uv {UV_VERSION}: {asset.url}")
        return destination

    preserve_invalid(destination)
    archive = TOOLS_ROOT / asset.filename
    download_verified(asset.url, archive, asset.sha256)
    try:
        extract_uv(archive, asset, destination)
    finally:
        archive.unlink(missing_ok=True)
    if not current_uv_is_valid(destination):
        preserve_invalid(destination)
        raise DeploymentError("The extracted uv executable did not report the pinned version.")
    return destination


def load_manifest() -> dict:
    try:
        with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise DeploymentError(f"Cannot read asset manifest: {MANIFEST_PATH}") from error


def ensure_checkpoint(*, preview: bool = False) -> Path:
    metadata = load_manifest()["sam2"]["checkpoint"]
    expected_hash = metadata["sha256"]
    expected_size = metadata["size"]
    if asset_is_valid(CHECKPOINT_PATH, expected_hash, expected_size):
        print(f"Reusing verified SAM2 checkpoint: {CHECKPOINT_PATH}")
        return CHECKPOINT_PATH

    if preview:
        source = LEGACY_CHECKPOINT_PATH if asset_is_valid(LEGACY_CHECKPOINT_PATH, expected_hash, expected_size) else metadata["url"]
        print(f"Would install and verify SAM2 checkpoint from: {source}")
        return CHECKPOINT_PATH

    preserve_invalid(CHECKPOINT_PATH)
    if asset_is_valid(LEGACY_CHECKPOINT_PATH, expected_hash, expected_size):
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=CHECKPOINT_PATH.name + ".", suffix=".part", dir=CHECKPOINT_PATH.parent)
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            shutil.copyfile(LEGACY_CHECKPOINT_PATH, temporary_path)
            if not asset_is_valid(temporary_path, expected_hash, expected_size):
                raise DeploymentError("Legacy checkpoint changed while it was being copied.")
            os.replace(temporary_path, CHECKPOINT_PATH)
        finally:
            temporary_path.unlink(missing_ok=True)
        print(f"Reused verified legacy checkpoint: {LEGACY_CHECKPOINT_PATH}")
        return CHECKPOINT_PATH

    return download_verified(metadata["url"], CHECKPOINT_PATH, expected_hash, expected_size, timeout=900)


def check_host_python() -> None:
    if sys.version_info < MIN_HOST_PYTHON:
        raise DeploymentError(
            f"Python {MIN_HOST_PYTHON[0]}.{MIN_HOST_PYTHON[1]} or newer is required to run this deployer. "
            "The application itself uses a separate managed Python 3.12 environment."
        )


def preflight(profile: str, *, preview: bool = False) -> tuple[str, str, UvAsset]:
    check_host_python()
    result = detect_platform()
    system, machine, _asset = result
    if system == "Darwin" and machine == "x86_64":
        raise DeploymentError(
            "The locked TensorFlow and PyTorch versions do not publish Intel macOS wheels. "
            "Use Apple Silicon macOS or Windows x86-64 for this locked release."
        )
    if not LOCK_PATH.is_file():
        raise DeploymentError(f"Locked dependency file is missing: {LOCK_PATH}")
    required_bytes = 10 * 1024**3 if profile == "full" else 5 * 1024**3
    free_bytes = shutil.disk_usage(PROJECT_ROOT).free
    if free_bytes < required_bytes and not preview:
        raise DeploymentError(
            f"At least {required_bytes // 1024**3} GB free space is required; "
            f"only {free_bytes / 1024**3:.1f} GB is available."
        )
    if free_bytes < required_bytes:
        print(
            f"Warning: setup needs about {required_bytes // 1024**3} GB free; "
            f"only {free_bytes / 1024**3:.1f} GB is currently available."
        )
    if not preview and not os.access(PROJECT_ROOT, os.W_OK):
        raise DeploymentError(f"Project directory is not writable: {PROJECT_ROOT}")
    return result


def verification_commands(python_path: Path, profile: str) -> list[list[str]]:
    commands = [
        [
            str(python_path),
            "-c",
            "import cv2, numpy, PIL, pyautogui, serial, PyQt5; import flake_searcher.main_window",
        ],
        [
            str(python_path),
            "-c",
            (
                "from tensorflow import keras; from flake_searcher.paths import MODELS_ROOT; "
                "files=sorted(MODELS_ROOT.glob('*.h5')); "
                "assert len(files)==6; [keras.models.load_model(str(p), compile=False) for p in files]"
            ),
        ],
    ]
    if profile == "full":
        commands.append(
            [
                str(python_path),
                "-c",
                "import sklearn, tqdm, torch, torchvision, sam2; from flake_searcher.training.sam2_predictor import FastSAMPredictor",
            ]
        )
    return commands


def verify_installation(profile: str | None = None, *, preview: bool = False) -> None:
    if profile is None:
        state = read_state()
        if not state:
            raise DeploymentError("Setup has not completed successfully. Run setup before verification.")
        profile = state["profile"]
    python_path = managed_python()
    if preview:
        print(f"Would verify the {profile} environment with: {python_path}")
        return
    if not python_path.is_file():
        raise DeploymentError("Managed environment is missing. Run setup first.")
    if profile == "full":
        metadata = load_manifest()["sam2"]["checkpoint"]
        if not asset_is_valid(CHECKPOINT_PATH, metadata["sha256"], metadata["size"]):
            raise DeploymentError("The SAM2 checkpoint is missing or invalid. Run full setup again.")
    environment = command_environment()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    for command in verification_commands(python_path, profile):
        print("Verifying:", command[-1].split(";")[0])
        try:
            subprocess.run(
                command,
                cwd=tempfile.gettempdir(),
                env=environment,
                check=True,
                timeout=300,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise DeploymentError(f"Installation verification failed: {' '.join(command[:2])}") from error
    print(f"Verified {profile} installation successfully.")


def write_state(profile: str, system: str, machine: str) -> None:
    MANAGED_ROOT.mkdir(parents=True, exist_ok=True)
    data = {
        "profile": profile,
        "platform": system,
        "architecture": machine,
        "python": PYTHON_VERSION,
        "uv": UV_VERSION,
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix="install-state.", suffix=".tmp", dir=MANAGED_ROOT)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_path, STATE_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_state() -> dict:
    if not STATE_PATH.is_file():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def setup(profile: str, *, preview: bool = False) -> None:
    system, machine, asset = preflight(profile, preview=preview)
    print(f"Platform: {system} {machine}")
    print(f"Installation: {profile}")
    uv_path = ensure_uv(asset, preview=preview)
    commands = [
        [str(uv_path), "python", "install", PYTHON_VERSION],
        sync_command(uv_path, profile),
    ]
    if preview:
        for command in commands:
            print("Would run:", " ".join(command))
        if profile == "full":
            ensure_checkpoint(preview=True)
        verify_installation(profile, preview=True)
        print("Preview complete; no files were changed.")
        return

    for directory in (TOOLS_ROOT, PYTHON_ROOT, CACHE_ROOT):
        directory.mkdir(parents=True, exist_ok=True)
    for command in commands:
        run_checked(command)
    if profile == "full":
        ensure_checkpoint()
    verify_installation(profile)
    write_state(profile, system, machine)
    print("Setup complete. You can launch Flake Searcher Overlay from this deployer.")


def launch(*, preview: bool = False) -> None:
    python_path = managed_python()
    command = [str(python_path), "-m", "flake_searcher"]
    if preview:
        print("Would launch:", " ".join(command))
        return
    if not python_path.is_file():
        raise DeploymentError("Managed environment is missing. Run setup first.")
    if not read_state():
        raise DeploymentError("Setup is incomplete. Run setup successfully before launching.")
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, env=command_environment(), check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise DeploymentError("Flake Searcher Overlay did not launch successfully.") from error


def interactive_menu() -> None:
    options = {
        "1": lambda: setup("full"),
        "2": lambda: setup("runtime"),
        "3": lambda: verify_installation(),
        "4": lambda: launch(),
    }
    while True:
        print(
            "\nFlake Searcher Overlay\n"
            "1. Set up or update the full installation (recommended)\n"
            "2. Set up or update microscope runtime only\n"
            "3. Verify the installation\n"
            "4. Launch Flake Searcher Overlay\n"
            "5. Exit"
        )
        choice = input("Choose 1-5: ").strip()
        if choice == "5":
            return
        action = options.get(choice)
        if action is None:
            print("Please choose a number from 1 to 5.")
            continue
        try:
            action()
        except DeploymentError as error:
            print(f"Setup error: {error}")


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", choices=("full", "runtime"), help="create or update an installation")
    parser.add_argument("--verify", action="store_true", help="verify the managed installation")
    parser.add_argument("--launch", action="store_true", help="launch with the managed interpreter")
    parser.add_argument("--preview", action="store_true", help="show setup or launch actions without changing files")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        if args.setup:
            setup(args.setup, preview=args.preview)
        elif args.verify:
            verify_installation(preview=args.preview)
        elif args.launch:
            launch(preview=args.preview)
        elif args.preview:
            setup("full", preview=True)
        else:
            interactive_menu()
    except DeploymentError as error:
        print(f"Setup error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
