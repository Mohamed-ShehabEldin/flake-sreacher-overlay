#!/usr/bin/env python3
"""Conda-based setup, verification, and launcher for Flake Searcher Overlay.

Run this standard-library-only script with a Miniconda/Anaconda Python. It
creates or updates a dedicated Conda environment, installs the locked Python
packages, obtains verified SAM2 assets for a full setup, and always launches
the application through that environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parent
MANAGED_ROOT = PROJECT_ROOT / ".flake-searcher"
CACHE_ROOT = MANAGED_ROOT / "cache"
STATE_PATH = MANAGED_ROOT / "install-state.json"
ENVIRONMENT_FILE = PROJECT_ROOT / "environment.yml"
REQUIREMENTS_ROOT = PROJECT_ROOT / "requirements"
MANIFEST_PATH = PROJECT_ROOT / "assets" / "manifest.json"
CHECKPOINT_PATH = PROJECT_ROOT / "assets" / "checkpoints" / "sam2.1_hiera_small.pt"
LEGACY_CHECKPOINT_PATH = PROJECT_ROOT / "ai" / "auto_scan_v1" / "sam2.1_hiera_small.pt"

CONDA_ENVIRONMENT = "flake-searcher"
PYTHON_VERSION = "3.12"
MIN_HOST_PYTHON = (3, 10)
MINICONDA_URL = "https://www.anaconda.com/docs/getting-started/miniconda/install"
SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_SOURCE_URL = f"https://github.com/facebookresearch/sam2/archive/{SAM2_COMMIT}.tar.gz"
SAM2_SOURCE_SHA256 = "1f2fbfad3ffa38110368abac76c6ef9df9c282a66d5c2807bc94abf4d2fb30f8"
SAM2_SOURCE_SIZE = 55_645_345
SAM2_SOURCE_PATH = CACHE_ROOT / f"sam2-{SAM2_COMMIT}.tar.gz"


class DeploymentError(RuntimeError):
    """A setup or verification step failed with a user-actionable message."""


def normalize_machine(machine: str) -> str:
    value = machine.lower()
    if value in {"amd64", "x64", "x86_64"}:
        return "x86_64"
    if value in {"arm64", "aarch64"}:
        return "arm64"
    return value


def detect_platform(system: str | None = None, machine: str | None = None) -> tuple[str, str]:
    system = system or platform.system()
    machine = normalize_machine(machine or platform.machine())
    supported = system == "Windows" and machine == "x86_64"
    supported = supported or system == "Darwin" and machine in {"arm64", "x86_64"}
    if not supported:
        if system == "Windows":
            detail = "Only 64-bit x86 Windows 10/11 is supported; Windows ARM and 32-bit Windows are not."
        else:
            detail = "Supported systems are Windows x86-64 and macOS."
        raise DeploymentError(f"Unsupported platform: {system} {machine}. {detail}")
    return system, machine


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


def manual_download_help(url: str, destination: Path, expected_sha256: str) -> str:
    return (
        "\n\nIf automatic download is blocked:\n"
        f"1. Open this link in a browser: {url}\n"
        f"2. Save the file exactly here: {destination}\n"
        "3. Run full setup again. The deployer will verify and reuse it.\n"
        f"Expected SHA-256: {expected_sha256}"
    )


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
        print(f"Reusing verified download: {destination}")
        return destination
    preserve_invalid(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".part", dir=destination.parent
    )
    os.close(descriptor)
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
            raise DeploymentError(
                f"Downloaded file failed integrity validation: {failed_path}"
                + manual_download_help(url, destination, expected_sha256)
            )
        os.replace(temporary_path, destination)
        return destination
    except DeploymentError:
        raise
    except (OSError, TimeoutError, urllib.error.URLError) as error:
        temporary_path.unlink(missing_ok=True)
        raise DeploymentError(
            f"Automatic download failed: {error}"
            + manual_download_help(url, destination, expected_sha256)
        ) from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _conda_candidates(host_python: Path, environ: dict[str, str], home: Path) -> list[Path]:
    candidates: list[Path] = []
    configured = environ.get("CONDA_EXE")
    if configured:
        candidates.append(Path(configured))
    executable = shutil.which("conda", path=environ.get("PATH"))
    if executable:
        candidates.append(Path(executable))
    roots = [host_python.parent, host_python.parent.parent]
    roots.extend(list(host_python.parents[2:4]))
    local_app_data = Path(environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    program_data = Path(environ.get("ProgramData", "C:/ProgramData"))
    roots.extend(
        [
            home / "miniconda3",
            home / "anaconda3",
            local_app_data / "miniconda3",
            local_app_data / "anaconda3",
            program_data / "miniconda3",
            program_data / "anaconda3",
        ]
    )
    for root in roots:
        candidates.extend(
            [
                root / "Scripts" / "conda.exe",
                root / "condabin" / "conda.bat",
                root / "bin" / "conda",
            ]
        )
    return candidates


def find_conda_executable(
    explicit: Path | str | None = None,
    *,
    environ: dict[str, str] | None = None,
    host_python: Path | None = None,
    home: Path | None = None,
) -> Path:
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        raise DeploymentError(f"The requested Conda executable does not exist: {candidate}")
    environment = os.environ if environ is None else environ
    current_python = Path(sys.executable) if host_python is None else host_python
    user_home = Path.home() if home is None else home
    for candidate in _conda_candidates(current_python, environment, user_home):
        if candidate.is_file():
            return candidate.resolve()
    raise DeploymentError(
        "Conda was not found. Install Miniconda, restart VS Code, and run this script "
        f"with the Miniconda base interpreter. Instructions: {MINICONDA_URL}"
    )


def conda_environment_prefixes(conda: Path) -> tuple[Path, ...]:
    try:
        result = subprocess.run(
            [str(conda), "env", "list", "--json"],
            capture_output=True,
            text=True,
            env=command_environment(),
            check=True,
            timeout=60,
        )
        document = json.loads(result.stdout)
        values = document.get("envs")
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError
        return tuple(Path(value) for value in values)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError) as error:
        raise DeploymentError(f"Unable to read Conda environments using: {conda}") from error


def environment_prefix(prefixes: tuple[Path, ...]) -> Path | None:
    for prefix in prefixes:
        if prefix.name.casefold() == CONDA_ENVIRONMENT.casefold():
            return prefix
    return None


def conda_environment_command(conda: Path, exists: bool) -> list[str]:
    command = [
        str(conda),
        "env",
        "update" if exists else "create",
        "--name",
        CONDA_ENVIRONMENT,
        "--file",
        str(ENVIRONMENT_FILE),
    ]
    if exists:
        command.append("--prune")
    return command


def command_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PYTHONNOUSERSITE": "1",
            "SAM2_BUILD_CUDA": "0",
            "CONDA_SOLVER": "classic",
        }
    )
    return environment


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
    except KeyboardInterrupt as error:
        raise DeploymentError("Setup was interrupted. It is safe to run the same setup again.") from error
    except (OSError, subprocess.CalledProcessError) as error:
        raise DeploymentError(f"Command failed: {' '.join(command)}") from error


def requirements_path(profile: str) -> Path:
    if profile not in {"runtime", "full"}:
        raise DeploymentError(f"Unknown setup profile: {profile}")
    return REQUIREMENTS_ROOT / f"{profile}.lock.txt"


def dependency_command(python: Path, profile: str) -> list[str]:
    return [
        str(python),
        "-m",
        "pip",
        "install",
        "--require-hashes",
        "--requirement",
        str(requirements_path(profile)),
    ]


def sam2_install_command(python: Path, source: Path = SAM2_SOURCE_PATH) -> list[str]:
    return [
        str(python),
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-build-isolation",
        str(source),
    ]


def project_install_command(python: Path) -> list[str]:
    return [
        str(python),
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-build-isolation",
        "--editable",
        str(PROJECT_ROOT),
    ]


def load_manifest() -> dict:
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DeploymentError(f"Cannot read asset manifest: {MANIFEST_PATH}") from error


def ensure_sam2_source(*, preview: bool = False) -> Path:
    if asset_is_valid(SAM2_SOURCE_PATH, SAM2_SOURCE_SHA256, SAM2_SOURCE_SIZE):
        print(f"Reusing verified SAM2 source: {SAM2_SOURCE_PATH}")
        return SAM2_SOURCE_PATH
    if preview:
        print(f"Would download and verify pinned SAM2 source: {SAM2_SOURCE_URL}")
        return SAM2_SOURCE_PATH
    return download_verified(
        SAM2_SOURCE_URL,
        SAM2_SOURCE_PATH,
        SAM2_SOURCE_SHA256,
        SAM2_SOURCE_SIZE,
        timeout=900,
    )


def ensure_checkpoint(*, preview: bool = False) -> Path:
    metadata = load_manifest()["sam2"]["checkpoint"]
    expected_hash = metadata["sha256"]
    expected_size = metadata["size"]
    if asset_is_valid(CHECKPOINT_PATH, expected_hash, expected_size):
        print(f"Reusing verified SAM2 checkpoint: {CHECKPOINT_PATH}")
        return CHECKPOINT_PATH
    if preview:
        source = (
            LEGACY_CHECKPOINT_PATH
            if asset_is_valid(LEGACY_CHECKPOINT_PATH, expected_hash, expected_size)
            else metadata["url"]
        )
        print(f"Would install and verify SAM2 checkpoint from: {source}")
        return CHECKPOINT_PATH
    preserve_invalid(CHECKPOINT_PATH)
    if asset_is_valid(LEGACY_CHECKPOINT_PATH, expected_hash, expected_size):
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=CHECKPOINT_PATH.name + ".", suffix=".part", dir=CHECKPOINT_PATH.parent
        )
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
    return download_verified(
        metadata["url"], CHECKPOINT_PATH, expected_hash, expected_size, timeout=900
    )


def check_host_python() -> None:
    if sys.version_info < MIN_HOST_PYTHON:
        raise DeploymentError(
            f"Python {MIN_HOST_PYTHON[0]}.{MIN_HOST_PYTHON[1]} or newer is required to run "
            "the deployer. Select the Miniconda base interpreter in VS Code and try again."
        )


def preflight(
    profile: str, *, conda_hint: Path | str | None = None, preview: bool = False
) -> tuple[str, str, Path, Path | None]:
    check_host_python()
    system, machine = detect_platform()
    if system == "Darwin" and machine == "x86_64":
        raise DeploymentError(
            "The pinned TensorFlow and PyTorch versions do not publish Intel macOS wheels. "
            "Use Apple Silicon macOS or Windows x86-64 for this release."
        )
    conda = find_conda_executable(conda_hint)
    if not ENVIRONMENT_FILE.is_file():
        raise DeploymentError(f"Conda environment definition is missing: {ENVIRONMENT_FILE}")
    lock = requirements_path(profile)
    if not lock.is_file():
        raise DeploymentError(f"Locked dependency file is missing: {lock}")
    required_bytes = 10 * 1024**3 if profile == "full" else 5 * 1024**3
    free_bytes = shutil.disk_usage(PROJECT_ROOT).free
    if free_bytes < required_bytes and not preview:
        raise DeploymentError(
            f"At least {required_bytes // 1024**3} GB free space is required; "
            f"only {free_bytes / 1024**3:.1f} GB is available."
        )
    prefix = environment_prefix(conda_environment_prefixes(conda))
    return system, machine, conda, prefix


def verification_commands(python: Path, profile: str) -> list[list[str]]:
    commands = [
        [
            str(python),
            "-c",
            "import cv2, numpy, PIL, pyautogui, serial, PyQt5; import flake_searcher.main_window",
        ],
        [
            str(python),
            "-c",
            (
                "from tensorflow import keras; from flake_searcher.paths import MODELS_ROOT; "
                "files=sorted(MODELS_ROOT.glob('*.h5')); assert len(files)==6; "
                "[keras.models.load_model(str(p), compile=False) for p in files]"
            ),
        ],
    ]
    if profile == "full":
        commands.append(
            [
                str(python),
                "-c",
                "import sklearn, tqdm, torch, torchvision, sam2; from flake_searcher.training.sam2_predictor import FastSAMPredictor",
            ]
        )
    return commands


def read_state() -> dict:
    if not STATE_PATH.is_file():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_state(profile: str, system: str, machine: str, conda: Path, prefix: Path) -> None:
    MANAGED_ROOT.mkdir(parents=True, exist_ok=True)
    data = {
        "profile": profile,
        "platform": system,
        "architecture": machine,
        "python": PYTHON_VERSION,
        "environment": CONDA_ENVIRONMENT,
        "environment_prefix": str(prefix),
        "conda": str(conda),
        "sam2_commit": SAM2_COMMIT if profile == "full" else None,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="install-state.", suffix=".tmp", dir=MANAGED_ROOT
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_path, STATE_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)


def verify_installation(
    profile: str | None = None,
    *,
    conda_hint: Path | str | None = None,
    preview: bool = False,
) -> None:
    state = read_state()
    if profile is None:
        if not state or state.get("environment") != CONDA_ENVIRONMENT:
            raise DeploymentError("Conda setup has not completed successfully. Run setup first.")
        profile = state["profile"]
    conda = find_conda_executable(conda_hint or state.get("conda"))
    prefix = environment_prefix(conda_environment_prefixes(conda))
    if prefix is None:
        raise DeploymentError(f"Conda environment '{CONDA_ENVIRONMENT}' does not exist. Run setup first.")
    if preview:
        print(f"Would verify the {profile} installation in Conda environment: {CONDA_ENVIRONMENT}")
        return
    if profile == "full":
        metadata = load_manifest()["sam2"]["checkpoint"]
        if not asset_is_valid(CHECKPOINT_PATH, metadata["sha256"], metadata["size"]):
            raise DeploymentError("The SAM2 checkpoint is missing or invalid. Run full setup again.")
    python = environment_python(prefix)
    if not python.is_file():
        raise DeploymentError(f"Conda environment Python is missing: {python}")
    for command in verification_commands(python, profile):
        run_checked(command, timeout=300, cwd=Path(tempfile.gettempdir()))
    print(f"Verified {profile} installation successfully.")


def environment_python(prefix: Path, system: str | None = None) -> Path:
    current_system = system or platform.system()
    return prefix / ("python.exe" if current_system == "Windows" else "bin/python")


def environment_python_is_compatible(python: Path) -> bool:
    if not python.is_file():
        return False
    expected = tuple(int(part) for part in PYTHON_VERSION.split("."))
    try:
        result = subprocess.run(
            [
                str(python),
                "-c",
                f"import sys; raise SystemExit(0 if sys.version_info[:2] == {expected!r} else 1)",
            ],
            env=command_environment(),
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def print_vscode_instructions(prefix: Path, system: str) -> None:
    interpreter = environment_python(prefix, system)
    print(
        "\nVS Code interpreter:\n"
        f"  {interpreter}\n"
        "To use the Run button: press Ctrl+Shift+P, choose 'Python: Select Interpreter', "
        "then choose or enter the path above.\n"
        "You can also use option 4 in this deployer; it always launches with the correct Conda environment."
    )


def setup(
    profile: str, *, conda_hint: Path | str | None = None, preview: bool = False
) -> None:
    system, machine, conda, prefix = preflight(
        profile, conda_hint=conda_hint, preview=preview
    )
    print(f"Platform: {system} {machine}")
    print(f"Conda: {conda}")
    print(f"Environment: {CONDA_ENVIRONMENT}")
    print(f"Installation: {profile}")
    existing_python = environment_python(prefix, system) if prefix is not None else None
    reuse_environment = existing_python is not None and environment_python_is_compatible(existing_python)
    conda_command = conda_environment_command(conda, prefix is not None)
    if preview:
        preview_prefix = prefix if reuse_environment else Path(f"<{CONDA_ENVIRONMENT}-environment>")
        preview_python = environment_python(preview_prefix, system)
        commands = [] if reuse_environment else [conda_command]
        commands.append(dependency_command(preview_python, profile))
        if profile == "full":
            ensure_sam2_source(preview=True)
            commands.append(sam2_install_command(preview_python))
        commands.append(project_install_command(preview_python))
        for command in commands:
            print("Would run:", " ".join(command))
        if profile == "full":
            ensure_checkpoint(preview=True)
        print("Preview complete; no files were changed.")
        return
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if reuse_environment:
        refreshed_prefix = prefix
        print(f"Reusing compatible Conda environment: {refreshed_prefix}")
    else:
        run_checked(conda_command)
        refreshed_prefix = environment_prefix(conda_environment_prefixes(conda))
    if refreshed_prefix is None:
        raise DeploymentError("Conda reported success but the new environment could not be found.")
    python = environment_python(refreshed_prefix, system)
    if not python.is_file():
        raise DeploymentError(f"Conda environment Python is missing: {python}")
    run_checked(dependency_command(python, profile))
    if profile == "full":
        source = ensure_sam2_source()
        run_checked(sam2_install_command(python, source))
        ensure_checkpoint()
    run_checked(project_install_command(python))
    verify_installation(profile, conda_hint=conda)
    write_state(profile, system, machine, conda, refreshed_prefix)
    print("\nSetup complete.")
    print_vscode_instructions(refreshed_prefix, system)


def launch(*, conda_hint: Path | str | None = None, preview: bool = False) -> None:
    state = read_state()
    if not state or state.get("environment") != CONDA_ENVIRONMENT:
        raise DeploymentError("Conda setup is incomplete. Run setup successfully before launching.")
    conda = find_conda_executable(conda_hint or state.get("conda"))
    prefix = environment_prefix(conda_environment_prefixes(conda))
    if prefix is None:
        raise DeploymentError(f"Conda environment '{CONDA_ENVIRONMENT}' is missing. Run setup again.")
    python = environment_python(prefix)
    if not python.is_file():
        raise DeploymentError(f"Conda environment Python is missing: {python}")
    command = [str(python), "-m", "flake_searcher"]
    if preview:
        print("Would launch:", " ".join(command))
        return
    run_checked(command, timeout=24 * 60 * 60)


def interactive_menu(*, conda_hint: Path | str | None = None) -> None:
    options = {
        "1": lambda: setup("full", conda_hint=conda_hint),
        "2": lambda: setup("runtime", conda_hint=conda_hint),
        "3": lambda: verify_installation(conda_hint=conda_hint),
        "4": lambda: launch(conda_hint=conda_hint),
    }
    while True:
        print(
            "\nFlake Searcher Overlay\n"
            "1. Create/update the full Conda environment (recommended)\n"
            "2. Create/update the microscope runtime Conda environment only\n"
            "3. Verify the installation\n"
            "4. Launch Flake Searcher Overlay in its Conda environment\n"
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
    parser.add_argument("--setup", choices=("full", "runtime"), help="create or update the Conda environment")
    parser.add_argument("--verify", action="store_true", help="verify the Conda installation")
    parser.add_argument("--launch", action="store_true", help="launch in the Conda environment")
    parser.add_argument("--preview", action="store_true", help="show actions without changing files")
    parser.add_argument("--conda", type=Path, help="path to conda.exe/conda if automatic detection fails")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        if args.setup:
            setup(args.setup, conda_hint=args.conda, preview=args.preview)
        elif args.verify:
            verify_installation(conda_hint=args.conda, preview=args.preview)
        elif args.launch:
            launch(conda_hint=args.conda, preview=args.preview)
        elif args.preview:
            setup("full", conda_hint=args.conda, preview=True)
        else:
            interactive_menu(conda_hint=args.conda)
    except DeploymentError as error:
        print(f"Setup error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
