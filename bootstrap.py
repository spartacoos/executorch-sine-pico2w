#!/usr/bin/env python3
"""
Bootstrap script for ExecuTorch Sine Wave Predictor on Pico 2 W.
Sets up ARM toolchain, Pico SDK, and ExecuTorch source.
"""
import subprocess
import sys
import shutil
import platform
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

# Configuration
PROJECT_DIR = Path(__file__).parent.absolute()
TOOLCHAIN_DIR = PROJECT_DIR / "toolchains"
ARM_TOOLCHAIN_VERSION = "13.3.rel1"
PICO_SDK_VERSION = "2.1.0"
EXECUTORCH_VERSION = "v1.0.0"  # Use stable release


def log_info(msg: str) -> None:
    print(f"\033[94m[INFO]\033[0m {msg}")


def log_success(msg: str) -> None:
    print(f"\033[92m[OK]\033[0m {msg}")


def log_error(msg: str) -> None:
    print(f"\033[91m[ERROR]\033[0m {msg}")


def check_prerequisites():
    """Check that required tools are installed."""
    required = ["git", "cmake", "uv"]
    for tool in required:
        if shutil.which(tool) is None:
            log_error(f"{tool} not found. Please install it first.")
            sys.exit(1)
    log_success("All prerequisites found.")


def detect_host() -> tuple[str, str]:
    """Detect host OS and architecture."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system not in ("linux", "darwin"):
        log_error(f"Unsupported OS: {system}. Only Linux and macOS are supported.")
        sys.exit(1)

    if machine in ("x86_64", "amd64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "aarch64"
    else:
        log_error(f"Unsupported architecture: {machine}")
        sys.exit(1)

    log_info(f"Detected host: {system} / {arch}")
    return system, arch


def download_file(url: str, dest: Path) -> None:
    """Download a file with progress bar."""
    log_info(f"Downloading {dest.name}...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def setup_arm_toolchain(host_os: str, host_arch: str) -> Path:
    """Download and extract ARM GNU Embedded Toolchain."""
    TOOLCHAIN_DIR.mkdir(exist_ok=True)

    # Build toolchain filename based on host
    if host_os == "darwin":
        os_name = "darwin"
        arch_name = "arm64" if host_arch == "aarch64" else "x86_64"
        toolchain_name = f"arm-gnu-toolchain-{ARM_TOOLCHAIN_VERSION}-{os_name}-{arch_name}-arm-none-eabi"
    else:
        toolchain_name = f"arm-gnu-toolchain-{ARM_TOOLCHAIN_VERSION}-{host_arch}-arm-none-eabi"

    toolchain_path = TOOLCHAIN_DIR / toolchain_name
    gcc_path = toolchain_path / "bin" / "arm-none-eabi-gcc"

    if gcc_path.exists():
        log_success(f"ARM toolchain already installed: {toolchain_path}")
        return toolchain_path

    archive_name = f"{toolchain_name}.tar.xz"
    url = f"https://developer.arm.com/-/media/Files/downloads/gnu/{ARM_TOOLCHAIN_VERSION}/binrel/{archive_name}"
    archive_path = TOOLCHAIN_DIR / archive_name

    if not archive_path.exists():
        try:
            download_file(url, archive_path)
        except requests.RequestException as e:
            log_error(f"Failed to download toolchain: {e}")
            log_info("Download manually from: https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads")
            sys.exit(1)

    log_info(f"Extracting {archive_name}...")
    with tarfile.open(archive_path, "r:xz") as tar:
        tar.extractall(TOOLCHAIN_DIR)

    archive_path.unlink()

    if not gcc_path.exists():
        log_error(f"Toolchain extraction failed: {gcc_path} not found")
        sys.exit(1)

    log_success(f"ARM toolchain installed: {toolchain_path}")
    return toolchain_path


def clone_repo(name: str, url: str, branch: str, target_path: Path) -> None:
    """Clone a git repository if it doesn't exist."""
    if target_path.exists():
        log_success(f"{name} already exists at {target_path}")
        return

    log_info(f"Cloning {name} ({branch})...")

    # ExecuTorch REQUIREMENT:
    # Buck2 build depends on non-shallow submodules (prelude)
    shallow = name.lower() not in ("executorch",)

    cmd = [
        "git", "clone",
        "-b", branch,
    ]

    if shallow:
        cmd += [
            "--depth", "1",
            "--recurse-submodules",
            "--shallow-submodules",
        ]
    else:
        # Full clone + full submodules (required)
        cmd += [
            "--recurse-submodules",
        ]

    cmd += [url, str(target_path)]

    subprocess.run(cmd, check=True)
    log_success(f"{name} cloned successfully.")


def setup_python_env():
    """Sync Python environment with uv."""
    log_info("Syncing Python environment...")
    subprocess.run(["uv", "sync"], check=True)
    log_success("Python environment ready.")


def generate_env_script(toolchain_path: Path, sdk_path: Path, et_path: Path) -> None:
    """Generate shell script to set up environment variables."""
    env_script = PROJECT_DIR / "env.sh"

    content = f"""#!/bin/bash
# Source this file to set up the build environment
# Usage: source env.sh

export PICO_SDK_PATH="{sdk_path.absolute()}"
export PICO_TOOLCHAIN_PATH="{toolchain_path.absolute()}"
export EXECUTORCH_ROOT="{et_path.absolute()}"
export PATH="{toolchain_path.absolute()}/bin:$PATH"

echo "Environment configured:"
echo "  PICO_SDK_PATH=$PICO_SDK_PATH"
echo "  EXECUTORCH_ROOT=$EXECUTORCH_ROOT"
echo "  arm-none-eabi-gcc: $(which arm-none-eabi-gcc)"
"""

    env_script.write_text(content)
    env_script.chmod(0o755)
    log_success(f"Generated {env_script} - source it before building manually")


def main():
    print("\n" + "=" * 60)
    print("  ExecuTorch Sine Wave Predictor - Bootstrap")
    print("=" * 60 + "\n")

    check_prerequisites()
    TOOLCHAIN_DIR.mkdir(exist_ok=True)

    # 1. Python Environment
    setup_python_env()

    # 2. Detect host system
    host_os, host_arch = detect_host()

    # 3. ARM Toolchain
    toolchain_path = setup_arm_toolchain(host_os, host_arch)

    # 4. Pico SDK
    sdk_path = TOOLCHAIN_DIR / "pico-sdk"
    clone_repo(
        "Pico SDK",
        "https://github.com/raspberrypi/pico-sdk.git",
        PICO_SDK_VERSION,
        sdk_path,
    )

    # 5. ExecuTorch Source
    et_path = TOOLCHAIN_DIR / "executorch"
    clone_repo(
        "ExecuTorch",
        "https://github.com/pytorch/executorch.git",
        EXECUTORCH_VERSION,
        et_path,
    )

    # 6. Generate environment script
    generate_env_script(toolchain_path, sdk_path, et_path)

    print("\n" + "=" * 60)
    log_success("Bootstrap complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. uv run python train_and_export.py")
    print("  2. uv run python build_firmware.py")
    print("\nOr source env.sh to use manual cmake commands.")


if __name__ == "__main__":
    main()