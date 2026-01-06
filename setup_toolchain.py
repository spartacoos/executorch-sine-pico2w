#!/usr/bin/env python3
"""
Toolchain Setup for Raspberry Pi Pico 2 W.

Downloads and configures:
- ARM GNU Embedded Toolchain (for cross-compilation)
- Pico SDK 2.0+ (for Pico 2 W support)

All artifacts are stored in the 'toolchains/' directory.
"""

import hashlib
import os
import platform
import shutil
import subprocess
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

# Configuration
TOOLCHAIN_DIR = Path("toolchains")
ARM_TOOLCHAIN_VERSION = "13.3.rel1"
PICO_SDK_VERSION = "2.1.0"  # 2.0+ required for Pico 2 W


def log_info(msg: str) -> None:
    print(f"\033[94m[INFO]\033[0m {msg}")


def log_success(msg: str) -> None:
    print(f"\033[92m[OK]\033[0m {msg}")


def log_warn(msg: str) -> None:
    print(f"\033[93m[WARN]\033[0m {msg}")


def log_error(msg: str) -> None:
    print(f"\033[91m[ERROR]\033[0m {msg}")


def detect_host() -> tuple[str, str]:
    """Detect host OS and architecture."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system not in ("linux", "darwin"):
        log_error(f"Unsupported OS: {system}. Only Linux and macOS are supported.")
        raise SystemExit(1)
    
    # Normalize architecture names
    if machine in ("x86_64", "amd64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "aarch64"
    else:
        log_error(f"Unsupported architecture: {machine}")
        raise SystemExit(1)
    
    log_info(f"Detected host: {system} / {arch}")
    return system, arch


def download_file(url: str, dest: Path) -> None:
    """Download a file with progress bar."""
    log_info(f"Downloading {dest.name}...")
    
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    
    with (
        open(dest, "wb") as f,
        tqdm(total=total_size, unit="B", unit_scale=True, desc=dest.name) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_arm_toolchain(host_os: str, host_arch: str) -> Path:
    """Download ARM GNU Embedded Toolchain if not present."""
    TOOLCHAIN_DIR.mkdir(exist_ok=True)
    
    # Build toolchain filename based on host
    if host_os == "darwin":
        os_name = "darwin"
        arch_name = "arm64" if host_arch == "aarch64" else "x86_64"
    else:
        os_name = host_arch  # Linux uses arch in the name
        arch_name = ""
    
    if host_os == "darwin":
        toolchain_name = f"arm-gnu-toolchain-{ARM_TOOLCHAIN_VERSION}-{os_name}-{arch_name}-arm-none-eabi"
    else:
        toolchain_name = f"arm-gnu-toolchain-{ARM_TOOLCHAIN_VERSION}-{host_arch}-arm-none-eabi"
    
    toolchain_path = TOOLCHAIN_DIR / toolchain_name
    gcc_path = toolchain_path / "bin" / "arm-none-eabi-gcc"
    
    if gcc_path.exists():
        log_success(f"ARM toolchain already installed: {toolchain_path}")
        return toolchain_path
    
    # Download
    archive_name = f"{toolchain_name}.tar.xz"
    url = f"https://developer.arm.com/-/media/Files/downloads/gnu/{ARM_TOOLCHAIN_VERSION}/binrel/{archive_name}"
    archive_path = TOOLCHAIN_DIR / archive_name
    
    if not archive_path.exists():
        try:
            download_file(url, archive_path)
        except requests.RequestException as e:
            log_error(f"Failed to download toolchain: {e}")
            log_info("Please download manually from: https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads")
            raise SystemExit(1)
    
    # Extract
    log_info(f"Extracting {archive_name}...")
    with tarfile.open(archive_path, "r:xz") as tar:
        tar.extractall(TOOLCHAIN_DIR)
    
    # Cleanup archive
    archive_path.unlink()
    
    # Verify installation
    if not gcc_path.exists():
        log_error(f"Toolchain extraction failed: {gcc_path} not found")
        raise SystemExit(1)
    
    log_success(f"ARM toolchain installed: {toolchain_path}")
    return toolchain_path


def clone_pico_sdk() -> Path:
    """Clone Pico SDK if not present."""
    TOOLCHAIN_DIR.mkdir(exist_ok=True)
    sdk_path = TOOLCHAIN_DIR / "pico-sdk"
    
    if (sdk_path / "pico_sdk_init.cmake").exists():
        log_success(f"Pico SDK already installed: {sdk_path}")
        return sdk_path
    
    log_info(f"Cloning Pico SDK v{PICO_SDK_VERSION}...")
    
    if shutil.which("git") is None:
        log_error("git is required but not installed")
        raise SystemExit(1)
    
    result = subprocess.run(
        [
            "git", "clone",
            "-b", PICO_SDK_VERSION,
            "--depth", "1",
            "--recurse-submodules",
            "--shallow-submodules",
            "https://github.com/raspberrypi/pico-sdk.git",
            str(sdk_path),
        ],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        log_error(f"Failed to clone Pico SDK: {result.stderr}")
        raise SystemExit(1)
    
    log_success(f"Pico SDK installed: {sdk_path}")
    return sdk_path


def check_cmake() -> None:
    """Verify cmake is installed."""
    if shutil.which("cmake") is None:
        log_error("cmake is required but not installed")
        log_info("Install with: brew install cmake (macOS) or sudo apt install cmake (Linux)")
        raise SystemExit(1)
    
    result = subprocess.run(["cmake", "--version"], capture_output=True, text=True)
    version_line = result.stdout.split("\n")[0]
    log_success(f"cmake found: {version_line}")


def generate_env_script(toolchain_path: Path, sdk_path: Path) -> None:
    """Generate shell script to set up environment variables."""
    env_script = Path("env.sh")
    
    content = f"""#!/bin/bash
# Source this file to set up the build environment
# Usage: source env.sh

export PICO_SDK_PATH="{sdk_path.absolute()}"
export PICO_TOOLCHAIN_PATH="{toolchain_path.absolute()}"
export PATH="{toolchain_path.absolute()}/bin:$PATH"

echo "Environment configured:"
echo "  PICO_SDK_PATH=$PICO_SDK_PATH"
echo "  PICO_TOOLCHAIN_PATH=$PICO_TOOLCHAIN_PATH"
echo "  arm-none-eabi-gcc: $(which arm-none-eabi-gcc)"
"""
    
    env_script.write_text(content)
    env_script.chmod(0o755)
    log_success(f"Generated {env_script} - source it before building manually")


def main() -> None:
    print()
    log_info("Setting up toolchains for Pico 2 W...")
    print()
    
    # Check prerequisites
    check_cmake()
    
    # Detect host and download/setup toolchains
    host_os, host_arch = detect_host()
    toolchain_path = download_arm_toolchain(host_os, host_arch)
    sdk_path = clone_pico_sdk()
    
    # Generate helper script
    generate_env_script(toolchain_path, sdk_path)
    
    print()
    log_success("Toolchain setup complete!")
    print()


if __name__ == "__main__":
    main()