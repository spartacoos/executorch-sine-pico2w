#!/usr/bin/env python3
"""
Build firmware for Raspberry Pi Pico 2 W with ExecuTorch.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.absolute()
TOOLCHAIN_DIR = PROJECT_DIR / "toolchains"
BUILD_DIR = PROJECT_DIR / "build"
PTE_FILE = PROJECT_DIR / "sine_model.pte"


def log_info(msg: str) -> None:
    print(f"\033[94m[INFO]\033[0m {msg}")


def log_success(msg: str) -> None:
    print(f"\033[92m[OK]\033[0m {msg}")


def log_error(msg: str) -> None:
    print(f"\033[91m[ERROR]\033[0m {msg}")


def find_paths() -> dict[str, Path]:
    paths = {
        "et": TOOLCHAIN_DIR / "executorch",
        "sdk": TOOLCHAIN_DIR / "pico-sdk",
        "gcc": None,
    }

    gcc_candidates = list(TOOLCHAIN_DIR.glob("arm-gnu-toolchain*"))
    if gcc_candidates:
        paths["gcc"] = gcc_candidates[0]

    for name, path in paths.items():
        if path is None or not path.exists():
            log_error(f"Missing {name}. Run: uv run python bootstrap.py")
            sys.exit(1)

    return paths


def patch_executorch_for_baremetal(et_root: Path) -> None:
    """Patch ExecuTorch source tree for baremetal builds."""
    # Fix 1: Create cmake_macros.h stub
    stub_path = et_root / "runtime/core/portable_type/c10/torch/headeronly/macros/cmake_macros.h"
    if not stub_path.exists():
        log_info("Creating cmake_macros.h stub...")
        stub_path.write_text("// Auto-generated stub for baremetal builds\n")

    # Fix 2: Create 'executorch' symlink
    symlink_path = et_root / "executorch"
    if not symlink_path.exists():
        log_info("Creating executorch symlink...")
        symlink_path.symlink_to(".")


def create_arm_toolchain_file(paths: dict[str, Path]) -> Path:
    """Create CMake toolchain file with correct settings for Pico 2 W."""
    toolchain_file = BUILD_DIR / "arm-pico2w-toolchain.cmake"
    gcc = paths["gcc"]
    
    toolchain_file.parent.mkdir(parents=True, exist_ok=True)
    toolchain_file.write_text(f"""
# ARM Toolchain for Pico 2 W (Cortex-M33, soft float)
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_CROSSCOMPILING TRUE)

set(CMAKE_C_COMPILER "{gcc}/bin/arm-none-eabi-gcc")
set(CMAKE_CXX_COMPILER "{gcc}/bin/arm-none-eabi-g++")
set(CMAKE_ASM_COMPILER "{gcc}/bin/arm-none-eabi-gcc")

# Must match Pico SDK settings: Cortex-M33, soft float ABI
set(CMAKE_C_FLAGS_INIT "-mcpu=cortex-m33 -mthumb -mfloat-abi=soft")
set(CMAKE_CXX_FLAGS_INIT "-mcpu=cortex-m33 -mthumb -mfloat-abi=soft -fno-exceptions -fno-rtti")

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
""")
    
    return toolchain_file


def build_executorch_baremetal(paths: dict[str, Path]) -> Path:
    et_root = paths["et"]
    et_build_dir = et_root / "cmake-out"

    # Apply patches
    patch_executorch_for_baremetal(et_root)

    # Check if already built
    if (et_build_dir / "lib" / "libexecutorch.a").exists():
        log_success("ExecuTorch baremetal already built.")
        return et_build_dir

    log_info("Cross-compiling ExecuTorch for ARM baremetal...")

    # Create toolchain file with matching float ABI
    toolchain_file = create_arm_toolchain_file(paths)

    env = os.environ.copy()
    env["PATH"] = f"{paths['gcc']}/bin:{env['PATH']}"

    cmake_args = [
        "cmake",
        "-B", str(et_build_dir),
        "-S", str(et_root),
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file}",
        "-DCMAKE_BUILD_TYPE=MinSizeRel",
        "-DEXECUTORCH_BUILD_ARM_BAREMETAL=ON",
        "-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF",
        "-DEXECUTORCH_ENABLE_LOGGING=OFF",
        "-DEXECUTORCH_PAL_DEFAULT=minimal",
        f"-DCMAKE_INSTALL_PREFIX={et_build_dir}",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]

    subprocess.check_call(cmake_args, env=env)
    subprocess.check_call(
        ["cmake", "--build", str(et_build_dir), "--target", "install", "-j"],
        env=env,
    )

    log_success("ExecuTorch baremetal build complete.")
    return et_build_dir


def convert_pte_to_header(pte_path: Path, output_path: Path) -> None:
    """Convert .pte to C header. Skip if already up-to-date."""
    if output_path.exists() and output_path.stat().st_mtime >= pte_path.stat().st_mtime:
        log_success("Model header already up-to-date.")
        return

    log_info("Converting model to C header...")

    data = pte_path.read_bytes()
    hex_bytes = [f"0x{b:02x}" for b in data]
    lines = [", ".join(hex_bytes[i:i+12]) for i in range(0, len(hex_bytes), 12)]

    output_path.write_text(f"""#ifndef MODEL_PTE_H
#define MODEL_PTE_H

#include <stdint.h>

const uint8_t model_pte[] __attribute__((aligned(8))) = {{
{",".join("    " + l for l in lines)}
}};

const unsigned int model_pte_len = {len(data)};

#endif
""")

    log_success(f"Generated {output_path}")


def build_pico_firmware(paths: dict[str, Path]) -> None:
    log_info("Building Pico 2 W firmware...")

    cmake_build_dir = BUILD_DIR / "cmake-out"
    
    main_cpp = PROJECT_DIR / "main.cpp"
    cmake_file = PROJECT_DIR / "CMakeLists.txt"
    cache_file = cmake_build_dir / "CMakeCache.txt"
    
    needs_reconfigure = (
        not cache_file.exists() or
        (main_cpp.exists() and main_cpp.stat().st_mtime > cache_file.stat().st_mtime) or
        (cmake_file.exists() and cmake_file.stat().st_mtime > cache_file.stat().st_mtime)
    )

    if needs_reconfigure:
        shutil.rmtree(cmake_build_dir, ignore_errors=True)
        cmake_build_dir.mkdir(parents=True)

        shutil.copy(main_cpp, BUILD_DIR / "main.cpp")
        shutil.copy(cmake_file, BUILD_DIR / "CMakeLists.txt")

        env = os.environ.copy()
        env["PICO_SDK_PATH"] = str(paths["sdk"])
        env["PATH"] = f"{paths['gcc']}/bin:{env['PATH']}"

        cmake_args = [
            "cmake",
            "-B", str(cmake_build_dir),
            "-S", str(BUILD_DIR),
            "-DPICO_BOARD=pico2_w",
            f"-DEXECUTORCH_ROOT={paths['et']}",
        ]

        subprocess.check_call(cmake_args, env=env)

    env = os.environ.copy()
    env["PICO_SDK_PATH"] = str(paths["sdk"])
    env["PATH"] = f"{paths['gcc']}/bin:{env['PATH']}"
    
    subprocess.check_call(["cmake", "--build", str(cmake_build_dir), "-j"], env=env)

    uf2 = cmake_build_dir / "sine_predictor.uf2"
    shutil.copy(uf2, BUILD_DIR / "sine_predictor.uf2")
    log_success(f"Firmware built: {BUILD_DIR / 'sine_predictor.uf2'}")


def main():
    print("\n" + "=" * 60)
    print("  ExecuTorch Sine Wave Predictor - Build Firmware")
    print("=" * 60 + "\n")

    paths = find_paths()
    BUILD_DIR.mkdir(exist_ok=True)

    build_executorch_baremetal(paths)
    convert_pte_to_header(PTE_FILE, BUILD_DIR / "model_pte.h")
    build_pico_firmware(paths)

    log_success("Build complete.")


if __name__ == "__main__":
    main()