# ExecuTorch Sine Wave Predictor for Raspberry Pi Pico 2 W

A minimal, end-to-end tutorial for deploying a neural network on the RP2350 microcontroller using PyTorch and ExecuTorch. This project bridges the gap between cloud-based ML frameworks and bare-metal embedded systems.

---

## What This Project Teaches

By completing this tutorial, you will understand:

1. **PyTorch Fundamentals** — How tensors flow through a computational graph
2. **Model Export** — Converting dynamic Python to a static, inspectable graph
3. **ExecuTorch Lowering** — Transforming that graph for embedded execution
4. **Memory Planning** — Pre-calculating every byte of RAM before runtime
5. **Embedded Deployment** — Running inference without an OS or heap allocator

---

## The Big Picture: From Python to Silicon

Most ML tutorials stop at `model.predict()`. On embedded systems, we must go much further. Here's the complete journey:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEVELOPMENT MACHINE (Python)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│  │   PyTorch    │      │    Export    │      │  ExecuTorch  │              │
│  │    Model     │ ──▶  │    Graph     │ ──▶  │   Lowering   │              │
│  │  (Dynamic)   │      │   (Static)   │      │  (Optimize)  │              │
│  └──────────────┘      └──────────────┘      └──────────────┘              │
│         │                                           │                       │
│         │ nn.Module                                 │ .pte file             │
│         │ Python code                               │ (Flatbuffer)          │
│         ▼                                           ▼                       │
│  ┌──────────────┐                           ┌──────────────┐               │
│  │   Training   │                           │  C Header    │               │
│  │    Loop      │                           │  (bytes[])   │               │
│  └──────────────┘                           └──────────────┘               │
│                                                     │                       │
└─────────────────────────────────────────────────────│───────────────────────┘
                                                      │
                                                      │ Cross-compile
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PICO 2 W (Bare Metal C++)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│  │  ExecuTorch  │      │    Method    │      │    Output    │              │
│  │   Runtime    │ ──▶  │   Execute    │ ──▶  │   (sin(x))   │              │
│  │   (~50 KB)   │      │  (Kernels)   │      │              │              │
│  └──────────────┘      └──────────────┘      └──────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Understanding PyTorch's Execution Model

### Eager Mode vs. Graph Mode

PyTorch operates in two fundamentally different modes. Understanding when and why to use each is essential.

**Eager Mode** (default) — Operations execute immediately as Python runs:

```python
x = torch.tensor([1.0, 2.0])
y = torch.tensor([3.0, 4.0])
z = x + y  # Addition happens RIGHT NOW, result stored in z
```

**Graph Mode** — Operations are recorded, then executed later:

```
       Eager Mode                         Graph Mode
    ┌───────────────┐               ┌───────────────────────┐
    │   Python      │               │   Python              │
    │   x = ...     │               │   x = placeholder     │
    │   y = ...     │               │   y = placeholder     │
    │   z = x + y   │◀─ executes    │   z = add(x, y)       │◀─ records
    │   print(z)    │   instantly   │   return z            │   graph
    └───────────────┘               └───────────────────────┘
                                              │
                                              ▼ export
                                    ┌───────────────────────┐
                                    │      Graph IR         │
                                    │  ┌───┐    ┌───┐       │
                                    │  │ x │───▶│add│──▶ z  │
                                    │  └───┘  ▲ └───┘       │
                                    │  ┌───┐  │             │
                                    │  │ y │──┘             │
                                    │  └───┘                │
                                    └───────────────────────┘
```

**Why Graph Mode for Embedded?**
- No Python interpreter needed at runtime
- The graph can be analyzed, optimized, and memory-planned ahead of time
- All operations are known before execution begins

### What is a Tensor?

A tensor is a multi-dimensional array with metadata. Every tensor carries:

```
┌─────────────────────────────────────────────────────────────┐
│                         Tensor                              │
├─────────────────────────────────────────────────────────────┤
│  Data Buffer:  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]              │
│                 ▲                                           │
│                 │ (contiguous memory)                       │
├─────────────────┼───────────────────────────────────────────┤
│  Shape:        │ (2, 3)      ─▶ 2 rows, 3 columns          │
│  Stride:       │ (3, 1)      ─▶ jump 3 to next row         │
│  Dtype:        │ float32     ─▶ 4 bytes per element        │
│  Device:       │ cpu         ─▶ where data lives           │
└─────────────────┴───────────────────────────────────────────┘

Logical View (2x3 matrix):         Memory Layout (flat):
┌─────┬─────┬─────┐                ┌───┬───┬───┬───┬───┬───┐
│ 1.0 │ 2.0 │ 3.0 │   row 0       │1.0│2.0│3.0│4.0│5.0│6.0│
├─────┼─────┼─────┤                └───┴───┴───┴───┴───┴───┘
│ 4.0 │ 5.0 │ 6.0 │   row 1         0   1   2   3   4   5
└─────┴─────┴─────┘
```

**Stride determines memory traversal:** To access element [1, 2] (value 6.0):
- Row index: 1 × stride[0] = 1 × 3 = 3
- Col index: 2 × stride[1] = 2 × 1 = 2
- Final offset: 3 + 2 = 5 → buffer[5] = 6.0

---

## Part 2: The Export Pipeline

### torch.export: Capturing the Graph

`torch.export` traces your model by running it with example inputs and recording every operation:

```
                    torch.export.export(model, example_inputs)
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                            ATen Dialect                                     │
│  (All 2000+ PyTorch operators available)                                   │
├────────────────────────────────────────────────────────────────────────────┤
│  graph():                                                                  │
│      %x : Tensor = placeholder[target=x]                                   │
│      %weight : Tensor = get_attr[target=fc1.weight]                        │
│      %bias : Tensor = get_attr[target=fc1.bias]                            │
│      %linear : Tensor = call_function[target=aten.linear](%x, %weight, %bias)
│      %relu : Tensor = call_function[target=aten.relu](%linear)             │
│      return (%relu,)                                                       │
└────────────────────────────────────────────────────────────────────────────┘
```

**What gets captured:**
- The exact sequence of tensor operations
- All weight tensors and their values (frozen at export time)
- Shape information for every intermediate tensor

**What does NOT get captured:**
- Python control flow (if/else based on tensor values)
- Side effects (print statements, file I/O)
- Dynamic Python logic

### Core ATen Decomposition

ATen (the "A TENsor" library) has thousands of operators. Many are "convenience" operators built from simpler primitives. Decomposition breaks them down:

```
Before Decomposition:                After Decomposition:
┌─────────────────────┐              ┌─────────────────────────────────┐
│  aten.linear        │     ──▶      │  aten.t (transpose)             │
│  (W, x, bias)       │              │  aten.mm (matrix multiply)      │
│                     │              │  aten.add (add bias)            │
└─────────────────────┘              └─────────────────────────────────┘

Operator Count:   ~2000              Operator Count:   ~180 (Core ATen)
```

**Why decompose?**
- Smaller operator set = smaller runtime binary
- Backend developers implement fewer kernels
- More optimization opportunities (fuse patterns across decomposed ops)

---

## Part 3: ExecuTorch Lowering

### The to_edge Transformation

`to_edge()` converts the graph for edge devices. Key changes:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Edge Dialect                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. DTYPE SPECIALIZATION                                                    │
│     ─────────────────────                                                   │
│     Operators become dtype-specific:                                        │
│                                                                             │
│     aten.add(Tensor, Tensor) ──▶ aten.add.float32(Tensor, Tensor)          │
│                                                                             │
│  2. SCALAR TO TENSOR                                                        │
│     ────────────────────                                                    │
│     All scalars become 0-dimensional tensors:                               │
│                                                                             │
│     x + 2.0 ──▶ x + tensor([2.0])                                          │
│                                                                             │
│  3. OUT-VARIANT CONVERSION                                                  │
│     ─────────────────────────                                               │
│     Operators take pre-allocated output buffers:                            │
│                                                                             │
│     z = add(x, y)  ──▶  add(x, y, out=z_buffer)                            │
│     (allocates z)       (writes to existing buffer)                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Memory Planning: The Zero-Allocation Guarantee

This is where ExecuTorch differs fundamentally from PyTorch. Before any code runs on the device, the memory planner:

1. Analyzes the **lifetime** of every tensor (when it's created, when it's last used)
2. Calculates the **size** of every tensor
3. Assigns **offsets** into a pre-allocated arena

```
Timeline of Tensor Lifetimes:
                    
Operation:   Linear1    ReLU1    Linear2    ReLU2    Output
             ────────────────────────────────────────────▶ time
                │         │         │         │
Tensor t1:     [=========]          │         │         created by Linear1
Tensor t2:               [=========]          │         created by ReLU1
Tensor t3:                         [=========]          created by Linear2
Tensor t4:                                   [=========] created by ReLU2

Memory Arena Layout (offsets pre-computed):

Offset:  0        256      512      768     1024
         ├────────┼────────┼────────┼────────┤
         │   t1   │   t2   │   t3   │   t4   │
         │ reuse  │ reuse  │        │        │
         └────────┴────────┴────────┴────────┘
                      │
                      ▼
            t1 and t2 can share memory!
            (non-overlapping lifetimes)
```

**The Memory Plan is Embedded in the .pte File:**

```
┌─────────────────────────────────────────────────────────────┐
│                     .pte File Contents                       │
├─────────────────────────────────────────────────────────────┤
│  • Instructions (which operators to call, in what order)    │
│  • Weight data (frozen model parameters)                    │
│  • Memory map (offset assignments for every tensor)         │
│  • Kernel references (which C++ functions to invoke)        │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 4: The Embedded Runtime

### Runtime Architecture on Pico 2 W

The RP2350 has severe constraints. Understanding them is essential:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RP2350 MEMORY MAP                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FLASH (4MB)                        SRAM (520KB)                            │
│  ────────────                       ────────────                            │
│  ┌────────────────────┐             ┌────────────────────┐                  │
│  │ Program Code       │             │ Stack              │  ← grows down    │
│  │ (.text section)    │             │                    │                  │
│  │ ~50KB runtime      │             ├────────────────────┤                  │
│  ├────────────────────┤             │ Heap (UNUSED)      │  ← we avoid this │
│  │ Read-Only Data     │             │                    │                  │
│  │ (.rodata)          │             ├────────────────────┤                  │
│  │ - Model weights    │─────────────│►Planned Memory     │  ← our arena     │
│  │ - .pte payload     │  XIP        │  Arena             │                  │
│  │ ~3KB for this model│  (execute   │  ~2KB activations  │                  │
│  └────────────────────┘   in place) ├────────────────────┤                  │
│                                     │ Input/Output       │                  │
│                                     │ Buffers            │                  │
│                                     │ (provided by app)  │                  │
│                                     └────────────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

XIP = Execute In Place: Code runs directly from Flash, no copy to RAM needed
```

### The MemoryManager Hierarchy

ExecuTorch uses a hierarchical memory allocation system:

```
                    ┌─────────────────────────────────┐
                    │         MemoryManager           │
                    └─────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │   Method    │  │   Planned   │  │  Temporary  │
     │  Allocator  │  │   Memory    │  │  Allocator  │
     └─────────────┘  └─────────────┘  └─────────────┘
           │               │                 │
           ▼               ▼                 ▼
     Metadata &       Activation        Scratch space
     control flow     tensors           for kernels
```

**In practice (main.cpp):**

```cpp
// Fixed-size buffers, no malloc() at runtime
static uint8_t method_pool[32 * 1024];     // 32KB for method metadata
static uint8_t activation_pool[16 * 1024]; // 16KB for tensor activations

MemoryAllocator method_alloc(sizeof(method_pool), method_pool);
HierarchicalAllocator planned({activation_pool, sizeof(activation_pool)});
MemoryManager memory_manager(&method_alloc, &planned);
```

### Kernel Registry and Selective Build

The runtime must know how to execute each operator. This mapping is the Kernel Registry:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           Kernel Registry                                   │
├──────────────────────┬─────────────────────────────────────────────────────┤
│  Operator Name       │  C++ Implementation                                 │
├──────────────────────┼─────────────────────────────────────────────────────┤
│  aten.linear.out     │  → torch::executor::linear_out(...)                 │
│  aten.relu.out       │  → torch::executor::relu_out(...)                   │
│  aten.add.out        │  → torch::executor::add_out(...)                    │
│  ...                 │  ...                                                │
└──────────────────────┴─────────────────────────────────────────────────────┘
```

**Selective Build Problem:**
- Full kernel library = megabytes of code
- Your model uses 3 operators = you only need 3 kernels
- CMake `EXECUTORCH_SELECT_OPS_LIST` builds only what's needed

```cmake
# CMakeLists.txt
set(EXECUTORCH_SELECT_OPS_LIST "aten::linear.out,aten::relu.out,aten::add.out")
```

---

## Part 5: The Sine Wave Model

### Network Architecture

Our model learns to approximate `sin(x)` for x ∈ [0, 2π]:

```
Input          Hidden Layer 1      Hidden Layer 2      Output
  │            (16 neurons)        (16 neurons)          │
  │                                                      │
  x ──┬──▶ [Linear] ──▶ [ReLU] ──▶ [Linear] ──▶ [ReLU] ──▶ [Linear] ──▶ ŷ
      │        │                       │                       │
      │        ▼                       ▼                       ▼
      │   W₁: (1,16)              W₂: (16,16)             W₃: (16,1)
      │   b₁: (16,)               b₂: (16,)               b₃: (1,)
      │                                                      │
      └──────────────────────────────────────────────────────┘
                        Total: 337 parameters
```

**Universal Approximation Theorem:** A neural network with at least one hidden layer and sufficient neurons can approximate any continuous function to arbitrary precision. We're using this to learn the sine function.

### Why ReLU Works for Sine Approximation

ReLU (Rectified Linear Unit) outputs `max(0, x)`. A piecewise linear function:

```
ReLU(x):           sin(x):              Approximation:
    │      /           │    ╭──╮             │    /\
    │    /             │   /    \            │   /  \
────┼──/───           ─┼──/──────\──        ─┼──/────\──
    │                  │          \          │        \
    │                  │           ╰──       │         \/
```

With multiple ReLUs and learned weights, we can construct a piecewise linear approximation that closely follows the sine curve.

---

## Part 6: Project Files Explained

```
executorch-sine-pico2w/
├── pyproject.toml          # Python dependencies (managed by uv)
├── README.md               # This document
├── bootstrap.py            # Downloads ARM toolchain, sets up environment
├── setup_toolchain.py      # Configures cross-compilation paths
├── train_and_export.py     # Defines model, trains, exports to .pte
├── build_firmware.py       # Cross-compiles C++ runtime + model
├── main.cpp                # Pico 2 W entry point, runs inference loop
├── CMakeLists.txt          # Build configuration for embedded target
├── pte_to_header.py        # Converts .pte binary to C byte array
└── serial_plotter.py       # Visualizes predictions over USB serial
```

### Data Flow Through the Pipeline

```
train_and_export.py                    build_firmware.py
        │                                      │
        ▼                                      ▼
┌───────────────┐                    ┌──────────────────┐
│ 1. Define MLP │                    │ 4. CMake config  │
│ 2. Train loop │                    │    (ARM target)  │
│ 3. Export:    │                    │ 5. Compile:      │
│    - capture  │                    │    - runtime     │
│    - to_edge  │                    │    - kernels     │
│    - to_exec  │                    │    - main.cpp    │
│ 4. Serialize  │                    │ 6. Link .uf2     │
└───────┬───────┘                    └────────┬─────────┘
        │                                     │
        ▼                                     ▼
   sine_model.pte ──▶ pte_to_header.py ──▶ model_data.h
        │                                     │
        │                                     ▼
        │                              sine_predictor.uf2
        │                                     │
        └─────────────────────────────────────┘
                         │
                         ▼
                   Flash to Pico 2 W
```

---

## Part 7: Running the Project

### Prerequisites

- Python 3.12+ with [uv](https://docs.astral.sh/uv/) package manager
- Raspberry Pi Pico 2 W
- USB cable

### Quick Start

```bash
# Step 1: Bootstrap environment and toolchain
uv run python bootstrap.py

# Step 2: Train model and export to .pte
uv run python train_and_export.py

# Step 3: Build firmware
uv run python build_firmware.py

# Step 4: Flash to device
#   Hold BOOTSEL button, connect USB, release
#   Copy build/sine_predictor.uf2 to RPI-RP2 drive

# Step 5: View results
uv run python serial_plotter.py
```

---

## Part 8: Troubleshooting

### "Operator not found" at Runtime

**Symptom:** The firmware compiles, but crashes with a missing operator error.

**Cause:** The C++ kernel for that operator wasn't included in the selective build.

**Fix:** Add the operator to `CMakeLists.txt`:
```cmake
set(EXECUTORCH_SELECT_OPS_LIST 
    "aten::linear.out,aten::relu.out,aten::missing_op.out"
)
```

**How to find the missing operator:** Run `train_and_export.py` with verbose logging to see the full operator list.

### Model Too Large for SRAM

**Symptom:** Build succeeds, but device hangs or crashes immediately.

**Cause:** Activation memory exceeds the 520KB SRAM budget.

**Diagnostic Questions:**
1. What is the activation memory requirement? (Check `to_executorch()` output)
2. Can you reduce hidden layer sizes?
3. Can you use int8 quantization?

### Serial Output Not Appearing

**Cause:** USB CDC (serial) initialization takes time.

**Fix in main.cpp:**
```cpp
stdio_init_all();
sleep_ms(2000);  // Wait for USB enumeration
```

**Check:** Correct serial port (`/dev/ttyACM0` on Linux, `/dev/cu.usbmodem*` on macOS)

---

## Part 9: Extending the Project

### Adding a New Operator

If your model uses an operator not in the default kernel library:

1. Check if it's in the [Core ATen Op Set](https://pytorch.org/docs/stable/torch.compiler_ir.html#core-aten-ir)
2. If yes, add to `EXECUTORCH_SELECT_OPS_LIST`
3. If no, you'll need to write a custom kernel (advanced)

### Quantization (Reducing Model Size)

For larger models, 8-bit quantization reduces memory 4x:

```python
from executorch.exir.backend.backend_api import to_backend
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# After to_edge(), before to_executorch()
edge_program = to_backend(edge_program, XnnpackPartitioner())
```

### Different Target Boards

ExecuTorch supports multiple embedded targets:

| Target | CPU | Memory | Toolchain |
|--------|-----|--------|-----------|
| Pico 2 W | Cortex-M33 | 520KB SRAM | arm-none-eabi-gcc |
| STM32H7 | Cortex-M7 | 1MB SRAM | arm-none-eabi-gcc |
| ESP32-S3 | Xtensa LX7 | 512KB SRAM | xtensa-esp32s3-elf-gcc |

---

## References

- [ExecuTorch Documentation](https://pytorch.org/executorch/)
- [PyTorch Export Tutorial](https://pytorch.org/docs/stable/export.html)
- [Pico SDK Documentation](https://www.raspberrypi.com/documentation/pico-sdk/)
- [Core ATen Operator Set](https://pytorch.org/docs/stable/torch.compiler_ir.html#core-aten-ir)

---

## Appendix A: Key Concepts Quick Reference

| Concept | Definition | Why It Matters |
|---------|------------|----------------|
| **Eager Mode** | Operations execute immediately | Development/debugging |
| **Graph Mode** | Operations recorded, executed later | Deployment/optimization |
| **ATen** | PyTorch's tensor operation library | Foundation for all ops |
| **Core ATen** | Minimal subset of ATen (~180 ops) | What backends must implement |
| **Edge Dialect** | Graph specialized for edge devices | Dtype-specific, out-variants |
| **Out-Variant** | Operator writes to pre-allocated buffer | Enables memory planning |
| **Memory Plan** | Pre-computed tensor offset assignments | Zero allocation at runtime |
| **Selective Build** | Include only needed kernels | Minimize binary size |
| **.pte File** | Serialized ExecuTorch program | Flatbuffer format, portable |
| **XIP** | Execute In Place (from Flash) | Saves RAM on embedded |

---

## Appendix B: Memory Budget Worksheet

Use this to estimate if your model fits on Pico 2 W:

```
Available SRAM:                520 KB
  - ExecuTorch runtime:        - 50 KB (approx)
  - Method allocator:          - 32 KB (configurable)
  - Stack + globals:           - 20 KB (approx)
                               ────────
Remaining for activations:     ~418 KB

Your model's activation size:  ______ KB
  (from to_executorch() output)

Fits? [ ] Yes  [ ] No

If No, consider:
  [ ] Reduce hidden layer sizes
  [ ] Apply int8 quantization
  [ ] Use operator fusion
  [ ] Stream inputs in chunks
```