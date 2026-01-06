#!/usr/bin/env python3
"""
Train a tiny sine wave predictor and export to ExecuTorch .pte format.

Hardened version:
- Explicit float32 everywhere
- Model boundary enforces dtype
- Safe against accidental int/Long inputs
"""

from pathlib import Path
import torch
import torch.nn as nn
from torch.export import export
from executorch.exir import to_edge, EdgeCompileConfig

# =============================================================================
# Configuration
# =============================================================================

HIDDEN_SIZE = 16
NUM_EPOCHS = 2000
LEARNING_RATE = 0.01
OUTPUT_FILE = "sine_model.pte"

DTYPE = torch.float32
DEVICE = torch.device("cpu")

# =============================================================================
# Model
# =============================================================================


class SinePredictor(nn.Module):
    """
    Simple MLP to approximate sin(x).

    Architecture: 1 → 16 → 16 → 1
    """

    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HARDENING: enforce float32 at model boundary
        return self.net(x.to(dtype=DTYPE))


# =============================================================================
# Data
# =============================================================================


def generate_training_data(n_samples: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate training data: x ∈ [0, 2π], y = sin(x)
    """
    x = torch.linspace(
        0.0,
        2.0 * torch.pi,
        n_samples,
        dtype=DTYPE,
        device=DEVICE,
    ).unsqueeze(1)

    y = torch.sin(x)
    return x, y


# =============================================================================
# Training
# =============================================================================


def train_model(model: nn.Module, epochs: int) -> None:
    print("Training model...")

    x_train, y_train = generate_training_data()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

    print(f"Training complete. Final loss: {loss.item():.6f}")


# =============================================================================
# Evaluation
# =============================================================================


def test_model(model: nn.Module) -> None:
    """
    Quick sanity check on canonical sine points.
    """
    print("\nTesting model:")
    model.eval()

    test_points = [
        0.0,
        torch.pi / 4,
        torch.pi / 2,
        torch.pi,
        3 * torch.pi / 2,
    ]

    with torch.no_grad():
        for x_val in test_points:
            x = torch.tensor([[x_val]], dtype=DTYPE, device=DEVICE)
            pred = model(x).item()
            actual = torch.sin(x).item()
            print(f"  x={float(x_val):.4f}: pred={pred:.4f}, actual={actual:.4f}")


# =============================================================================
# Export
# =============================================================================


def export_model(model: nn.Module, output_path: str) -> None:
    """
    Export trained model to ExecuTorch .pte format.
    """
    print(f"\nExporting model to {output_path}...")
    model.eval()

    example_input = (
        torch.randn(1, 1, dtype=DTYPE, device=DEVICE),
    )

    exported_program = export(model, example_input)

    edge_config = EdgeCompileConfig(_check_ir_validity=False)
    edge_program = to_edge(exported_program, compile_config=edge_config)

    et_program = edge_program.to_executorch()

    output_file = Path(output_path)
    with open(output_file, "wb") as f:
        f.write(et_program.buffer)

    size_kb = len(et_program.buffer) / 1024
    print(f"Model exported: {output_file} ({size_kb:.1f} KB)")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("=" * 50)
    print("  ExecuTorch Sine Wave Predictor - Training")
    print("=" * 50 + "\n")

    model = SinePredictor(hidden_size=HIDDEN_SIZE).to(device=DEVICE, dtype=DTYPE)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count}")

    train_model(model, epochs=NUM_EPOCHS)
    test_model(model)
    export_model(model, OUTPUT_FILE)

    print("\n" + "=" * 50)
    print("Done! Next: uv run python build_firmware.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
