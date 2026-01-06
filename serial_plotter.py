#!/usr/bin/env python3
"""
Serial Plotter for Sine Wave Predictor.

Connects to the Pico 2 W over USB serial and plots the predicted vs expected
sine wave values in real-time.

Usage:
    python serial_plotter.py [--port /dev/ttyACM0]
"""

import argparse
import sys
from collections import deque
from typing import Optional

import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def find_pico_port() -> Optional[str]:
    """
    Auto-detect the Pico's USB serial port.
    
    Returns the first matching port or None if not found.
    """
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        # Pico shows up with these identifiers
        if "usbmodem" in port.device.lower():
            return port.device
        if "ttyacm" in port.device.lower():
            return port.device
        # Raspberry Pi vendor ID
        if port.vid == 0x2E8A:
            return port.device
    
    # Fallback: return first available port
    if ports:
        return ports[0].device
    
    return None


class SinePlotter:
    """Real-time plotter for sine wave predictions."""
    
    def __init__(self, port: str, baud: int = 115200, max_points: int = 200):
        self.port = port
        self.baud = baud
        self.max_points = max_points
        
        # Data storage (using deque for efficient append/pop)
        self.x_data: deque[float] = deque(maxlen=max_points)
        self.predicted: deque[float] = deque(maxlen=max_points)
        self.expected: deque[float] = deque(maxlen=max_points)
        self.errors: deque[float] = deque(maxlen=max_points)
        
        # Serial connection
        self.serial: Optional[serial.Serial] = None
        
        # Setup matplotlib
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle("ExecuTorch Sine Wave Predictor - Pico 2 W", fontsize=14)
        
        # Line objects for animation
        self.line_pred, = self.ax1.plot([], [], "b-", label="Predicted", linewidth=2)
        self.line_exp, = self.ax1.plot([], [], "r--", label="Expected", linewidth=1.5)
        self.line_err, = self.ax2.plot([], [], "g-", label="Error", linewidth=1)
        
        # Configure axes
        self._setup_axes()
        
        # Stats text
        self.stats_text = self.ax1.text(
            0.02, 0.98, "",
            transform=self.ax1.transAxes,
            verticalalignment="top",
            fontfamily="monospace",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        )
    
    def _setup_axes(self) -> None:
        """Configure plot axes."""
        # Top plot: sine wave
        self.ax1.set_xlim(0, 2 * np.pi)
        self.ax1.set_ylim(-1.3, 1.3)
        self.ax1.set_xlabel("x (radians)")
        self.ax1.set_ylabel("sin(x)")
        self.ax1.legend(loc="upper right")
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_title("Predicted vs Expected")
        
        # Bottom plot: error
        self.ax2.set_xlim(0, 2 * np.pi)
        self.ax2.set_ylim(-0.15, 0.15)
        self.ax2.set_xlabel("x (radians)")
        self.ax2.set_ylabel("Prediction Error")
        self.ax2.legend(loc="upper right")
        self.ax2.grid(True, alpha=0.3)
        self.ax2.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    
    def connect(self) -> bool:
        """Establish serial connection."""
        try:
            self.serial = serial.Serial(self.port, self.baud, timeout=1)
            print(f"Connected to {self.port} at {self.baud} baud")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect: {e}")
            return False
    
    def parse_line(self, line: str) -> Optional[tuple[float, float, float]]:
        """
        Parse a data line from the Pico.
        
        Expected format: DATA,<x>,<predicted>,<expected>
        """
        try:
            if not line.startswith("DATA,"):
                return None
            
            parts = line.strip().split(",")
            if len(parts) != 4:
                return None
            
            x = float(parts[1])
            predicted = float(parts[2])
            expected = float(parts[3])
            
            return (x, predicted, expected)
        except (ValueError, IndexError):
            return None
    
    def update(self, frame: int) -> tuple:
        """Animation update function - called for each frame."""
        if not self.serial or not self.serial.is_open:
            return self.line_pred, self.line_exp, self.line_err
        
        # Read available lines
        while self.serial.in_waiting:
            try:
                raw_line = self.serial.readline()
                line = raw_line.decode("utf-8", errors="ignore")
                data = self.parse_line(line)
                
                if data:
                    x, pred, exp = data
                    self.x_data.append(x)
                    self.predicted.append(pred)
                    self.expected.append(exp)
                    self.errors.append(pred - exp)
            except Exception as e:
                print(f"Read error: {e}")
        
        # Update plot data
        if self.x_data:
            x_arr = list(self.x_data)
            pred_arr = list(self.predicted)
            exp_arr = list(self.expected)
            err_arr = list(self.errors)
            
            self.line_pred.set_data(x_arr, pred_arr)
            self.line_exp.set_data(x_arr, exp_arr)
            self.line_err.set_data(x_arr, err_arr)
            
            # Update stats
            if err_arr:
                mae = np.mean(np.abs(err_arr))
                max_err = np.max(np.abs(err_arr))
                self.stats_text.set_text(
                    f"MAE: {mae:.6f}\n"
                    f"Max Error: {max_err:.6f}\n"
                    f"Points: {len(err_arr)}"
                )
        
        return self.line_pred, self.line_exp, self.line_err
    
    def run(self) -> None:
        """Start the real-time plotting."""
        if not self.connect():
            return
        
        print("Starting real-time plot (Ctrl+C to exit)...")
        print("Waiting for data from Pico 2 W...")
        
        # Create animation
        _ = animation.FuncAnimation(
            self.fig,
            self.update,
            interval=50,
            blit=False,
            cache_frame_data=False,
        )
        
        plt.tight_layout()
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            if self.serial:
                self.serial.close()


def list_ports() -> None:
    """List available serial ports."""
    print("Available serial ports:")
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("  No ports found")
    for port in ports:
        print(f"  {port.device}: {port.description}")
        if port.vid:
            print(f"    VID:PID = {port.vid:04x}:{port.pid:04x}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot sine wave predictions from Pico 2 W"
    )
    parser.add_argument(
        "--port", "-p",
        help="Serial port (auto-detected if not specified)"
    )
    parser.add_argument(
        "--baud", "-b",
        type=int,
        default=115200,
        help="Baud rate (default: 115200)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available serial ports and exit"
    )
    
    args = parser.parse_args()
    
    # List ports if requested
    if args.list:
        list_ports()
        return
    
    # Find port
    port = args.port or find_pico_port()
    if not port:
        print("No serial port found. Is the Pico connected?")
        print()
        list_ports()
        sys.exit(1)
    
    print(f"Using port: {port}")
    print()
    
    # Run plotter
    plotter = SinePlotter(port, args.baud)
    plotter.run()


if __name__ == "__main__":
    main()