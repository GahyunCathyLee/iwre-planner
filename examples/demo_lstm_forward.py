#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uncertainty.config import LSTMConfig
from uncertainty.lstm_model import LSTMUncertaintyEstimator


def main() -> None:
    import torch

    x = torch.randn(32, 10, 10)
    gaussian = LSTMUncertaintyEstimator(LSTMConfig(output_mode="gaussian"))
    gaussian_out = gaussian(x)
    print("gaussian", gaussian_out["mu"].shape, gaussian_out["logvar"].shape)

    scalar = LSTMUncertaintyEstimator(LSTMConfig(output_mode="scalar"))
    scalar_out = scalar(x)
    print("scalar", scalar_out["sigma"].shape)


if __name__ == "__main__":
    main()
