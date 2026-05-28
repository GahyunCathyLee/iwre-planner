#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uncertainty.track_a import TrackAEstimator


def main() -> None:
    estimator = TrackAEstimator()
    sequence = [(10.0 + i * 0.05, 0.0, 1.0, 0.0, True) for i in range(8)]
    sequence.extend([(10.4 + i * 0.05, 0.0, 1.0, 0.0, False) for i in range(5)])
    for index, (dx, dy, dvx, dvy, visible) in enumerate(sequence):
        out = estimator.step("veh1", dx, dy, dvx, dvy, visible)
        print(
            f"step={index:02d} visible={int(visible)} "
            f"sigma_phys={out['sigma_phys']:.3f} "
            f"sigma_kf={out['sigma_kf']:.3f} "
            f"sigma_a={out['sigma_a']:.3f}"
        )


if __name__ == "__main__":
    main()
