"""Compare C++ network inference with PyTorch on raw data records.

    python -m nmo_train.check NETWORK.pt DATA_PREFIX [--records N]

Runs `nmo-datagen evaluate` on DATA_PREFIX.raw and the PyTorch model on the
featurized DATA_PREFIX arrays, and reports the largest logit difference.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from .data import build_dataset, make_batch
from .models import ARCHITECTURES

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data")
    parser.add_argument("--records", type=int, default=2000)
    parser.add_argument("--datagen", type=Path, default=ROOT / "build/release/nmo-datagen")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "logits.bin"
        subprocess.run([str(args.datagen), "evaluate", args.data + ".raw", "--strategy",
                        str(args.checkpoint.with_suffix(".nmo")), "--out", str(out), "--limit", str(args.records)],
                       check=True)
        cpp = np.fromfile(out, dtype=np.float32)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = ARCHITECTURES[checkpoint["arch"]](**checkpoint["model_args"])
    model.load_state_dict(checkpoint["state"])
    model.eval()
    data = build_dataset(args.data, keep_all=True)
    with torch.no_grad():
        batch = make_batch(data, torch.arange(args.records))
        python = model(batch).numpy()
    assert python.shape == cpp.shape, (python.shape, cpp.shape)
    difference = np.abs(python - cpp)
    print(f"{len(cpp)} logits: max |diff| {difference.max():.2e}, mean {difference.mean():.2e}, "
          f"logit std {python.std():.2f}")


if __name__ == "__main__":
    main()
