"""Write and edit .nmo weight files read by src/neural/weights.cpp."""

import struct
from pathlib import Path

import numpy as np
import torch

VERSION = 1


def _string(value: str) -> bytes:
    data = value.encode()
    return struct.pack("<I", len(data)) + data


def write_nmo(path: str | Path, architecture: str, settings: dict, tensors: dict) -> None:
    out = bytearray(b"NMOW" + struct.pack("<I", VERSION) + _string(architecture))
    out += struct.pack("<I", len(settings))
    for key, value in settings.items():
        out += _string(key) + struct.pack("<d", float(value))
    out += struct.pack("<I", len(tensors))
    for name, tensor in tensors.items():
        array = (tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor))
        array = np.ascontiguousarray(array, dtype="<f4")
        out += _string(name) + struct.pack("<I", array.ndim) + struct.pack(f"<{array.ndim}I", *array.shape)
        out += array.tobytes()
    Path(path).write_bytes(bytes(out))


def read_nmo(path: str | Path) -> tuple[str, dict, dict]:
    data = memoryview(Path(path).read_bytes())
    position = 0

    def take(fmt):
        nonlocal position
        values = struct.unpack_from(fmt, data, position)
        position += struct.calcsize(fmt)
        return values

    def string():
        nonlocal position
        (length,) = take("<I")
        value = bytes(data[position:position + length]).decode()
        position += length
        return value

    assert bytes(data[:4]) == b"NMOW", "not a .nmo file"
    position = 4
    (version,) = take("<I")
    assert version == VERSION, f"unsupported version {version}"
    architecture = string()
    settings = {}
    for _ in range(take("<I")[0]):
        key = string()
        settings[key] = take("<d")[0]
    tensors = {}
    for _ in range(take("<I")[0]):
        name = string()
        (rank,) = take("<I")
        shape = take(f"<{rank}I")
        count = int(np.prod(shape))
        tensors[name] = np.frombuffer(data, dtype="<f4", count=count, offset=position).reshape(shape).copy()
        position += 4 * count
    return architecture, settings, tensors


def export_model(model, path: str | Path, **settings) -> None:
    tensors, model_settings = model.export()
    model_settings.update(settings)
    write_nmo(path, model.arch, model_settings, tensors)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Copy a .nmo file with changed settings, e.g. history_weight=0.001")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("settings", nargs="*", help="key=value pairs")
    parser.add_argument("--leaf-head", type=Path,
                        help="tactical .nmo used instead of the full network near the leaves")
    parser.add_argument("--leaf-depth", type=int, default=0,
                        help="deepest remaining depth that uses the leaf head (0: quiescence only)")
    args = parser.parse_args()
    architecture, settings, tensors = read_nmo(args.source)
    if args.leaf_head:
        head_architecture, _, head = read_nmo(args.leaf_head)
        assert head_architecture == "tactical", "the leaf head must be a tactical network"
        for name, value in head.items():
            tensors[f"leaf.{name}"] = value
        # Keep the depth buckets 0..leaf_depth of the tactical biases.
        tensors["leaf.tactic.bias"] = head["tactic.bias"][:(args.leaf_depth + 1) * 6 * 7 * 256]
        settings["leaf_depth"] = float(args.leaf_depth)
        settings["quiescence"] = 1.0
    for pair in args.settings:
        key, value = pair.split("=", 1)
        settings[key] = float(value)
    write_nmo(args.destination, architecture, settings, tensors)
    print(f"{args.destination}: {architecture} {settings}")


if __name__ == "__main__":
    main()
