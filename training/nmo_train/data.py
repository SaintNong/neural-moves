"""Load featurized search-tree samples and turn probe labels into training targets.

The C++ generator (nmo-datagen) records, for every move of a sampled node, the
fail-soft score of probing that move with the node's (alpha, beta) window and
the probe's node count. From these we derive, per node:

* cut nodes (some move scores >= beta): every move's *efficiency* if searched
  first. A cutting move m costs c_m, so its efficiency is c_min / c_m where
  c_min is the cheapest refutation. A failing move costs c_m and is followed,
  at best, by the cheapest refutation: efficiency c_min / (c_m + c_min).
* PV nodes (no cut, but some move raises alpha): efficiency 1 for the best
  moves and 0 otherwise.
* all-nodes (every move fails low) are dropped: every move must be searched,
  so their ordering does not change the node's cost.

The search places its hash move first regardless of the orderer, so the hash
move is masked out, and nodes where it already cuts are dropped.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

MOVE_FIELDS = 16  # Bytes per move record written by `nmo-datagen featurize`.


@dataclass
class Arrays:
    """Raw featurized arrays (numpy, possibly memory-mapped)."""

    features: np.ndarray  # [N, 64] int16: 32 stm then 32 opponent piece-square indices, -1 padded.
    info: np.ndarray      # [N, 8] int32: depth, label_depth, ply, alpha, beta, offset, count, root.
    aux: np.ndarray       # [N, 8] uint8: castling bits, ep file + 1, halfmove, in check, pieces.
    moves: np.ndarray     # [M, 16] uint8: from, to, piece, victim, promo, threats, castle, ep, hash.
    labels: np.ndarray    # [M, 2] int32: score, nodes.


def load_arrays(prefix: str | Path) -> Arrays:
    prefix = str(prefix)
    meta = json.loads(Path(prefix + ".json").read_text())
    n, m = meta["positions"], meta["moves"]
    load = lambda suffix, dtype, width, rows: np.memmap(f"{prefix}.{suffix}.bin", dtype=dtype, mode="r").reshape(rows, width)
    return Arrays(
        features=load("features", np.int16, 64, n),
        info=load("info", np.int32, 8, n),
        aux=load("aux", np.uint8, 8, n),
        moves=load("moves", np.uint8, MOVE_FIELDS, m),
        labels=load("labels", np.int32, 2, m),
    )


def node_targets(arrays: Arrays, tau: float = 1.0, failing_credit: bool = True):
    """Return (keep, efficiency, valid, kind) for every node and move.

    keep[N] marks nodes with an ordering decision worth learning; efficiency[M]
    is each move's first-pick efficiency raised to tau; valid[M] is false for
    the hash move; kind[N] is 0 for all-nodes, 1 for cut nodes, 2 for PV nodes.
    """
    # 32-bit arrays throughout: there can be over 100M moves.
    info = arrays.info
    offsets, counts = info[:, 5].astype(np.int64), info[:, 6].astype(np.int64)
    alpha, beta = info[:, 3], info[:, 4]
    node = np.repeat(np.arange(len(info), dtype=np.int32), counts)
    score = np.asarray(arrays.labels[:, 0])
    cost = np.maximum(arrays.labels[:, 1].astype(np.float32), np.float32(1.0))
    is_hash = arrays.moves[:, 8].astype(bool)
    valid = ~is_hash

    cut = score >= beta[node]
    hash_cut = np.maximum.reduceat((cut & is_hash).astype(np.int8), offsets) > 0
    node_cut = np.maximum.reduceat((cut & valid).astype(np.int8), offsets) > 0
    best = np.maximum.reduceat(np.where(valid, score, np.int32(-(1 << 30))), offsets)
    pv = ~node_cut & (best > alpha)

    cut_cost = np.where(cut & valid, cost, np.float32(np.inf))
    c_min = np.minimum.reduceat(cut_cost, offsets)[node]
    del cut_cost
    if tau == 0:
        # Cost-agnostic: any refutation is equally good.
        cut_eff = cut.astype(np.float32)
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            failing = c_min / (cost + c_min) if failing_credit else 0.0
            cut_eff = np.where(cut, c_min / cost, failing) ** tau
    pv_eff = (score == best[node]).astype(np.float32)
    efficiency = np.where(node_cut[node], cut_eff, np.where(pv[node], pv_eff, np.float32(0.0)))
    efficiency = np.where(valid, efficiency, np.float32(0.0))

    kind = np.where(node_cut, 1, np.where(pv, 2, 0))
    keep = (kind != 0) & ~hash_cut & (np.add.reduceat(valid.astype(np.int64), offsets) >= 2)
    return keep, efficiency.astype(np.float32), valid, kind


@dataclass
class Dataset:
    """Compact tensors for the kept nodes; moves of node i are contiguous."""

    features: torch.Tensor   # [N, 64] int16
    aux: torch.Tensor        # [N, 8] uint8
    depth: torch.Tensor      # [N] int8
    offsets: torch.Tensor    # [N + 1] int64
    moves: torch.Tensor      # [M, 8] uint8: from, to, piece, victim, promo, threats, castle, ep
    target: torch.Tensor     # [M] float32 efficiency (0 for the masked hash move)
    valid: torch.Tensor      # [M] bool
    cost: torch.Tensor       # [M] float32 probe node count (for offline evaluation)
    cut: torch.Tensor        # [M] bool (score >= beta)
    root: torch.Tensor       # [N] int32

    def __len__(self) -> int:
        return self.features.shape[0]

    def to(self, device) -> "Dataset":
        return Dataset(**{name: getattr(self, name).to(device) for name in self.__dataclass_fields__})


def build_dataset(prefix: str | Path, tau: float = 1.0, failing_credit: bool = True,
                  depths: tuple[int, ...] | None = None, keep_all: bool = False, cache: bool = True) -> Dataset:
    """Build (or load a cached copy of) the training tensors for a data prefix."""
    cacheable = cache and depths is None and not keep_all
    cache_path = Path(f"{prefix}.cache-tau{tau:g}-fc{int(failing_credit)}.pt")
    if cacheable and cache_path.exists() and cache_path.stat().st_mtime > Path(f"{prefix}.json").stat().st_mtime:
        return Dataset(**torch.load(cache_path))
    data = _build_dataset(prefix, tau, failing_credit, depths, keep_all)
    if cacheable:
        torch.save({name: getattr(data, name) for name in Dataset.__dataclass_fields__}, cache_path)
    return data


def _build_dataset(prefix, tau, failing_credit, depths, keep_all) -> Dataset:
    arrays = load_arrays(prefix)
    keep, efficiency, valid, _ = node_targets(arrays, tau, failing_credit)
    if keep_all:
        keep = np.ones_like(keep)
    depth = arrays.info[:, 0]
    if depths is not None:
        keep &= np.isin(depth, depths)
    kept = np.flatnonzero(keep)
    offsets, counts = arrays.info[:, 5].astype(np.int64), arrays.info[:, 6].astype(np.int64)
    new_offsets = np.zeros(len(kept) + 1, np.int64)
    new_offsets[1:] = np.cumsum(counts[kept])
    move_index = np.repeat(offsets[kept] - new_offsets[:-1], counts[kept]) + np.arange(new_offsets[-1])
    beta = arrays.info[:, 4]
    node = np.repeat(kept, counts[kept])
    score = arrays.labels[move_index, 0]
    return Dataset(
        features=torch.from_numpy(np.ascontiguousarray(arrays.features[kept])),
        aux=torch.from_numpy(np.ascontiguousarray(arrays.aux[kept])),
        depth=torch.from_numpy(depth[kept].astype(np.int8)),
        offsets=torch.from_numpy(new_offsets),
        moves=torch.from_numpy(np.ascontiguousarray(arrays.moves[move_index, :8])),
        target=torch.from_numpy(efficiency[move_index]),
        valid=torch.from_numpy(valid[move_index]),
        cost=torch.from_numpy(np.maximum(arrays.labels[move_index, 1], 1).astype(np.float32)),
        cut=torch.from_numpy(score >= beta[node]),
        root=torch.from_numpy(arrays.info[kept, 7].copy()),
    )


@dataclass
class Batch:
    features: torch.Tensor  # [B, 64] int64, padding mapped to 768
    aux: torch.Tensor       # [B, 8] int64
    depth: torch.Tensor     # [B] int64
    segment: torch.Tensor   # [M] batch row of each move
    slot: torch.Tensor      # [M] position of each move within its row
    moves: torch.Tensor     # [M, 8] int64
    target: torch.Tensor    # [M]
    valid: torch.Tensor     # [M]
    cost: torch.Tensor      # [M]
    cut: torch.Tensor       # [M]
    width: int              # Max moves per row


def make_batch(data: Dataset, rows: torch.Tensor) -> Batch:
    """Gather positions `rows` (on data's device) and their moves into a batch."""
    starts = data.offsets[rows]
    counts = data.offsets[rows + 1] - starts
    segment = torch.repeat_interleave(torch.arange(len(rows), device=rows.device), counts)
    first = torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
    slot = torch.arange(segment.shape[0], device=rows.device) - first
    move_index = torch.repeat_interleave(starts, counts) + slot
    features = data.features[rows].long()
    features = torch.where(features < 0, torch.full_like(features, 768), features)
    return Batch(
        features=features,
        aux=data.aux[rows].long(),
        depth=data.depth[rows].long(),
        segment=segment,
        slot=slot,
        moves=data.moves[move_index].long(),
        target=data.target[move_index],
        valid=data.valid[move_index],
        cost=data.cost[move_index],
        cut=data.cut[move_index],
        width=int(counts.max().item()),
    )


def padded(batch: Batch, values: torch.Tensor, fill: float) -> torch.Tensor:
    """Scatter per-move values into a [B, width] tensor."""
    out = torch.full((batch.features.shape[0], batch.width), fill, dtype=values.dtype, device=values.device)
    out[batch.segment, batch.slot] = values
    return out
