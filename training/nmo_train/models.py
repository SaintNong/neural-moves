"""Move-ordering network architectures.

Each model maps a Batch to one logit per move (flattened over the batch).
Squares are relative to the side to move. Every model can export its weights
as named float tensors for the C++ inference in src/neural/.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .data import Batch

PADDING = 768
DEPTH_BUCKETS = 4  # Quiescence, depth 1, depth 2, depth >= 3.
THREATS = 256      # Threat-bit combinations (features::threat_combinations).


def depth_bucket(depth: torch.Tensor) -> torch.Tensor:
    return depth.clamp(max=DEPTH_BUCKETS - 1)


def activate(x: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu":
        return F.relu(x)
    if kind == "crelu":
        return x.clamp(0.0, 1.0)
    if kind == "screlu":
        return x.clamp(0.0, 1.0).square()
    raise ValueError(f"unknown activation {kind}")


class Accumulator(nn.Module):
    """NNUE feature transformer: a bias plus one learned row per active piece-square feature."""

    def __init__(self, hidden: int, scale: float = 1.0):
        super().__init__()
        self.rows = nn.EmbeddingBag(PADDING + 1, hidden, mode="sum", padding_idx=PADDING)
        nn.init.normal_(self.rows.weight, std=scale / math.sqrt(32))
        with torch.no_grad():
            self.rows.weight[PADDING].zero_()
        self.bias = nn.Parameter(torch.full((hidden,), 0.1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.rows(features) + self.bias

    def export(self, prefix: str) -> dict:
        return {f"{prefix}.weight": self.rows.weight[:PADDING], f"{prefix}.bias": self.bias}


class MoveMap(nn.Module):
    """Neural MoveMap (Kocsis et al., 2003): one hidden layer over the board and
    an output map with a unit per from-square and per to-square; a move scores
    the sum of its two units. Evaluated from scratch at every node."""

    arch = "movemap"

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.input = Accumulator(hidden)
        self.output = nn.Linear(hidden, 128)

    def forward(self, batch: Batch) -> torch.Tensor:
        h = F.relu(self.input(batch.features[:, :32]))
        out = self.output(h)
        return out[batch.segment, batch.moves[:, 0]] + out[batch.segment, 64 + batch.moves[:, 1]]

    def export(self) -> tuple[dict, dict]:
        tensors = self.input.export("input")
        tensors.update({"output.weight": self.output.weight, "output.bias": self.output.bias})
        return tensors, {"hidden": self.hidden}


class NNUEPolicy(nn.Module):
    """Efficiently updatable policy in the style of Monty: dual-perspective
    accumulators feed a per-move output vector indexed by (from, to)."""

    arch = "nnue-policy"

    def __init__(self, hidden: int = 64, activation: str = "crelu"):
        super().__init__()
        self.hidden, self.activation = hidden, activation
        self.input = Accumulator(hidden)
        self.move = nn.Embedding(64 * 64, 2 * hidden)
        self.move_bias = nn.Embedding(64 * 64, 1)
        self.promotion = nn.Embedding(5, 2 * hidden)
        self.promotion_bias = nn.Embedding(5, 1)
        for table in (self.move, self.promotion):
            nn.init.normal_(table.weight, std=0.1 / math.sqrt(hidden))
        for table in (self.move_bias, self.promotion_bias):
            nn.init.zeros_(table.weight)

    def hidden_state(self, batch: Batch) -> torch.Tensor:
        own = activate(self.input(batch.features[:, :32]), self.activation)
        theirs = activate(self.input(batch.features[:, 32:]), self.activation)
        return torch.cat([own, theirs], dim=1)

    def forward(self, batch: Batch) -> torch.Tensor:
        h = self.hidden_state(batch)[batch.segment]
        m = batch.moves
        slot = m[:, 0] * 64 + m[:, 1]
        vector = self.move(slot) + self.promotion(m[:, 4])
        return (vector * h).sum(-1) + self.move_bias(slot)[:, 0] + self.promotion_bias(m[:, 4])[:, 0]

    def export(self) -> tuple[dict, dict]:
        tensors = self.input.export("input")
        tensors.update({
            "move.weight": self.move.weight, "move.bias": self.move_bias.weight[:, 0],
            "promotion.weight": self.promotion.weight, "promotion.bias": self.promotion_bias.weight[:, 0],
        })
        return tensors, {"hidden": self.hidden, "activation": ["relu", "crelu", "screlu"].index(self.activation)}


class UPN(NNUEPolicy):
    """Updatable Policy Network (ours). Adds to the NNUE policy cheap per-move
    tactical context and the remaining search depth:

    * a learned bias per (depth bucket, moving piece, victim, threat bits), where
      threat bits say whether the from/to squares are attacked, attacked by a
      lower-valued piece or defended, whether the move gives direct check, and
      whether another own piece remains attacked;
    * a low-rank interaction between that context and the position: the
      hidden state is projected to `context` dims and dotted with embeddings of
      (piece, threat bits) and (piece, victim, promotion).

    The accumulator is incrementally updated in search; the tactical features
    are a handful of bitboard tests per move.
    """

    arch = "upn"

    def __init__(self, hidden: int = 64, activation: str = "crelu", context: int = 16, depth_aware: bool = True):
        super().__init__(hidden, activation)
        self.context, self.depth_aware = context, depth_aware
        buckets = DEPTH_BUCKETS if depth_aware else 1
        self.tactic_bias = nn.Embedding(buckets * 6 * 7 * THREATS, 1)
        nn.init.zeros_(self.tactic_bias.weight)
        if context:  # Zero disables the low-rank interaction.
            self.project = nn.Linear(2 * hidden, context)
            self.threat = nn.Embedding(buckets * 6 * THREATS, context)
            self.capture = nn.Embedding(6 * 7 * 5, context)
            for table in (self.threat, self.capture):
                nn.init.normal_(table.weight, std=0.1 / math.sqrt(context))

    def forward(self, batch: Batch) -> torch.Tensor:
        state = self.hidden_state(batch)
        g = self.project(state)[batch.segment] if self.context else None
        h = state[batch.segment]
        m = batch.moves
        bucket = depth_bucket(batch.depth)[batch.segment] if self.depth_aware else torch.zeros_like(batch.segment)
        slot = m[:, 0] * 64 + m[:, 1]
        piece, victim, promotion, threats = m[:, 2], m[:, 3], m[:, 4], m[:, 5]
        vector = self.move(slot) + self.promotion(promotion)
        logit = (vector * h).sum(-1) + self.move_bias(slot)[:, 0] + self.promotion_bias(promotion)[:, 0]
        logit = logit + self.tactic_bias(((bucket * 6 + piece) * 7 + victim) * THREATS + threats)[:, 0]
        if not self.context:
            return logit
        context = self.threat((bucket * 6 + piece) * THREATS + threats) + self.capture((piece * 7 + victim) * 5 + promotion)
        return logit + (context * g).sum(-1)

    def export(self) -> tuple[dict, dict]:
        tensors, settings = super().export()
        tensors["tactic.bias"] = self.tactic_bias.weight[:, 0]
        if self.context:
            tensors.update({
                "project.weight": self.project.weight, "project.bias": self.project.bias,
                "threat.weight": self.threat.weight, "capture.weight": self.capture.weight,
            })
        settings.update({"context": self.context, "depth_buckets": DEPTH_BUCKETS if self.depth_aware else 1})
        return tensors, settings


class TacticalTable(nn.Module):
    """Learned biases per (depth bucket, piece, victim, threat bits), per
    (from, to) and per promotion: UPN without a network, also used as UPN's
    cheap leaf head near the leaves of the tree."""

    arch = "tactical"

    def __init__(self):
        super().__init__()
        self.tactic_bias = nn.Embedding(DEPTH_BUCKETS * 6 * 7 * THREATS, 1)
        self.move_bias = nn.Embedding(64 * 64, 1)
        self.promotion_bias = nn.Embedding(5, 1)
        for table in (self.tactic_bias, self.move_bias, self.promotion_bias):
            nn.init.zeros_(table.weight)

    def forward(self, batch: Batch) -> torch.Tensor:
        m = batch.moves
        bucket = depth_bucket(batch.depth)[batch.segment]
        tactic = ((bucket * 6 + m[:, 2]) * 7 + m[:, 3]) * THREATS + m[:, 5]
        return (self.tactic_bias(tactic) + self.move_bias(m[:, 0] * 64 + m[:, 1]) + self.promotion_bias(m[:, 4]))[:, 0]

    def export(self) -> tuple[dict, dict]:
        return {"tactic.bias": self.tactic_bias.weight[:, 0], "move.bias": self.move_bias.weight[:, 0],
                "promotion.bias": self.promotion_bias.weight[:, 0]}, {"depth_buckets": DEPTH_BUCKETS}


def board_planes(batch: Batch) -> torch.Tensor:
    """[B, 18, 64] planes: 12 piece planes (side to move first), 4 castling
    planes, an en passant plane and a constant plane."""
    features = batch.features[:, :32]
    rows = features.shape[0]
    planes = torch.zeros(rows, 18 * 64, device=features.device)
    valid = features < PADDING
    index = torch.where(valid, features, torch.zeros_like(features))  # Feature index = plane * 64 + square.
    planes.scatter_(1, index, valid.float())
    planes = planes.view(rows, 18, 64)
    castling = batch.aux[:, 0]
    for bit in range(4):
        planes[:, 12 + bit] = ((castling >> bit) & 1).float()[:, None]
    ep_file = batch.aux[:, 1]
    has_ep = ep_file > 0
    planes[has_ep, 16, 40 + ep_file[has_ep] - 1] = 1.0
    planes[:, 17] = 1.0
    return planes


# AlphaZero policy encoding: 56 queen-move planes (8 directions x 7 distances),
# 8 knight planes, 9 underpromotion planes (3 pieces x 3 directions).
QUEEN_DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]


def alphazero_plane_table() -> torch.Tensor:
    """[64 * 64 * 5] plane of every (from, to, promotion) triple, -1 if none."""
    table = torch.full((64 * 64 * 5,), -1, dtype=torch.long)
    for origin in range(64):
        for target in range(64):
            df, dr = target % 8 - origin % 8, target // 8 - origin // 8
            if (df, dr) == (0, 0):
                continue
            base = None
            if (df, dr) in KNIGHT_DELTAS:
                base = 56 + KNIGHT_DELTAS.index((df, dr))
            elif df == 0 or dr == 0 or abs(df) == abs(dr):
                direction = QUEEN_DIRECTIONS.index((int(math.copysign(1, df)) if df else 0, int(math.copysign(1, dr)) if dr else 0))
                base = direction * 7 + max(abs(df), abs(dr)) - 1
            if base is None:
                continue
            for promotion in range(5):
                plane = base
                if promotion in (1, 2, 3) and abs(df) <= 1 and dr == 1:
                    plane = 64 + (promotion - 1) * 3 + df + 1
                table[(origin * 64 + target) * 5 + promotion] = plane
    return table


class ConvPolicy(nn.Module):
    """AlphaZero/Lc0-style residual CNN with a convolutional 73x64 policy head."""

    arch = "cnn"

    def __init__(self, filters: int = 32, blocks: int = 4):
        super().__init__()
        self.filters, self.blocks = filters, blocks
        conv = lambda cin, cout, k: nn.Conv2d(cin, cout, k, padding=k // 2, bias=False)
        self.stem = nn.Sequential(conv(18, filters, 3), nn.BatchNorm2d(filters))
        self.tower = nn.ModuleList(
            nn.ModuleList([conv(filters, filters, 3), nn.BatchNorm2d(filters), conv(filters, filters, 3), nn.BatchNorm2d(filters)])
            for _ in range(blocks))
        self.head = nn.Sequential(conv(filters, filters, 3), nn.BatchNorm2d(filters))
        self.policy = nn.Conv2d(filters, 73, 1)
        self.register_buffer("planes", alphazero_plane_table(), persistent=False)

    def forward(self, batch: Batch) -> torch.Tensor:
        x = board_planes(batch).view(-1, 18, 8, 8)
        x = F.relu(self.stem(x))
        for conv1, bn1, conv2, bn2 in self.tower:
            x = F.relu(x + bn2(conv2(F.relu(bn1(conv1(x))))))
        x = self.policy(F.relu(self.head(x))).flatten(2)  # [B, 73, 64]
        m = batch.moves
        plane = self.planes[(m[:, 0] * 64 + m[:, 1]) * 5 + m[:, 4]]
        return x[batch.segment, plane, m[:, 0]]

    def export(self) -> tuple[dict, dict]:
        tensors = {}

        def fold(name, conv, bn):
            # Fold inference-mode batch norm into the preceding convolution.
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            tensors[f"{name}.weight"] = conv.weight * scale[:, None, None, None]
            tensors[f"{name}.bias"] = bn.bias - bn.running_mean * scale

        fold("stem", self.stem[0], self.stem[1])
        for index, (conv1, bn1, conv2, bn2) in enumerate(self.tower):
            fold(f"block{index}.conv1", conv1, bn1)
            fold(f"block{index}.conv2", conv2, bn2)
        fold("head", self.head[0], self.head[1])
        tensors["policy.weight"] = self.policy.weight
        tensors["policy.bias"] = self.policy.bias
        return tensors, {"filters": self.filters, "blocks": self.blocks}


class AttentionPolicy(nn.Module):
    """Transformer over the 64 squares with an Lc0-style attention policy head:
    logit(from, to) = q_from . k_to / sqrt(d), plus a promotion term."""

    arch = "attention"

    def __init__(self, dim: int = 64, layers: int = 2, heads: int = 4, ffn: int = 128):
        super().__init__()
        self.dim, self.layers, self.heads, self.ffn = dim, layers, heads, ffn
        self.embed = nn.Linear(18, dim)
        self.position = nn.Parameter(torch.randn(64, dim) * 0.02)
        self.norm1 = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))
        self.qkv = nn.ModuleList(nn.Linear(dim, 3 * dim) for _ in range(layers))
        self.attn_out = nn.ModuleList(nn.Linear(dim, dim) for _ in range(layers))
        self.norm2 = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))
        self.ffn1 = nn.ModuleList(nn.Linear(dim, ffn) for _ in range(layers))
        self.ffn2 = nn.ModuleList(nn.Linear(ffn, dim) for _ in range(layers))
        self.final_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.promotion = nn.Embedding(5, dim)
        nn.init.zeros_(self.promotion.weight)

    def forward(self, batch: Batch) -> torch.Tensor:
        x = self.embed(board_planes(batch).transpose(1, 2)) + self.position  # [B, 64, d]
        rows, head_dim = x.shape[0], self.dim // self.heads
        for layer in range(self.layers):
            q, k, v = self.qkv[layer](self.norm1[layer](x)).split(self.dim, dim=-1)
            split = lambda t: t.view(rows, 64, self.heads, head_dim).transpose(1, 2)
            attention = F.scaled_dot_product_attention(split(q), split(k), split(v))
            x = x + self.attn_out[layer](attention.transpose(1, 2).reshape(rows, 64, self.dim))
            x = x + self.ffn2[layer](F.relu(self.ffn1[layer](self.norm2[layer](x))))
        x = self.final_norm(x)
        q, k = self.query(x), self.key(x)
        m = batch.moves
        key = k[batch.segment, m[:, 1]]
        return ((q[batch.segment, m[:, 0]] + self.promotion(m[:, 4])) * key).sum(-1) / math.sqrt(self.dim)

    def export(self) -> tuple[dict, dict]:
        tensors = {"embed.weight": self.embed.weight, "embed.bias": self.embed.bias, "position": self.position,
                   "final_norm.weight": self.final_norm.weight, "final_norm.bias": self.final_norm.bias,
                   "query.weight": self.query.weight, "query.bias": self.query.bias,
                   "key.weight": self.key.weight, "key.bias": self.key.bias, "promotion.weight": self.promotion.weight}
        for layer in range(self.layers):
            for name, module in (("norm1", self.norm1), ("qkv", self.qkv), ("attn_out", self.attn_out),
                                 ("norm2", self.norm2), ("ffn1", self.ffn1), ("ffn2", self.ffn2)):
                tensors[f"layer{layer}.{name}.weight"] = module[layer].weight
                tensors[f"layer{layer}.{name}.bias"] = module[layer].bias
        return tensors, {"dim": self.dim, "layers": self.layers, "heads": self.heads, "ffn": self.ffn}


ARCHITECTURES = {cls.arch: cls for cls in (MoveMap, NNUEPolicy, UPN, TacticalTable, ConvPolicy, AttentionPolicy)}
