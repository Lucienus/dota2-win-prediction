"""Small reproducible models with explicit source-node resource weighting."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from feature_extractor import NODE_FEATURES, TS_FEATURES, ROLE_COLUMN


class RoleGraphAttention(nn.Module):
    """alpha_ij = softmax_j LeakyReLU(a_i Wh_i + a_j Wh_j + a_e e_ij).

    h'_i = ELU(sum_j alpha_ij * omega_j * Wh_j). The adjacency includes
    self-loops. Ten nodes make an explicit dense implementation inexpensive.
    """
    def __init__(self, input_dim, output_dim, heads=2):
        super().__init__()
        self.heads, self.output_dim = heads, output_dim
        self.project = nn.Linear(input_dim, heads * output_dim, bias=False)
        self.target = nn.Parameter(torch.empty(heads, output_dim))
        self.source = nn.Parameter(torch.empty(heads, output_dim))
        self.edge = nn.Linear(2, heads, bias=False)
        nn.init.xavier_uniform_(self.target)
        nn.init.xavier_uniform_(self.source)

    def forward(self, nodes, edges, roles, complete=False):
        b, n, _ = nodes.shape
        values = self.project(nodes).reshape(b, n, self.heads, self.output_dim)
        left = (values * self.target).sum(-1)
        right = (values * self.source).sum(-1)
        scores = F.leaky_relu(left[:, :, None] + right[:, None, :] + self.edge(edges), 0.2)
        mask = edges.abs().sum(-1) > 0
        mask = mask | torch.eye(n, dtype=torch.bool, device=nodes.device)[None]
        if complete:
            mask = torch.ones_like(mask)
        attention = scores.masked_fill(~mask[..., None], -torch.inf).softmax(dim=2)
        messages = values * roles[:, :, None, None]
        result = torch.einsum("bijh,bjhd->bihd", attention, messages)
        return F.elu(result.mean(dim=2))


class DotaMultiModalPredictor(nn.Module):
    def __init__(self, num_heroes=512, embed_dim=16, hidden_dim=64,
                 temporal="transformer", graph="semantic", use_role=True,
                 draft="cross", temporal_only=False, dropout=0.2):
        super().__init__()
        self.config = dict(num_heroes=num_heroes, embed_dim=embed_dim, hidden_dim=hidden_dim,
                           temporal=temporal, graph=graph, use_role=use_role, draft=draft,
                           temporal_only=temporal_only, dropout=dropout)
        self.temporal, self.graph, self.use_role, self.draft = temporal, graph, use_role, draft
        self.temporal_only = temporal_only
        self.ts_proj = nn.Linear(len(TS_FEATURES), hidden_dim)
        if temporal == "transformer":
            layer = nn.TransformerEncoderLayer(hidden_dim, 4, 2 * hidden_dim, dropout, batch_first=True)
            self.ts_encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        elif temporal == "gru":
            self.ts_encoder = nn.GRU(hidden_dim, hidden_dim, num_layers=2, dropout=dropout, batch_first=True)
        else:
            raise ValueError("Unknown temporal encoder")
        self.ts_attention = nn.Sequential(nn.Linear(hidden_dim, 32), nn.Tanh(), nn.Linear(32, 1))
        if not temporal_only:
            self.hero_embedding = nn.Embedding(num_heroes, embed_dim)
            if draft == "cross":
                self.cross_r = nn.MultiheadAttention(embed_dim, 4, batch_first=True)
                self.cross_d = nn.MultiheadAttention(embed_dim, 4, batch_first=True)
            if draft != "off":
                self.draft_fc = nn.Sequential(nn.Linear(2 * embed_dim, 32), nn.ReLU())
            if graph == "none":
                self.node_mlp = nn.Sequential(nn.Linear(len(NODE_FEATURES) + embed_dim, 64), nn.ReLU(), nn.Linear(64, 32))
            else:
                self.gat1 = RoleGraphAttention(len(NODE_FEATURES) + embed_dim, 64)
                self.gat2 = RoleGraphAttention(64, 32)
        # Separate team pooling preserves which side each node belongs to.
        self.shared = nn.Sequential(nn.Linear(hidden_dim + (0 if temporal_only else 96), 64),
                                    nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout))
        self.win_head = nn.Linear(64, 1)
        self.aux_head = nn.Linear(64, 2)

    def forward(self, batch, uniform_attention=False):
        sequence, lengths = batch["sequence"], batch["lengths"]
        b, length, _ = sequence.shape
        padding = torch.arange(length, device=sequence.device)[None] >= lengths[:, None]
        z = self.ts_proj(sequence)
        if self.temporal == "transformer":
            positions = torch.arange(length, device=z.device, dtype=z.dtype)[:, None]
            divisor = torch.exp(torch.arange(0, z.shape[-1], 2, device=z.device, dtype=z.dtype) * (-math.log(10000) / z.shape[-1]))
            pe = torch.zeros_like(z[0])
            pe[:, 0::2], pe[:, 1::2] = torch.sin(positions * divisor), torch.cos(positions * divisor)
            # Bidirectional processing of the observed prefix is intentional; no future is supplied.
            z = self.ts_encoder(z + pe, src_key_padding_mask=padding)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
            encoded, _ = self.ts_encoder(packed)
            z, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=length)
        scores = self.ts_attention(z).squeeze(-1)
        if uniform_attention:
            scores = torch.zeros_like(scores)
        weights = scores.masked_fill(padding, -torch.inf).softmax(dim=1)
        ts = (z * weights[..., None]).sum(1)
        representations = [ts]
        if not self.temporal_only:
            heroes = self.hero_embedding(batch["heroes"])
            r, d = heroes[:, :5], heroes[:, 5:]
            if self.draft == "cross":
                ra, _ = self.cross_r(r, d, d, need_weights=False)
                da, _ = self.cross_d(d, r, r, need_weights=False)
                r, d = r + ra, d + da
            draft = self.draft_fc(torch.cat((r.mean(1), d.mean(1)), dim=-1)) if self.draft != "off" else torch.zeros(b, 32, device=z.device)
            nodes = batch["nodes"].clone()
            roles = batch["roles"] if self.use_role else torch.ones_like(batch["roles"])
            if not self.use_role:
                nodes[..., ROLE_COLUMN] = 0
            nodes = torch.cat((nodes, heroes), dim=-1)
            if self.graph == "none":
                nodes = self.node_mlp(nodes)
            else:
                nodes = self.gat1(nodes, batch["edges"], roles, complete=self.graph == "complete")
                nodes = self.gat2(nodes, batch["edges"], roles, complete=self.graph == "complete")
            representations += [draft, nodes[:, :5].mean(1), nodes[:, 5:].mean(1)]
        shared = self.shared(torch.cat(representations, dim=-1))
        return {"logits": self.win_head(shared).squeeze(-1), "attention": weights, "aux": self.aux_head(shared)}


VARIANTS = {
    "full": {}, "gru": {"temporal": "gru"}, "no_graph": {"graph": "none"},
    "no_role": {"use_role": False}, "complete_graph": {"graph": "complete"},
    "no_draft_branch": {"draft": "off"}, "mean_draft": {"draft": "mean"},
    "single_task": {}, "temporal_only": {"temporal_only": True},
}
