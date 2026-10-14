"""CoTaReS: count-Tanimoto reference-residual molecular regressor."""

import torch
from torch import nn
import torch.nn.functional as F


class CoTaReS(nn.Module):
    """A learned count baseline corrected by a local, full-Gram residual solve."""

    def __init__(self, features=8192, width=64, dropout=0.1):
        super().__init__()
        self.environment = nn.Parameter(torch.randn(features, width) * 0.025)
        self.potential = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 128), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(128, 1),
        )
        # Preserve the archived initialization stream after removing inactive
        # experimental branches. These temporary layers are not part of the model.
        nn.Linear(width, width)
        nn.Linear(width, width, bias=False)
        nn.Linear(width, 64)
        nn.Linear(64, 1)
        nn.LayerNorm(width)
        nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.Sequential(nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.Linear(width, 1)
        self.kernel_norm = nn.LayerNorm(width)
        self.kernel_metric = nn.Linear(width, width)
        self.kernel_ridge = nn.Parameter(torch.tensor(-3.0))

    def encode(self, counts):
        return counts @ self.environment / counts.sum(-1, keepdim=True).clamp_min(1).sqrt()

    @staticmethod
    def count_kernel_gram(counts, weights):
        weighted = counts * weights.sqrt()[:, None, :]
        dot = weighted @ weighted.transpose(-1, -2)
        norm = weighted.square().sum(-1)
        denominator = (norm[:, :, None] + norm[:, None, :] - dot).clamp_min(1e-8)
        return dot / denominator

    def forward(self, query, references, reference_y):
        base = self.potential(self.encode(query)).squeeze(-1)
        query_state = self.kernel_norm(self.encode(query))
        feature_weights = torch.exp(
            2 * torch.tanh(self.kernel_metric(query_state) @ self.environment.T)
        )
        all_counts = torch.cat([query[:, None, :], references], dim=1)
        similarity = self.count_kernel_gram(all_counts, feature_weights)
        query_reference = similarity[:, 0, 1:]
        reference_gram = similarity[:, 1:, 1:]
        reference_base = self.potential(self.encode(references)).squeeze(-1)
        ridge = F.softplus(self.kernel_ridge) + 0.001
        eye = torch.eye(references.shape[1], device=query.device, dtype=query.dtype)
        coefficients = torch.linalg.solve(
            reference_gram + ridge * eye, query_reference.unsqueeze(-1)
        ).squeeze(-1)
        prediction = base + (coefficients * (reference_y - reference_base)).sum(-1)
        return prediction, {"base": base, "weights": coefficients}
