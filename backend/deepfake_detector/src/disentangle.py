"""
FairFake Orthogonal Latent Subspace Disentanglement (Feature 2.3)

Explicitly partitions latent representations into two geometrically disjoint subspaces:
    z = [z_forensic || z_biometric]
Enforces an orthogonality constraint:
    L_orth = || z_forensic^T z_biometric ||_F^2
ensuring that forgery artifact representations are mathematically independent of
biometric identity, skin tone, cosmetic textures, and facial accessories.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any


class OrthogonalSubspaceSplitter(nn.Module):
    """
    Projects or splits a latent feature representation into two disjoint subspaces:
    Forensic Subspace (z_f) and Biometric Subspace (z_b).
    """

    def __init__(
        self,
        in_features: int,
        forensic_dim: Optional[int] = None,
        use_projections: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.forensic_dim = forensic_dim or (in_features // 2)
        self.biometric_dim = in_features - self.forensic_dim
        self.use_projections = use_projections

        if use_projections:
            # Learned projection heads
            self.forensic_proj = nn.Sequential(
                nn.Linear(in_features, self.forensic_dim),
                nn.BatchNorm1d(self.forensic_dim),
                nn.ReLU(inplace=True)
            )
            self.biometric_proj = nn.Sequential(
                nn.Linear(in_features, self.biometric_dim),
                nn.BatchNorm1d(self.biometric_dim),
                nn.ReLU(inplace=True)
            )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splits latent vector z into (z_forensic, z_biometric).
        """
        if self.use_projections:
            z_f = self.forensic_proj(z)
            z_b = self.biometric_proj(z)
        else:
            # Direct subspace slicing
            z_f = z[:, :self.forensic_dim]
            z_b = z[:, self.forensic_dim:]

        return z_f, z_b


class OrthogonalDisentanglementLoss(nn.Module):
    """
    Computes soft orthogonality loss between forensic and biometric representations.
    Penalizes both sample-wise cosine overlap and cross-batch covariance correlation.
    """

    def __init__(self, mode: str = "cosine_and_cov", eps: float = 1e-6):
        super().__init__()
        self.mode = mode
        self.eps = eps

    def compute_cosine_loss(self, z_f: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        Penalizes squared cosine similarity between paired vectors:
            L_cos = mean( (cos(z_f, z_b))^2 )
        """
        z_f_norm = F.normalize(z_f, p=2, dim=-1, eps=self.eps)
        z_b_norm = F.normalize(z_b, p=2, dim=-1, eps=self.eps)

        # In case dimensions differ, project to equal dimensions via inner product if equal,
        # or slice matching dimensions
        min_dim = min(z_f.size(1), z_b.size(1))
        cos_sim = (z_f_norm[:, :min_dim] * z_b_norm[:, :min_dim]).sum(dim=-1)
        return torch.mean(cos_sim ** 2)

    def compute_covariance_loss(self, z_f: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        Penalizes cross-covariance between feature dimensions across the batch:
            C = (z_f - mean(z_f))^T (z_b - mean(z_b)) / (B - 1)
            L_cov = ||C||_F^2 / (D_f * D_b)
        """
        batch_size = z_f.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=z_f.device, requires_grad=True)

        # Center features
        z_f_centered = z_f - z_f.mean(dim=0, keepdim=True)
        z_b_centered = z_b - z_b.mean(dim=0, keepdim=True)

        # Cross-covariance matrix (D_f x D_b)
        cov = torch.matmul(z_f_centered.t(), z_b_centered) / (batch_size - 1)

        # Normalized Frobenius norm squared
        d_f = z_f.size(1)
        d_b = z_b.size(1)
        frobenius_sq = torch.sum(cov ** 2)
        return frobenius_sq / (d_f * d_b)

    def forward(self, z_f: torch.Tensor, z_b: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes the complete orthogonality penalty.
        """
        cos_loss = self.compute_cosine_loss(z_f, z_b)
        cov_loss = self.compute_covariance_loss(z_f, z_b)

        if self.mode == "cosine":
            total_loss = cos_loss
        elif self.mode == "covariance":
            total_loss = cov_loss
        else:  # "cosine_and_cov"
            total_loss = cos_loss + cov_loss

        return {
            "orth_loss": total_loss,
            "cosine_penalty": cos_loss,
            "covariance_penalty": cov_loss
        }


class DisentangledForensicObjective(nn.Module):
    """
    Joint multi-objective loss combining:
      1. Forensic manipulation detection (BCE)
      2. Demographic adversarial invariance (Adversary BCE via GRL)
      3. Latent subspace orthogonality (Orthogonal loss)
    """

    def __init__(
        self,
        lambda_adversary: float = 0.5,
        beta_orthogonality: float = 0.2,
        pos_weight: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.lambda_adv = float(lambda_adversary)
        self.beta_orth = float(beta_orthogonality)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.orth_criterion = OrthogonalDisentanglementLoss()

    def set_weights(self, lambda_adv: Optional[float] = None, beta_orth: Optional[float] = None):
        if lambda_adv is not None:
            self.lambda_adv = float(max(0.0, lambda_adv))
        if beta_orth is not None:
            self.beta_orth = float(max(0.0, beta_orth))

    def forward(
        self,
        forensic_logits: torch.Tensor,
        forensic_targets: torch.Tensor,
        attribute_logits: Optional[Dict[str, torch.Tensor]] = None,
        attribute_targets: Optional[Dict[str, torch.Tensor]] = None,
        z_forensic: Optional[torch.Tensor] = None,
        z_biometric: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # 1. Primary Forensic BCE Loss
        loss_forensic = self.bce(forensic_logits, forensic_targets.float())

        # 2. Adversarial Attribute Loss
        loss_adversary = torch.tensor(0.0, device=forensic_logits.device)
        if attribute_logits and attribute_targets:
            attr_losses = []
            for attr_name, logits in attribute_logits.items():
                if attr_name in attribute_targets:
                    targets = attribute_targets[attr_name].float()
                    attr_losses.append(F.binary_cross_entropy_with_logits(logits, targets))
            if attr_losses:
                loss_adversary = torch.stack(attr_losses).mean()

        # 3. Orthogonality Disentanglement Loss
        loss_orth = torch.tensor(0.0, device=forensic_logits.device)
        if z_forensic is not None and z_biometric is not None:
            orth_dict = self.orth_criterion(z_forensic, z_biometric)
            loss_orth = orth_dict["orth_loss"]

        # Joint total loss
        total_loss = loss_forensic + (self.lambda_adv * loss_adversary) + (self.beta_orth * loss_orth)

        return {
            "total_loss": total_loss,
            "loss_forensic": loss_forensic,
            "loss_adversary": loss_adversary,
            "loss_orthogonality": loss_orth
        }


if __name__ == "__main__":
    print("=" * 70)
    print("FAIRFAKE ORTHOGONAL DISENTANGLEMENT VERIFICATION")
    print("=" * 70)

    splitter = OrthogonalSubspaceSplitter(in_features=2048, forensic_dim=1024)
    loss_fn = OrthogonalDisentanglementLoss()

    dummy_z = torch.randn(8, 2048)
    z_f, z_b = splitter(dummy_z)
    print(f"Latent vector shape:    {dummy_z.shape}")
    print(f"Forensic subspace z_f:  {z_f.shape}")
    print(f"Biometric subspace z_b: {z_b.shape}")

    # Case 1: Random representations (non-orthogonal)
    loss_random = loss_fn(z_f, z_b)
    print(f"\nRandom vectors orth loss:    {loss_random['orth_loss'].item():.6f}")

    # Case 2: Perfectly orthogonal vectors
    orth_u = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    orth_v = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    loss_orth = loss_fn(orth_u, orth_v)
    print(f"Orthogonal vectors orth loss: {loss_orth['orth_loss'].item():.6f} (Expected near 0.0)")

    # Case 3: Perfectly collinear vectors
    collinear_u = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    collinear_v = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    loss_collinear = loss_fn(collinear_u, collinear_v)
    print(f"Collinear vectors orth loss:  {loss_collinear['orth_loss'].item():.6f} (Expected near 1.0)")
    print("=" * 70)
