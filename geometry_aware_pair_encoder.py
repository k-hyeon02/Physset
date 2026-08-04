import torch
import torch.nn as nn

class GeometryAwarePairEncoder(nn.Module):
    """
    모든 마이크 pair에 동일한 네트워크 파라미터를 적용하는 shared pair encoder
    """
    def __init__(self, hidden_dim=64):
        super().__init__()

        # 5 acoustic + 8 geometry + 1 frequency = 14
        self.linear_projection = nn.Conv2d(14, hidden_dim, kernel_size=1)
        self.frequency_block = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3,1), padding=(1,0))
        self.temporal_block = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1,3), padding=(0,1))
        self.activation = nn.GELU()

    def forward(self, pair_features, pair_geometry):
        """
        Args:
            pair_features: (B, P, 5, K, T)
            pair_geometry: (B, P, 8)

        Returns:
            pair_embedding: (B, P, hidden_dim, K, T)
        """
        B, P, C, K, T = pair_features.shape
        geometry = pair_geometry[..., None, None]  # (B, P, 8, 1, 1)
        geometry = geometry.expand(B, P, 8, K, T)  # (B, P, 8, K, T)

        frequency = torch.arange(K, device=pair_features.device, dtype=pair_features.dtype)/K  # 0, ..., K-1/K
        frequency = frequency.view(1, 1, 1, K, 1)    # (1, 1, 1, K, 1)
        frequency = frequency.expand(B, P, 1, K, T)  # (B, P, 1, K, T)

        # acoustic features
        x = torch.cat([pair_features, geometry, frequency], dim=2)  # (B, P, 14, K, T)
        x = x.reshape(B*P, 14, K, T)

        # Linear projection
        x = self.activation(self.linear_projection(x))

        # Frequency block
        x = self.activation(self.frequency_block(x))

        # Temporal block
        x = self.activation(self.temporal_block(x))

        return x.reshape(B, P, -1, K, T)

