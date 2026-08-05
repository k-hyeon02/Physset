import torch
import torch.nn as nn

from utils import directed_features
from geometry_aware_pair_encoder import GeometryAwarePairEncoder

class EdgeToNodeAttention(nn.Module):
    """
    Edge embedding h_ij -> Node embedding h_i (permutatjon-invariant attention)
    각 마이크(node) i에 연결된 여러 pair(edge) embedding h_ij들을 attention으로 가중합해서
    하나의 node embedding h_i로 압축

        alpha_ij = softmax_j( (W_q q_i)^T (W_k h_ij) / sqrt(d) )
        h_i = sum_j alpha_ij * (W_v h_ij)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_q = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.W_k = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.W_v = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, pair_embedding, mic_pairs, num_mic):
        """
        Args:
            pair_embedding: (B, P, C, K, T)  -- GAPE 출력 (ordered pair 기준, P = M*(M-1))
            mic_pairs: pair 축과 동일한 순서의 (i, j) 인덱스 목록, list[tuple[int, int]]
                ex) [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
            num_mic: M

        Returns:
            node_embedding: (B, M, C, K, T)
        """

        B, P, C, K, T = pair_embedding.shape
        M = num_mic

        i = torch.tensor([p[0] for p in mic_pairs], device=pair_embedding.device)  # (P, )
        j = torch.tensor([p[1] for p in mic_pairs], device=pair_embedding.device)  # (P, )

        # GAPE에서 나온 pair_embedding의 P축이 i 기준으로 정렬돼 있지 않을 수 있음
        # 같은 i에 연결된 edge끼리 뭉쳐서 (M, M-1) 모양으로 펴기 위해 정렬 인덱스를 구함
        order = torch.argsort(i*M + j)  # argsort: 텐서 값 정렬했을 때 그 인덱스를 반환
        # ex) [1, 2, 3, 5, 6, 7] -> [0, 1, 2, 3, 4, 5]

        def to_dense(x):
            return x.index_select(dim=1, index=order).view(B, M, M-1, C, K, T)

        # 1. q_i 초기값: i에 인접한 edge embedding들의 평균 (초기 node representation)
        h_dense = to_dense(pair_embedding)  # (B, M, M-1, C, K, T)
        q_init = h_dense.mean(dim=2)  # (B, M, C, K, T)
        q = self.W_q(q_init.reshape(B * M, C, K, T)).view(B, M, C, K, T)

        # 2. key/value = W_k h_ij, W_v h_ij (B*P를 배치로 펴서 Conv2d)
        key = self.W_k(pair_embedding.reshape(B*P, C, K, T)).view(B, P, C, K, T)
        value = self.W_v(pair_embedding.reshape(B*P, C, K, T)).view(B, P, C, K, T)

        # 3. key/value를 node i 기준 dense (B, M, M-1, C, K, T)로 재배열
        key_dense = to_dense(key)
        value_dense = to_dense(value)

        # 4. score = (W_q q_i)^T (W_k h_ij) / sqrt(d)
        scale = C**-0.5
        score = (q.unsqueeze(2) * key_dense).sum(dim=3) * scale  # (B, M, M-1, K, T)

        # 5. softmax(dim=n)
        alpha = torch.softmax(score, dim=2)  # (B, M, M-1, K, T)

        # 6. h_i = sum_j alpha_ij * (W_v h_ij)
        node_embedding = (alpha.unsqueeze(3) * value_dense).sum(dim=2)  # (B, M, C, K, T)

        return node_embedding


class SetSelfAttention(nn.Module):
    """
    Node embedding h_i들끼리 self-attention으로 서로 문맥을 주고받아 집합 표현 H = E_set({h_i})를 계산
    (Set Transformer의 SAB - Set Attention Block. 완전 그래프이므로 GAT의 masked attention과 동치)

        alpha_ii' = softmax_i'( (W_q h_i)^T (W_k h_i') / sqrt(d) )
        h_i' = h_i + sum_i' alpha_ii' * (W_v h_i')      (self-attention + residual)
        h_i'' = h_i' + rFF(h_i')                        (row-wise feedforward + residual)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_q = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.W_k = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.W_v = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm1 = nn.GroupNorm(1, hidden_dim)  # 채널 축 기준 LayerNorm 대용

        self.rff = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
        )
        self.norm2 = nn.GroupNorm(1, hidden_dim)

    def forward(self, node_embedding):
        """
        Args:
            node_embedding: (B, M, C, K, T)  -- EdgeToNodeAttention 출력

        Returns:
            set_embedding: (B, M, C, K, T)
        """
        B, M, C, K, T = node_embedding.shape
        x = node_embedding.reshape(B * M, C, K, T)

        # 1. q_i = W_q h_i, k_i = W_k h_i, v_i = W_v h_i
        q = self.W_q(x).view(B, M, C, K, T)
        k = self.W_k(x).view(B, M, C, K, T)
        v = self.W_v(x).view(B, M, C, K, T)

        # 2. score_ii' = (W_q h_i)^T (W_k h_i') / sqrt(d)  -- 모든 node 쌍(i, i') 사이의 attention
        scale = C**-0.5
        score = torch.einsum("bmckt,bnckt->bmnkt", q, k) * scale  # (B, M, M, K, T)

        # 3. softmax(dim=i')
        alpha = torch.softmax(score, dim=2)  # (B, M, M, K, T)

        # 4. h_i' = sum_i' alpha_ii' * (W_v h_i')
        attended = torch.einsum("bmnkt,bnckt->bmckt", alpha, v)  # (B, M, C, K, T)
        attended = self.out_proj(attended.reshape(B * M, C, K, T))  # (B*M, C, K, T)

        # 5. residual + norm
        h = self.norm1(x + attended)  # (B*M, C, K, T)

        # 6. row-wise feedforward + residual + norm
        h = self.norm2(h + self.rff(h))  # (B*M, C, K, T)

        return h.view(B, M, C, K, T)


class MicSetGraphEncoder(nn.Module):
    """
    파이프라인 전체를 하나로 묶는 진입점
    directed_features(4) -> GeometryAwarePairEncoder(5) -> EdgeToNodeAttention(6) -> SetSelfAttention(7)
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.pair_encoder = GeometryAwarePairEncoder(hidden_dim)
        self.edge_to_node = EdgeToNodeAttention(hidden_dim)
        self.set_encoder = SetSelfAttention(hidden_dim)

    def forward(self, acoustic_features, geometry_features, mic_pairs):
        """
        Args:
            acoustic_features: (B, P, 5, K, T) -- i<j 순서의 pair 음향 특징
            geometry_features: (B, P, 8) -- i<j 순서의 pair 기하 특징
            mic_pairs: pair 축과 동일한 순서의 (i, j) 인덱스 목록, i<j

        Returns:
            set_embedding: (B, M, hidden_dim, K, T) -- 마이크 집합 전체의 최종 표현 H
        """
        num_mic = max(max(p) for p in mic_pairs) + 1

        directed_acoustic, directed_geometry, directed_mic_pairs = directed_features(
            acoustic_features, geometry_features, mic_pairs
        )
        pair_embedding = self.pair_encoder(directed_acoustic, directed_geometry)
        node_embedding = self.edge_to_node(pair_embedding, directed_mic_pairs, num_mic)
        set_embedding = self.set_encoder(node_embedding)

        return set_embedding
