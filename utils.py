import torch

def directed_features(
    acoustic_features: torch.Tensor,
    geometry_features: torch.Tensor,
    mic_pairs: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """
    i<j 순서로 되어있는 input feature를 (i,j)/(j,i) 양쪽으로 확장 - 방향 구분이 되는 입력

    반대 방향으로 바꿀 때 성분별 변환:
        acoustic:  [log|X_i|, log|X_j|, coh_ij, Re(C_ij), Im(C_ij)]
            -> [log|X_j|, log|X_i|, coh_ij, Re(C_ij), -Im(C_ij)]   (C_ji = conj(C_ij))
        geometry:  [d_ij(3), bar_d_ij(3), l_ij, tau_ij_max]
                -> [-d_ij(3), -bar_d_ij(3), l_ij, tau_ij_max]          (d_ji = -d_ij)

    Args:
        acoustic_features (torch.Tensor): (B, P, 5, K, T) acoustic pair feature (i<j 순서)
        geometry_features (torch.Tensor): (B, P, 8) geometry pair feature (i<j 순서)
        mic_pairs (list[tuple[int, int]]): pair 축과 동일한 순서의 (i, j) 인덱스 목록, i<j

    Returns:
        directed_acoustic_features (torch.Tensor): (B, 2P, 5, K, T)
        directed_geometry_features (torch.Tensor): (B, 2P, 8)
        directed_mic_pairs (list[tuple[int, int]]): 앞 P개는 (i,j), 뒤 P개는 (j,i)
    """
    reverse_acoustic = acoustic_features[
        :, :, [1, 0, 2, 3, 4], :, :
    ].clone()  # 자리 교체
    reverse_acoustic[:, :, 4, :, :] *= -1  # Im(C_ij) -> -Im(C_ij)

    reverse_geometry = geometry_features.clone()
    reverse_geometry[:, :, :6] *= -1

    directed_acoustic_features = torch.cat([acoustic_features, reverse_acoustic], dim=1)
    directed_geometry_features = torch.cat([geometry_features, reverse_geometry], dim=1)
    directed_mic_pairs = mic_pairs + [(j, i) for i, j in mic_pairs]

    return directed_acoustic_features, directed_geometry_features, directed_mic_pairs
