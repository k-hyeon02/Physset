import torch

def center_mic_coordinates(mic_coordinate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    마이크 좌표를 배열 중심(무게중심) 기준으로 이동한다.
        r_c = (1/M) * sum_m r_m
        tilde_r_m = r_m - r_c

    Args:
        mic_coordinate (torch.Tensor): (..., M, 3) 마이크 좌표

    Returns:
        centered (torch.Tensor): (..., M, 3) 중심화된 좌표 (tilde_r_m)
        center (torch.Tensor): (..., 1, 3) 배열 중심 (r_c)
    """
    center = mic_coordinate.mean(dim=-2, keepdim=True)
    centered = mic_coordinate - center
    return centered, center


def pair_geometry_features(
    centered_mic_coordinate: torch.Tensor,
    mic_pairs: list[tuple[int, int]],
    speed_of_sound: float = 343.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    마이크 쌍 (i, j)의 baseline 기하학 특징을 계산한다.
        d_ij = tilde_r_i - tilde_r_j
        l_ij = ||d_ij||_2
        bar_d_ij = d_ij / (l_ij + eps)
        tau_ij_max = l_ij / c

    Args:
        centered_mic_coordinate (torch.Tensor): (..., M, 3) 중심화된 마이크 좌표 (tilde_r_m)
        mic_pairs (list[tuple[int, int]]): pair_features와 동일한 순서의 (i, j) 인덱스 목록
        speed_of_sound (float): 음속 c [m/s]
        eps (float): division-by-zero 방지용 작은 값

    Returns:
        pair_geometry (torch.Tensor): (..., P, 8) = [d_ij(3), bar_d_ij(3), l_ij(1), tau_ij_max(1)]
        array_aperture (torch.Tensor): (...,) D_array = max_{i,j} ||r_i - r_j||_2
    """
    pair_i = [i for i, j in mic_pairs]
    pair_j = [j for i, j in mic_pairs]

    r_i = centered_mic_coordinate[..., pair_i, :]  # (..., P, 3)
    r_j = centered_mic_coordinate[..., pair_j, :]  # (..., P, 3)

    d_ij = r_i - r_j  # (..., P, 3)
    l_ij = torch.linalg.norm(d_ij, dim=-1, keepdim=True)  # (..., P, 1)
    d_ij_unit = d_ij / (l_ij + eps)  # (..., P, 3)
    tau_ij_max = l_ij / speed_of_sound  # (..., P, 1)

    pair_geometry = torch.cat([d_ij, d_ij_unit, l_ij, tau_ij_max], dim=-1)  # (..., P, 8)

    # 배열 aperture: 모든 마이크 쌍 중 최대 거리 (mic_pairs가 완전그래프이므로 그 안의 최대 l_ij와 동일)
    all_diff = centered_mic_coordinate.unsqueeze(-2) - centered_mic_coordinate.unsqueeze(-3)  # (..., M, M, 3)
    all_dist = torch.linalg.norm(all_diff, dim=-1)  # (..., M, M)
    array_aperture = all_dist.amax(dim=(-1, -2))  # (...,)

    return pair_geometry, array_aperture