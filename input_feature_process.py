import torch
import torch.nn.functional as F
from STFT import ConvSTFT


class Input_feature_process(torch.nn.Module):
    def __init__(self, win_len, ref_channel=0):
        super().__init__()
        self.STFT = ConvSTFT(win_len = win_len,
                             win_inc = 128,
                             fft_len=None,
                             vad_threshold = 2/3,
                             win_type = "hann")

        self.ref = ref_channel
        self.eps = 1e-8

    def cross_correlation(self, x_real, x_imag, PHAT=True):
        """
        모든 마이크 쌍의 cross-spectrum(GCC/GCC-PHAT)을 계산

        Args:
            x_real (Tensor): STFT 적용 결과의 실수부 - (B, C, K=257, T)
            x_imag (Tensor): STFT 적용 결과의 허수부 - (B, C, K=257, T)
            PHAT (bool): True면 PHAT 정규화(크기를 1로 만들고 위상만 남김) 적용
                기본값 True

        Returns:
            Tuple[Tensor, Tensor, list[tuple[int, int]]]:
                - cross-spectrum 실수부: (B, P, K, T), P = C(C-1)/2
                - cross-spectrum 허수부: (B, P, K, T)
                - 각 pair 축에 대응하는 마이크 인덱스 목록
        """

        # 하나의 복소 텐서로 결합
        comp = torch.complex(x_real, x_imag)

        B, C, Freq, T = comp.shape
        pair_i = []
        pair_j = []
        for i in range(C-1):
            for j in range(i+1, C):
                pair_i.append(i)
                pair_j.append(j)

        X_i = comp[:, pair_i]
        X_j = comp[:, pair_j]

        G = X_i * X_j.conj()
        if PHAT:  # 복소 스펙트럼의 크기를 1로 정규화하고 위상만 남김
            G = G / (G.abs() + self.eps)

        mic_pairs = list(zip(pair_i, pair_j))
        return X_i, X_j, G.real, G.imag, mic_pairs

    @staticmethod
    def _temporal_mean(x, kernel_size):
        """
        입력의 마지막 시간축에 centerend moving average
        한 개의 TF bin만 사용하면 관련 없는 두 신호도 항상 coherence가 1이 되기 떄문에,
        여러 시간 프레임을 평균하여 계산
        
        Args:
            x (torch.Tensor): 시간 평균을 적용할 실수 Tensor, (B, P, K, T)
            kernel_size(int): 이동평균에 사용할 시간 프레임 개수

        Return:
            torch.Tensor: 시간축 이동평균이 적용된 Tensor, (B, P, K, T)
        """
        B, P, K, T = x.shape
        x = x.reshape(-1, 1, T)
        x = F.avg_pool1d(
            x, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size//2, 
            count_include_pad=False
            )

        return x.reshape(B, P, K, T)

    def extract_pair_features(self, x_real, x_imag, coherence_frames=9):
        """
        complex STFT로부터 모든 마이크 쌍의 acoustic feature 계산
        a_ij(f, t) = [
            log(|X_i(f, t)| + eps),
            log(|X_j(f, t)| + eps),
            coherence_ij(f, t),
            Re(C_ij(f, t)),  - C_ij: 정규화된 cross-spectrum
            Im(C_ij(f, t))
            ]

        Args:
            x_real (torch.Tensor): 복소 STFT의 실수부, (B, C, F, T)
            x_imag (torch.Tensor): 복소 STFT의 허수부, (B, C, F, T)
            coherence_frames (int, optional): coherence의 local spectral estimate에 사용할 시간 프레임 개수

        Returns:
            tuple[Tensor, list[tuple[int, int]]]
                - pair_features(a_ij): 모든 마이크 쌍의 5채널 acoustic feature, (B, P, 5, F, T)
                - mic_pairs (list[tuple[int, int]]): pair 축의 각 위치에 대응하는 마이크 인덱스 목록
        """

        X_i, X_j, cross_real, cross_imag, mic_pairs = self.cross_correlation(x_real, x_imag, PHAT=False)
        cross_spec = torch.complex(cross_real, cross_imag)

        # 마이크별 log-magnitude
        log_mag_i = torch.log(X_i.abs() + self.eps)
        log_mag_j = torch.log(X_j.abs() + self.eps)

        # Local cross-power estimate: 실수부/허수부 따로 평균한 뒤 결합
        cross_power = torch.complex(self._temporal_mean(cross_spec.real, coherence_frames),
                                    self._temporal_mean(cross_spec.imag, coherence_frames))

        # Local auto-power estimates
        auto_power_i = self._temporal_mean(X_i.abs().square(), coherence_frames)
        auto_power_j = self._temporal_mean(X_j.abs().square(), coherence_frames)

        # magnitude-squared coherence: |S_ij|^2 / S_ii * S_jj
        # 위상 관계가 안정적일수록 1, 서로 관련 없는 잡음일수록 0에 가까워짐
        coherence = (cross_power.abs().square() / (auto_power_i * auto_power_j + self.eps)).clamp(0.0, 1.0)

        # GCC-PHAT real/imag
        phase_spec = cross_spec / (cross_spec.abs() + self.eps)

        pair_features = torch.stack(
            [
                log_mag_i,        # 마이크 i에서 해당 TF bin의 신호 세기
                log_mag_j,        # 마이크 j에서 해당 TF bin의 신호 세기
                coherence,        # 두 마이크 신호의 시간적 일관성과 신뢰도
                phase_spec.real,  # 마이크 간 위상차(IPD)의 cos 성분: IPD의 경계(±pi) 불연속 회피
                phase_spec.imag   # 마이크 간 위상차(IPD)의 sin 성분 
            ],
            dim=2
        )

        return pair_features, mic_pairs

    def forward(self, inputs, coherence_frames=9):
        """
        waveform input을 STFT 변환 후, pair feature 계산
        Args:
            inputs: (B, C, L) 형태의 다채널 waveform

        Returns:
            pair_features: (B, P, 5, F, T)
            mic_pairs: 마이크 pair 인덱스 목록
        """
        x_real, x_imag = self.STFT(inputs, cplx=True)
        return self.extract_pair_features(x_real, x_imag, coherence_frames=coherence_frames)
