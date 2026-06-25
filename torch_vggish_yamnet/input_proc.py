import numpy as np
import torch
import torchaudio.transforms as ta_trans

from .params import CommonParams, YAMNetParams


class WaveformToInput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        audio_sample_rate = CommonParams.TARGET_SAMPLE_RATE
        window_length_samples = int(round(audio_sample_rate * CommonParams.STFT_WINDOW_LENGTH_SECONDS))
        hop_length_samples = int(round(audio_sample_rate * CommonParams.STFT_HOP_LENGTH_SECONDS))
        fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
        assert window_length_samples == 400
        assert hop_length_samples == 160
        assert fft_length == 512
        self.mel_trans_ope = VGGishLogMelSpectrogram(CommonParams.TARGET_SAMPLE_RATE,
                                                     n_fft=fft_length,
                                                     win_length=window_length_samples,
                                                     hop_length=hop_length_samples,
                                                     f_min=CommonParams.MEL_MIN_HZ,
                                                     f_max=CommonParams.MEL_MAX_HZ,
                                                     n_mels=CommonParams.NUM_MEL_BANDS)

        # note that the STFT filtering logic is exactly the same as that of a
        # conv kernel. It is the center of the kernel, not the left edge of the
        # kernel that is aligned at the start of the signal.

    def __call__(self, waveform, sample_rate):
        '''
        Args:
            waveform: torch tsr [num_audio_channels, num_time_steps]
            sample_rate: per second sample rate
        
        Returns:
            batched torch tsr of shape [N, C, T]
        '''
        x, _ = self.waveform_to_log_mel(waveform, sample_rate)

        return x

    def waveform_to_log_mel(self, waveform, sample_rate, patch_hop_seconds = YAMNetParams.PATCH_HOP_SECONDS):
        '''
        Args:
            waveform: torch tsr [num_audio_channels, num_time_steps]
            sample_rate: per second sample rate
        Returns:
            x: batched torch tsr of shape [N, 1, F, M], where F and M are typically 96 and 64 respectively
            spectrogram: torch tsr of shape [M, L]
        '''
        x = waveform.mean(axis=0, keepdims=True)  # average over channels
        resampler = ta_trans.Resample(sample_rate, CommonParams.TARGET_SAMPLE_RATE)
        x = resampler(x)
        x = self.mel_trans_ope(x)
        x = x.squeeze(dim=0).T  # # [1, C, T] -> [T, C]
        spectrogram = x.cpu().numpy().copy()

        window_size_in_frames = int(round(CommonParams.PATCH_WINDOW_IN_SECONDS / CommonParams.STFT_HOP_LENGTH_SECONDS))
        patch_hop_in_frames = int(round(patch_hop_seconds / CommonParams.STFT_HOP_LENGTH_SECONDS))

        x = x.unfold(0, window_size_in_frames, patch_hop_in_frames).transpose(1, 2).unsqueeze(1)

        return x, spectrogram


class BatchedWaveformToInput(WaveformToInput):
    def __init__(self, sample_rate = CommonParams.TARGET_SAMPLE_RATE, patch_hop_seconds = YAMNetParams.PATCH_HOP_SECONDS):
        super().__init__()
        self.resampler = ta_trans.Resample(sample_rate, CommonParams.TARGET_SAMPLE_RATE)
        self.window_size_in_frames = int(round(CommonParams.PATCH_WINDOW_IN_SECONDS / CommonParams.STFT_HOP_LENGTH_SECONDS))
        self.patch_hop_in_frames = int(round(patch_hop_seconds / CommonParams.STFT_HOP_LENGTH_SECONDS))

    def waveform_to_log_mel(self, waveform):
        '''
        Args:
            waveform: torch tsr [Batch, num_audio_channels, num_time_steps]
        Returns:
            x: batched torch tsr of shape [B, N, 1, F, M], where F and M are typically 96 and 64 respectively
        '''
        x = waveform.mean(dim=1, keepdims=True)  # average over channels
        x = self.resampler(x)
        x = self.mel_trans_ope(x) # shape = [B, 1, M, T]

        x = x.unfold(-1, self.window_size_in_frames, self.patch_hop_in_frames) # shape = [B, 1, M, N, F]
        x = x.permute(0,3,1,4,2)

        return x


######################################################################################################################

class VGGishLogMelSpectrogram(ta_trans.MelSpectrogram):
    '''
    This is a _log_ mel-spectrogram transform that adheres to the transform
    used by Google's vggish model input processing pipeline
    '''
    def forward(self, waveform):
        """
        Args:
            waveform (torch.Tensor): Tensor of audio of dimension (..., time)

        Returns:
            torch.Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time)
        """
        specgram = self.spectrogram(waveform)
        # NOTE at mel_features.py:98, googlers used np.abs on fft output and
        # as a result, the output is just the norm of spectrogram raised to power 1
        # For torchaudio.MelSpectrogram, however, the default
        # power for its spectrogram is 2.0. Hence we need to sqrt it.
        # I can change the power arg at the constructor level, but I don't
        # want to make the code too dirty
        specgram = specgram ** 0.5

        mel_specgram = self.mel_scale(specgram)
        mel_specgram = torch.log(mel_specgram + CommonParams.LOG_OFFSET)

        return mel_specgram
