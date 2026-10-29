import torch
import torch.nn as nn
import torchaudio


class ModularSED(nn.Module):
    def __init__(self, config):
        super().__init__()
        # On-the-fly GPU audio feature extraction
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SR,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
        )

        # Feature Encoder (CNN) - Pools Frequency, preserves Time
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )

        # Temporal Context Mechanism
        freq_bins_after_pool = config.N_MELS // 4
        self.temporal = nn.GRU(
            input_size=32 * freq_bins_after_pool,
            hidden_size=128,
            batch_first=True,
            bidirectional=True,
        )

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Outputs raw frame-level logits
        )

    def forward(self, x):
        # 1. Feature Extraction
        mel = self.mel_transform(x)               # [B, F, T]
        mel = torch.log(mel + 1e-6).unsqueeze(1)   # [B, 1, F, T]

        # 2. Spatial Encoding
        x_enc = self.encoder(mel)                  # [B, C, F_pool, T]
        batch, channels, freq_pool, frames = x_enc.shape

        # 3. Temporal Encoding
        x_enc = x_enc.permute(0, 3, 1, 2).contiguous()  # [B, T, C, F_pool]
        x_enc = x_enc.view(batch, frames, -1)            # [B, T, Features]
        x_temp, _ = self.temporal(x_enc)                 # [B, T, 256]

        # 4. Classification
        logits = self.classifier(x_temp).squeeze(-1)     # [B, T]
        return logits
