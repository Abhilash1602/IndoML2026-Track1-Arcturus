from src.config import select_device


class Config:
    SR = 16000
    N_FFT = 1024
    HOP_LENGTH = 160
    WIN_LENGTH = 400
    N_MELS = 80
    MAX_DURATION = 10.0
    BATCH_SIZE = 16
    EPOCHS = 10
    LR = 1e-3
    DEVICE = select_device()
