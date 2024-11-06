import os
import torch
import numpy as np
import logging
import random


def set_logger(path, flag):
    log_file_path = os.path.join(path, 'log-' + flag + '.log')
    logger = logging.Logger('file_logger')

    # save in file
    file_handler = logging.FileHandler(filename=log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # print on screen
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def get_time(start_time, end_time):
    time_g = int(end_time - start_time)
    hour = int(time_g / 3600)
    minu = int(time_g / 60) % 60
    secon = time_g % 60
    return hour, minu, secon


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_model(now_epoch, model, model_dir, optimizer, rng_state, cuda_rng_state, log_file):
    ckpt_path = os.path.join(model_dir, f'epoch-{now_epoch}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'rng_state': rng_state,
        'cuda_rng_state': cuda_rng_state
    }, ckpt_path)
    log_file.info(f'Model saved to {ckpt_path}')


def dcor_custom(x, y):
    x = x[:, None]
    y = y[:, None]
    a = torch.norm(x[:, None] - x, p=2, dim=2)
    b = torch.norm(y[:, None] - y, p=2, dim=2)

    A = a - a.mean(dim=0)[None, :] - a.mean(dim=1)[:, None] + a.mean()
    B = b - b.mean(dim=0)[None, :] - b.mean(dim=1)[:, None] + b.mean()

    n = x.size(0)

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = torch.sqrt(dcov2_xy) / torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

    return dcor


def logloss(preds, trues):
    eps = 1e-15
    preds = preds.clamp(min=eps, max=1-eps)
    loss = torch.sum(-trues * preds.log() - (1-trues) * (1-preds).log())
    return loss / len(preds)


def create_sinusoidal_embeddings(n_pos: int, dim: int, out: torch.Tensor):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
