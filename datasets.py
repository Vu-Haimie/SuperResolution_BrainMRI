import random
import h5py
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, h5_file, patch_size=None, scale=4):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file
        self.scale = scale
        self.use_crop = patch_size not in (None, 0)

        if self.use_crop:
            self.patch_size = patch_size // scale

    def random_crop(self, lr, hr):
        lr_c, lr_h, lr_w = lr.shape
        hr_c, hr_h, hr_w = hr.shape

        # patch size kiểm tra kỹ
        if lr_h < self.patch_size or lr_w < self.patch_size:
            raise ValueError(f"LR quá nhỏ: {lr.shape} so với patch_size {self.patch_size}")

        # max toạ độ có thể crop
        max_x = lr_w - self.patch_size
        max_y = lr_h - self.patch_size

        # nếu max_x hoặc max_y = 0 → randint(0,0) vẫn hợp lệ
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        # Crop LR
        lr_crop = lr[:, y:y + self.patch_size, x:x + self.patch_size]

        # Vị trí tương ứng trên HR
        x_hr = x * self.scale
        y_hr = y * self.scale
        hr_crop = hr[:, y_hr:y_hr + self.patch_size * self.scale,
                        x_hr:x_hr + self.patch_size * self.scale]

        # Kiểm tra cuối
        if lr_crop.shape[1:] != (self.patch_size, self.patch_size):
            raise ValueError(f"LR crop sai shape: {lr_crop.shape}")
        if hr_crop.shape[1:] != (self.patch_size * self.scale, self.patch_size * self.scale):
            raise ValueError(f"HR crop sai shape: {hr_crop.shape}")

        return lr_crop, hr_crop

    def __getitem__(self, idx):
        max_attempts = 10

        for attempt in range(max_attempts):
            with h5py.File(self.h5_file, 'r') as f:
                lr = f['lr'][str(idx)][:]
                hr = f['hr'][str(idx)][:]

            lr = lr.astype(np.float32) / 255.0
            hr = hr.astype(np.float32) / 255.0

            if lr.ndim == 2:
                lr = lr[np.newaxis, :, :]
                hr = hr[np.newaxis, :, :]

            try:
                if self.use_crop:
                    lr_h, lr_w = lr.shape[-2:]
                    hr_h, hr_w = hr.shape[-2:]
                    required_hr_h = self.patch_size * self.scale
                    required_hr_w = self.patch_size * self.scale

                    if lr_h < self.patch_size or lr_w < self.patch_size:
                        raise ValueError(f"LR quá nhỏ để crop: {lr.shape}")
                    if hr_h < required_hr_h or hr_w < required_hr_w:
                        raise ValueError(f"HR quá nhỏ để crop: {hr.shape}")

                    lr, hr = self.random_crop(lr, hr)

                else:
                    if any(dim == 0 for dim in lr.shape) or any(dim == 0 for dim in hr.shape):
                        raise ValueError(f"Empty tensor: {lr.shape}, {hr.shape}")

                return lr, hr

            except Exception as e:
                print(f"[Dataset] Warning: lỗi khi xử lý index {idx}: {e}")
                idx = (idx + 1) % self.__len__()
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Too many failed attempts at idx {idx}: {e}")

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr = f['lr'][str(idx)][:]
            hr = f['hr'][str(idx)][:]

        lr = lr.astype(np.float32) / 255.0
        hr = hr.astype(np.float32) / 255.0

        if lr.ndim == 2:
            lr = lr[np.newaxis, :, :]
            hr = hr[np.newaxis, :, :]

        return lr, hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
