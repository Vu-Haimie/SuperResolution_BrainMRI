import h5py
import torch
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from models import SRDenseNet  # import đúng theo repo của bạn
from datasets import EvalDataset  # hoặc viết tay nếu cần
from tqdm import tqdm

# ==== Cấu hình ====
eval_file = 'eval_SR_x4_cleaned.h5'
weights_file = 'best.pth'
scale = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== Load mô hình đã train ====
model = SRDenseNet(growth_rate=16, num_blocks=8, num_layers=8).to(device)
model.load_state_dict(torch.load(weights_file, map_location=device))
model.eval()

# ==== Load tập đánh giá ====
eval_dataset = EvalDataset(eval_file)
eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

# ==== Hàm tính SSIM (cho ảnh 3 kênh) ====
def compute_ssim(pred, gt):
    pred = pred.squeeze(0).cpu()
    gt = gt.squeeze(0).cpu()

    # Nếu ảnh grayscale: [1, H, W]
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred_np = pred.squeeze(0).numpy()  # [H, W]
        gt_np = gt.squeeze(0).numpy()
        return structural_similarity(pred_np, gt_np, data_range=1.0)
    
    # Nếu ảnh RGB: [3, H, W]
    elif pred.ndim == 3 and pred.shape[0] == 3:
        pred_np = pred.permute(1, 2, 0).numpy()  # [H, W, C]
        gt_np = gt.permute(1, 2, 0).numpy()
        return structural_similarity(pred_np, gt_np, multichannel=True, data_range=1.0)
    
    else:
        raise ValueError(f"Không hỗ trợ shape: {pred.shape}")

# ==== Đánh giá ====
psnr_list = []
ssim_list = []

with torch.no_grad():
    for inputs, labels in tqdm(eval_dataloader, desc="Evaluating"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        preds = model(inputs).clamp(0.0, 1.0)

        # Cắt viền scale pixel như trong lúc train
        preds = preds[:, :, scale:-scale, scale:-scale]
        labels = labels[:, :, scale:-scale, scale:-scale]

        psnr = peak_signal_noise_ratio(labels.cpu().numpy(), preds.cpu().numpy(), data_range=1.0)
        ssim = compute_ssim(preds, labels)

        psnr_list.append(psnr)
        ssim_list.append(ssim)

# ==== In kết quả ====
print(f"\n✅ Đánh giá trên {len(psnr_list)} ảnh:")
print(f"📈 PSNR trung bình: {np.mean(psnr_list):.2f} dB")
print(f"📊 SSIM trung bình: {np.mean(ssim_list):.4f}")

print(psnr_list)
print(ssim_list)

import matplotlib.pyplot as plt

# ==== Vẽ biểu đồ ====
x = np.arange(len(psnr_list))

plt.figure(figsize=(10, 5))
plt.plot(x, psnr_list, marker='o', label='PSNR (dB)')
plt.title('PSNR trên từng ảnh')
plt.xlabel('Ảnh')
plt.ylabel('PSNR (dB)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(x, ssim_list, marker='s', color='green', label='SSIM')
plt.title('SSIM trên từng ảnh')
plt.xlabel('Ảnh')
plt.ylabel('SSIM')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()