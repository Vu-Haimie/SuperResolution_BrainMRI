import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import matplotlib.pyplot as plt  # ✅ thêm cho hiển thị ảnh

from models import SRDenseNet
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    # Lấy tên ảnh để tạo file đầu ra
    basename = os.path.basename(args.image_file)
    name, ext = os.path.splitext(basename)

    # Tạo thư mục outputs (nếu cần)
    os.makedirs("outputs", exist_ok=True)
    out_bicubic = os.path.join("outputs", f"{name}_bicubic_x{args.scale}{ext}")
    out_sr = os.path.join("outputs", f"{name}_srdensenet_x{args.scale}{ext}")

    # Setup model
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SRDenseNet().to(device)

    # Load weights
    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=device).items():
        if n in state_dict:
            state_dict[n].copy_(p)
        else:
            raise KeyError(f"Unexpected key in state_dict: {n}")
    model.eval()

    # Load ảnh và resize
    image = pil_image.open(args.image_file).convert('RGB')
    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
    out_lr = os.path.join("outputs", f"{name}_lr{ext}")
    lr.save(out_lr)
    bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
    bicubic.save(out_bicubic)

    # Chuẩn hóa
    lr_tensor, _ = preprocess(lr, device)
    hr_tensor, _ = preprocess(hr, device)
    _, ycbcr = preprocess(bicubic, device)

    with torch.no_grad():
        preds = model(lr_tensor).clamp(0.0, 1.0)

    psnr = calc_psnr(hr_tensor, preds)
    print('✅ PSNR: {:.2f} dB'.format(psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(out_sr)

    # ===== ✅ Hiển thị ảnh kết quả =====
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(hr); axs[0].set_title('Ảnh gốc'); axs[0].axis('off')
    axs[1].imshow(lr.resize(hr.size, resample=pil_image.NEAREST)); axs[1].set_title('Ảnh LR'); axs[1].axis('off')
    axs[2].imshow(bicubic); axs[2].set_title('Bicubic Upscale'); axs[2].axis('off')
    axs[3].imshow(output); axs[3].set_title('SRDenseNet Output'); axs[3].axis('off')
    plt.tight_layout()
    plt.show()
