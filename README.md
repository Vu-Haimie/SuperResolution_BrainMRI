# 🧠 Tăng cường độ phân giải MRI não bằng Deep Learning

Dự án này sử dụng mô hình học sâu (SRDenseNet) để **phục hồi ảnh MRI não có độ phân giải thấp**, giúp nâng cao chất lượng ảnh y tế phục vụ phân tích và chẩn đoán.

---

## 📸 Kết quả minh họa

| Ảnh gốc (Low-Res) | Nội suy Bicubic | Kết quả từ SRDenseNet |
|-------------------|------------------|------------------------|
| ![Low](brain_img\3.jpg) | ![Bicubic](brain_img\3_bicubic_x4.jpg) | ![SR](brain_img\3_srdensenet_x4.jpg) |

---

## 🚀 Tính năng nổi bật

- 🔍 Tăng cường độ phân giải ảnh y tế (MRI, grayscale)
- 📈 Đánh giá chất lượng ảnh bằng PSNR
- 💻 Giao diện dòng lệnh đơn giản, dễ dùng
- ⚡ Tối ưu cho cả CPU và GPU với PyTorch

---

## 📁 Cấu trúc thư mục

\`\`\`
SuperResolution_BrainMRI/
├── brain_img/             # Thư mục chứa ảnh đầu vào
├── model/                 
│   └── SR_MRI.pth         # Trọng số mô hình đã huấn luyện
├── models.py              # Định nghĩa mô hình SRDenseNet
├── prepare.py             # Chuẩn bị ảnh huấn luyện 
├── test.py                # Tập lệnh chính để chạy mô hình
├── train.py               # Huấn luyện mô hình SRDenseNet
├── utils.py               # Các hàm tiện ích (tiền xử lý, PSNR, ...)
└── README.md
\`\`\`

---

## 🛠 Cài đặt

\`\`\`bash
git clone https://github.com/Vu_Haimie/SuperResolution_BrainMRI.git
cd SuperResolution_BrainMRI
\`\`\`

> 📌 Đảm bảo bạn đã cài Python và pip trước đó.

---

## ▶️ Cách sử dụng

\`\`\`bash
python test.py \
  --weights-file model/SR_MRI.pth \
  --image-file brain_img/1.jpg \
  --scale 4
\`\`\`

> Lệnh trên sẽ tạo ra các ảnh mới như `1_bicubic_x4.jpg`, `1_srdensenet_x4.jpg` và in ra chỉ số PSNR.

---

## 📊 Chỉ số đánh giá

- **PSNR (Peak Signal-to-Noise Ratio)**: Được dùng để đo độ tương đồng giữa ảnh gốc và ảnh phục hồi.
- Hàm được cài trong `utils.py`.

---