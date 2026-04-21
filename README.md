# Sign-Language Translation / Recognition (Real-time Webcam)

# [Tên mô hình / Dự án: Multi-Stream ... cho Nhận dạng & Dịch Ngôn ngữ Ký hiệu](LINK_PAPER_ARXIV_HOẶC_PDF)

<a href="https://pytorch.org/get-started/locally/">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white">
</a>

## Mục tiêu dự án
Dự án này hướng tới:
- **Nhận diện ngôn ngữ ký hiệu (SLR) theo thời gian thực** bằng **webcam** (real-time inference).
- **Dịch ngôn ngữ ký hiệu (SLT)** từ chuỗi động tác sang câu văn bản.

Luồng tổng quan:
1.Webcam capture → 2.Trích xuất keypoints (body/hand/face) → 3.Mô hình SLR/SLT → 4.Hiển thị kết quả real-time.

## Giới thiệu
Dự án triển khai mô hình để **Nhận dạng (SLR)** và **Dịch (SLT)** ngôn ngữ ký hiệu.
Ý tưởng chính là biểu diễn chuỗi keypoints (ví dụ: body/hand/face) và dùng cơ chế attention đa luồng (multi-stream) để học tương tác giữa các luồng, giúp mô hình tận dụng tốt thông tin động học và ngữ cảnh.

## Hiệu năng

### **SLR (Sign Language Recognition)**
| Dataset | WER | Model | Training |
| :---: | :---: | :---: | :---: |
| Phoenix-2014T | 20.5 | [ckpt](https://drive.google.com/file/d/19gg5RI4U7ApemujqDiqgMBJ9a9bUsqyY/view?usp=sharing) | [config](configs/phoenix-2014t_s2g.yaml) |

### **SLT (Sign Language Translation)**
| Dataset | R | B1 | B2 | B3 | B4 | Model | Training |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Phoenix-2014T | 23.08 | 24.59 | 12.72 | 8.44 | 6.27 | [ckpt](https://drive.google.com/file/d/1ftaB4yZ3Wbw1i1TiVChCjoSXCRCzyrBo/view?usp=sharing) | [config](configs/phoenix-2014t_s2t.yaml) |

> Ghi chú:
> - WER: Word Error Rate (càng thấp càng tốt)
> - R/B1..B4: ROUGE/BLEU-1..4 (càng cao càng tốt)

## Cài đặt

### Yêu cầu
- Python: 3.10+
- PyTorch: cài theo CUDA tương ứng

### Thiết lập môi trường
```bash
conda create -n slt python==3.10.13
conda activate slt
# Cài PyTorch theo CUDA máy bạn: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

## Tải dữ liệu & mô hình

### Pretrained Models 
- `mbart_de`: mô hình ngôn ngữ khởi tạo cho nhánh dịch (translation).  

- Checkpoints đã huấn luyện sẵn sử dụng dataset Phoenix_2014T:
  - Model cho nhiệm vụ SLT
  - Model cho nhiệm vụ SLR

## Tải thêm các file/bộ module cần thiết (bắt buộc)

Cần tải thêm một số thư mục (models/modules) từ Google Drive và **đặt trực tiếp vào thư mục gốc của project** (cùng cấp với `train.py`, `configs/`, `requirements.txt`).

### 1) HR_Net (model)
- Link: [HR_Net](https://drive.google.com/drive/folders/1zLcqdmtl5SU9T-bz1UnT7bRgVNzN3mdz?usp=drive_link)  
- Nội dung: đường dẫn model **HR_Net**
- Cách làm:
  1. Tải toàn bộ thư mục/tệp trong link trên
  2. Giải nén (nếu có)
  3. **Copy vào thư mục gốc** của project

### 2) mmcv_src (module)
- Link: [mmcv_src](https://drive.google.com/drive/folders/1ANX7JYdDdtMZ6UPG9_Un5vSEQiy-wRNU?usp=drive_link)  
- Nội dung: module **mmcv_src**
- Cách làm:
  1. Tải thư mục `mmcv_src`
  2. **Đưa `mmcv_src` vào thư mục gốc** của project

### 3) mmpose (module)
- Link: [mmpose](https://drive.google.com/drive/folders/1YknXstdug7xwvQMRwB5fA1shPuBpZL-o?usp=drive_link)  
- Nội dung: module **mmpose**
- Cách làm:
  1. Tải thư mục `mmpose`
  2. **Đưa `mmpose` vào thư mục gốc** của project

### 4) Pretrained models: SLT, SLR và mBART
- Link: [Pretrained_models](https://drive.google.com/drive/folders/1D6gYKMBP2bsOErpg2F7tFbBRPVSf8Fz5?usp=drive_link) 
- Nội dung: pretrained **SLT**, **SLR** và **m_Bart**
- Cách làm:
  1. Tải toàn bộ nội dung trong thư mục Drive
  2. Giải nén (nếu có)
  3. **Đưa các thư mục model (SLT/SLR/m_Bart) vào thư mục gốc** của project

### Cấu trúc thư mục mong muốn
Sau khi tải xong, cấu trúc thư mục gốc nên tương tự:

```text
Sign-Language_Translation/
├─ configs/
├─ data/
├─ images/
├─ mmcv_src/          # tải thêm
├─ mmpose/            # tải thêm
├─ HR_Net/            # hoặc thư mục/tệp HRNet tương ứng (tải thêm)
├─ pretrained_models/ # hoặc SLT/SLR/m_Bart theo gói bạn tải (tải thêm)
├─ requirements.txt
├─ train.py
└─ README.md
```

## Chạy real-time với webcam
Phần này dành cho mục tiêu **nhận diện real-time**. :

Dùng script webcam có sẵn trong repo. Tùy vào nhiệm vụ mong muốn:
Chạy file runModelVideoInput.py cho nhiệm vụ SLR:
```bash
python runModelVideoInput.py 
```
Chạy file runModelVideoInputSLT.py cho nhiệm vụ SLT:
```bash
python runModelVideoInputSLT.py 
```
## Huấn luyện & Đánh giá

### SLR Training
```bash
python train.py --config configs/Phoenix2014T_s2g.yaml --epoch 100
```

### SLR Evaluation
```bash
python train.py --config configs/Phoenix2014T_s2g.yaml \
  --resume pretrained_models/Phoenix2014T_SLR/best.pth --eval
```

### SLT Training
```bash
python train.py --config configs/Phoenix2014T_s2t.yaml --epoch 40
```

### SLT Evaluation
```bash
python train.py --config configs/Phoenix2014T_s2t.yaml \
  --resume pretrained_models/Phoenix2014T_SLT/best.pth --eval
```

## Cấu trúc thư mục (tham khảo)
```text
.
├─ configs/
├─ data/
├─ images/
├─ pretrained_models/
├─ requirements.txt
├─ train.py
└─ README.md
```
