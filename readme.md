# SHAS Pipeline for Speech Segmentation and Translation (fr-en)

This project replicates the **SHAS (Supervised Hybrid Audio Segmentation)** pipeline to train a speech segmentation model and evaluate **speech-to-text translation quality** for the **French-English (fr-en)** pair using the **mTEDx** dataset.

---

## Mục tiêu / Objectives

- Huấn luyện mô hình phân đoạn lời nói đầu vào dựa trên `wav2vec2`.
- Dịch lời nói sang văn bản (Speech Translation) sử dụng mô hình `joint-s2t-multilingual`.
- Đánh giá chất lượng phân đoạn bằng chỉ số **BLEU score** (MWER-based evaluation).

---

## 1. Thiết lập môi trường trên Kaggle

> Sử dụng Micromamba để tạo môi trường Python 3.9 và cài các thư viện cần thiết:

```bash
# Tạo môi trường micromamba
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba create -y -p /kaggle/working/env -c conda-forge python=3.9.6

# Cài thư viện
./bin/micromamba run -p /kaggle/working/env pip install torch==1.10.0 torchaudio==0.10.0 ...
```

---

## 2. Clone SHAS và fairseq

```bash
git clone https://github.com/mt-upc/SHAS.git
git clone --branch shas https://github.com/mt-upc/fairseq.git /kaggle/working/models/fairseq_root
```

Cài `fairseq` ở chế độ editable:

```bash
cd /kaggle/working/models/fairseq_root
/kaggle/working/bin/micromamba run -p /kaggle/working/env pip install --editable ./
```

---

## 3. Tải mô hình dịch đã huấn luyện (joint-s2t-multilingual)

```bash
cd /kaggle/working/models
mkdir joint-s2t-multilingual
# Tải các file model
wget https://dl.fbaipublicfiles.com/.../checkpoint17.pt -O joint-s2t-multilingual/checkpoint17.pt
...
```

---

## 4. Cập nhật đường dẫn và sửa cấu hình

```bash
# Sửa đường dẫn đến spm.model
sed -i "s+/path/spm.model+/kaggle/working/models/joint-s2t-multilingual/spm.model+" .../config.yaml

# Cập nhật file checkpoint
python SHAS/src/data_prep/fix_joint_s2t_cfg.py -c joint-s2t-multilingual/checkpoint17.pt
```

---

## 5. Chuẩn hóa dữ liệu âm thanh

- Convert từ `.flac` ➝ `.wav` (mono, 16kHz) **ở Colab hoặc máy cá nhân** vì Kaggle không hỗ trợ ghi đè `input/`.
- Tải lại vào Kaggle Dataset.

---

## 6. Chuẩn bị dữ liệu phân đoạn

```bash
# Thực thi 3 lần với split: train, valid, test
python SHAS/src/data_prep/prepare_dataset_for_segmentation.py \
  -y .../train/txt/train.yaml \
  -w .../train/wav \
  -o segm_datasets/mTEDx/fr-en
```

---

## 7. Huấn luyện mô hình phân đoạn

```bash
python SHAS/src/supervised_hybrid/train.py \
    --datasets segm_datasets/mTEDx/fr-en \
    --results_path results/supervised_hybrid \
    --model_name facebook/wav2vec2-xls-r-300m \
    --experiment_name fr_sfc_model \
    --train_sets train --eval_sets valid \
    --batch_size 4 --learning_rate 2.5e-4 --update_freq 20 --max_epochs 1
```

---

## 8. Phân đoạn lời nói với mô hình đã huấn luyện

```bash
python SHAS/src/supervised_hybrid/segment.py \
  -wavs .../test/wav \
  -ckpt results/supervised_hybrid/fr_sfc_model/ckpts/step-8989.pt \
  -yaml results/fr-en-segm.yaml \
  -max 16
```

---

## 9. Chuẩn bị đầu vào cho đánh giá BLEU

```python
from SHAS.src.eval_scripts.prepare_custom_dataset import prepare_custom_dataset
prepare_custom_dataset(
    path_to_yaml="results/fr-en-segm.yaml",
    path_to_wavs=".../test/wav",
    lang="en", index_start=1
)
```

---

## 10. Đánh giá bằng MWER và BLEU (chạy local)

> Phần này yêu cầu **Python 2.7** và **mwerSegmenter** (không thể thực hiện trên Kaggle).

```bash
# Cài mwerSegmenter
wget https://www-i6.informatik.rwth-aachen.de/.../mwerSegmenter.tar.gz
tar -xzf mwerSegmenter.tar.gz

# Chuyển .xml sang định dạng chuẩn và normalize
python sgm2mref.py test.en.xml | normalizeTextNIST.pl -c > __mreference
normalizeTextNIST.pl -c translations_formatted.txt > __translation

# Chạy phân đoạn dựa trên MWER
./segmentBasedOnMWER.sh test.fr.xml test.en.xml translations_formatted.txt joint-s2t-multilingual en translations_aligned.xml normalize 1

# Chấm điểm
python SHAS/src/eval_scripts/score_translation.py ./
```

- **translations_aligned.xml**: file đã căn chỉnh phân đoạn để tính BLEU.
- Kết quả BLEU sẽ xuất hiện trong log hoặc lưu vào file `score.txt`.

---

## Tài liệu tham khảo

- SHAS Paper: [Approaching Optimal Segmentation for End-to-End Speech Translation (ACL 2022)](https://arxiv.org/abs/2202.04774)
- [mTEDx Dataset](https://www.openslr.org/100)
- [Fairseq](https://github.com/mt-upc/fairseq)

---
