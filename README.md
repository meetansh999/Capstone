# 🌊 Oil Spill Detection using Deep Learning

This project implements a deep learning pipeline to detect and segment oil spills in ocean imagery using Sentinel-1 SAR data. The final model utilizes a UNet architecture with a MobileNetV3 backbone to achieve high accuracy and efficiency in marine pollution monitoring.

---

## 📦 Repository Contents

| File/Folder | Description |
|-------------|-------------|
| `MobileNetv3.py` | Model architecture: UNet + MobileNetV3 encoder |
| `training.py` | Final training script used for best-performing model |
| `testing.py` | Evaluation pipeline for inference and metrics |
| `endpoint.py` | Lightweight script to generate prediction for a single image |
| `model.pth` | Trained PyTorch model weights (final version) |
| `predictions_vs_ground_truth/` | Sample output comparisons of model predictions |
| `README.md` | Project overview and usage instructions |

---

## 🧠 Model Highlights

- **Architecture**: UNet + MobileNetV3 (lightweight & fast)
- **Accuracy**: Achieved ~95% pixel accuracy and 0.74 Mean IoU
- **Data**: Sentinel-1 SAR imagery
- **Framework**: PyTorch

---

## 🚀 Quick Start

### 1. Install Requirements

```bash
pip install -r requirements.txt
### 2. Run Inference

```bash
python endpoint.py --image path/to/image.png
```


## 📊 Example Results

You can find visual examples of model predictions in the `predictions_vs_ground_truth/` folder.

---

## 📌 Notes

- Model is optimized for 5 class segmentation (Oil Spills, Land, Look-alikes, Boats/ Ships, Sea Surface).
- This implementation is designed to run efficiently on Google Colab or mid-range GPUs.
- For large-scale deployment, consider integrating with Google Earth Engine or cloud APIs.

---

## 🧑‍💻 Authors

**Meetansh Kharbanda**, **Gage Scott Anthony**, **Nolan Ayotte**, **Noah Serhan**
