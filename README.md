# StarEye
Official code repository for "An advanced three stage lightweight model for underwater human detection"

![StarEye Architecture](images/figure6.png)

## Overview
StarEye is a state-of-the-art lightweight deep learning model designed for robust underwater human body detection (UHBD) in complex marine environments. It achieves **95.1% mAP50** with only **3.8MB model size** (16.9% of YOLOv8s), making it ideal for deployment on resource-constrained AUVs. The model maintains high performance under challenging conditions including poor visibility, dynamic lighting, biological interference, and partial occlusions.

## Key Features
- 🚀 **Real-time Performance**: 198 FPS on RTX 4060 GPU
- ⚖️ **Lightweight Design**: 3.8MB model size (22.4x smaller than YOLOv8s)
- 🌊 **Underwater-Optimized**: Handles scattering, refraction, and low-visibility
- 🔍 **Attention Mechanisms**: Context Anchor Attention (CAA) for elongated targets
- 🧠 **Novel Architecture**: StarBlock backbone + SCBN detection head

## Model Architecture
StarEye features a three-stage architecture with key innovations:
1. **StarBlock Backbone**: Replaces standard convolutions with depthwise separable operations and Star Operations (Eq.2) for efficient feature extraction.
2. **CAA Attention**: Uses depthwise strip convolutions (Eq.3-5) to capture long-range context for elongated human bodies.
3. **SCBN Detection Head**: Shared Convolution Batch Normalization reduces parameters while maintaining accuracy.

## Performance Comparison
| Model         | mAP@0.5 | Size (MB) | FPS  | FLOPs (G) |
|---------------|---------|-----------|------|-----------|
| Faster-RCNN   | 79.2%   | 521       | 22   | 348.8     |
| YOLOv8s       | 95.0%   | 22.4      | 51   | 23.2      |
| YOLOv8n       | 92.3%   | 6.2       | 141  | 7.9       |
| MobileNet-CA  | 94.4%   | 13.2      | 71   | 15.7      |
| **StarEye**   | **95.1%**| **3.8**   | **198**| **4.5**  |

![Performance Comparison](images/figure13.png)  
*3D performance comparison (Bubble size = average of GFLOPs & Inference Time)*

## Installation
```bash
git clone https://github.com/Asuka008/StarEye.git
cd StarEye
pip install -r requirements.txt
```
### Usage

#### Training

1. **Prepare Dataset**
   - Organize images in YOLO format (`images/` + `labels/` folders)
   - Update `data/underwater.yaml` with your dataset paths

2. **Start Training**

   ```bash
   python3 train.py 
   ```

#### Inference

Run detection:

```bash
python3 detect.py 
```

# Citation
If you find this work useful, please cite:
```bibtex
@article{liao2025advanced,
  title={An advanced three stage lightweight model for underwater human detection},
  author={Liao, Zichen and Hu, Kai and Meng, Yuancheng and Shen, Shuai},
  journal={Scientific Reports},
  volume={15},
  pages={18137},
  year={2025},
  publisher={Nature Publishing Group UK London},
  doi={10.1038/s41598-025-03677-2}
}
```
