# 🛰️ Thermal Object Detection from UAV — HIT-UAV

**Real-time detection of people and vehicles in high-altitude infrared imagery using YOLOv8.**

Built end-to-end in Google Colab: data acquisition → label validation → transfer learning → quantitative + qualitative evaluation.

![Python](https://img.shields.io/badge/Python-3.13-blue)![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red)![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8n-orange)![mAP50](https://img.shields.io/badge/mAP%4050-0.784-brightgreen)![mAP50-95](https://img.shields.io/badge/mAP%4050--95-0.503-yellow)

---

## 🔥 Why This Project

Visible-light detection fails at night and in low-contrast conditions. Thermal infrared imaging solves this — butintroduces its own challenges: low resolution, tiny objects (a person is ~10 pixels from 100 m altitude),and textural similarity between warm inanimate objects and humans.

This project trains and evaluates a detector purpose-built for that regime, on a realUAV-collected dataset spanning day/night scenes at 60–130 m altitude and 30–90° camera pitch.

## 📊 Results

**YOLOv8n · 50 epochs · 640 px · 3.0 M parameters · 8.2 GFLOPs**

|Metric|Validation|
|---|---|
|Precision|**0.850**|
|Recall|0.709|
|[mAP@0.5](mailto:mAP@0.5)|**0.784**|
|[mAP@0.5](mailto:mAP@0.5):0.95|0.503|
|Inference speed|~3.8 ms / image (T4 GPU)|

### Per-class performance

|Class|Instances|[mAP@0.5](mailto:mAP@0.5):0.95|Analysis|
|---|---|---|---|
|Car|719|**0.728**|Large, high-contrast thermal signature|
|OtherVehicle|12|0.588|⚠️ Too few samples to be reliable|
|Bicycle|554|0.513|Small signature, riders confused with pedestrians|
|Person|1168|0.485|Hardest class — tiny blobs at altitude|
|DontCare|7|0.203|Near-zero support; effectively unlearnable|

### Qualitative results

![Predictions on held-out thermal scenes](https://chat.z.ai/c/assets/demo_grid.png)_Model predictions on test images — day and night scenes, mixed altitude and perspective. Boxes drawn with class + confidence._

## 📁 Dataset — HIT-UAV

[**HIT-UAV**](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset) (Suo et al., 2022) is ahigh-altitude infrared thermal dataset for UAVs:

- **2,898 images** extracted from 43,470 frames (train 2,008 / val 287 / test 603)
- 5 classes: Person, Car, Bicycle, OtherVehicle, DontCare
- Flight altitude: **60–130 m** · camera pitch: **30–90°** · day **and** night
- Scenes: schools, parking lots, roads, playgrounds
- Used in pre-converted YOLO format (class_id x_center y_center width height)

## ⚙️ Pipeline

1. **Data acquisition** — dataset pulled programmatically via `kagglehub` into Colab
2. **Label integrity check** — visualized ground-truth boxes over images _before_ training to catchformat/path/class-order errors early
3. **Config normalization** — rewrote `dataset.yaml` with absolute paths and verified class index mapping
4. **Transfer learning** — YOLOv8n fine-tuned from COCO weights (Bicycle/Car/Person head weights transferred directly)
5. **Evaluation** — mAP on held-out splits, per-class breakdown, confusion matrix analysis
6. **Threshold analysis** — tested conf thresholds 0.15–0.4 to map the precision/recall trade-off for small-object recall

## 🏋️ Training Configuration

|Parameter|Value|
|---|---|
|Model|`yolov8n.pt` (COCO-pretrained)|
|Epochs|50|
|Image size|640|
|Batch|16|
|Optimizer|AdamW (lr 0.0011, auto)|
|Hardware|Google Colab, T4 GPU (~1.5 h)|
|Augmentation|mosaic, fliplr 0.5, HSV jitter, erasing 0.4|

## 🚀 Reproduce

```bash
pip install ultralytics kagglehub
```

```python
import kagglehub, glob, os, yamlfrom ultralytics import YOLO# 1. Download datapath = kagglehub.dataset_download(    "pandrii000/hituav-a-highaltitude-infrared-thermal-dataset")# 2. Point config at ityaml_path = glob.glob(os.path.join(path, "**/dataset.yaml"),                      recursive=True)[0]# 3. Trainmodel = YOLO("yolov8n.pt")model.train(data=yaml_path, epochs=50, imgsz=640, batch=16,            name="hituav_yolov8n")# 4. Evaluate on held-out test splitmodel = YOLO("runs/detect/hituav_yolov8n/weights/best.pt")metrics = model.val(data=yaml_path, split="test")print(f"mAP50: {metrics.box.map50:.3f}")# 5. Detectresults = model.predict("your_thermal_image.png", conf=0.35)
```

## 🧠 Key Learnings

- **Validate labels before training, not after.** Visualizing 4 random GT images takes 30 seconds andcaught path and class-mapping issues that would have silently wasted a full training run.
- **Runtime matters.** An initial run on a CPU runtime moved at ~26 min/epoch; switching to a T4 GPUcut that to ~1.5 min/epoch — the difference between a 25-hour and 1.5-hour iteration cycle.
- **Class imbalance shapes what you can claim.** OtherVehicle (12 instances) and DontCare (7) producemisleading per-class scores; honest reporting means flagging low-support classes rather than averaging them away.
- **Small objects demand threshold tuning.** At 100 m altitude a person is a few pixels; loweringconfidence to 0.15–0.25 substantially raises recall at a modest precision cost — the right choicedepends on whether misses or false alarms are costlier in deployment.

## 🔮 Future Work

- Train `yolov8s` at `imgsz=960` to improve Person recall
- Day vs. night performance breakdown (metadata encoded in filenames)
- Merge or drop low-support classes; per-altitude error analysis
- Export to ONNX + lightweight Gradio demo for interactive inference

## 📜 Citation

```bibtex
@article{suo2022hituav,  title   = {HIT-UAV: A High-altitude Infrared Thermal Dataset for Unmanned Aerial Vehicles},  author  = {Suo, Jiashun and Wang, Tianyi and Zhang, Xingzhou and Chen, Haiyang             and Zhou, Wei and Shi, Weisong},  journal = {arXiv preprint arXiv:2204.03245},  year    = {2022}}@software{yolov8_ultralytics,  title = {Ultralytics YOLOv8},  author = {Jocher, Glenn and Chaurasia, Ayush and Qiu, Jian},  year  = {2023},  url   = {https://github.com/ultralytics/ultralytics}}
```

## ⚖️ License

Code released under the MIT License. The HIT-UAV dataset is credited to Suo et al. (2022) —please follow the original dataset's terms for data usage.

---
