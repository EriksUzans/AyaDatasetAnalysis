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

<img width="1589" height="815" alt="image" src="https://github.com/user-attachments/assets/682ead19-3340-4adf-993c-67333563adf9" />
<img width="1146" height="1589" alt="image" src="https://github.com/user-attachments/assets/ee0854fc-1d65-42b7-9595-a076b4d9f647" />


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



## Confusion matrix
<img width="3000" height="2250" alt="confusion_matrix" src="https://github.com/user-attachments/assets/dba5b84a-6c9a-495c-b295-8a3c34c4871d" />

Car is nearly perfectly separated (701/719 correct).   Person holds the highest diagonal count (1,080) but accounts for the most missed detections (88 lost to background).   Inter-class confusion between Person and Bicycle is negligible (only 1 cross-classification instance); the main failure mode is false positives from background regions (272 predicted Person, 195 predicted Bicycle).   OtherVehicle and DontCare are statistically unreliable due to extreme sample scarcity (12 and 7 true instances). 

## Training curves

<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/a5fc7e16-dcb4-4dbb-92b0-ae6391716e4f" />


### Losses (box, classification, DFL) and metrics tracked per epoch across the train and validation splits.

All three loss terms fall steeply within the first ~10 epochs and plateau by mid-training — the COCO-pretrained backbone transfers well to thermal imagery (Bicycle, Car and Person head weights mapped directly onto this task).
Validation loss tracks training loss without divergence at 50 epochs — no overfitting yet, implying the model is under-trained rather than over-trained. Additional capacity (YOLOv8s) or resolution (imgsz=960) should buy further gains before regularization becomes necessary.
mAP@0.5 and mAP@0.5:0.95 converge to 0.784 and 0.503 respectively. The large gap between them is characteristic of small-object detection: at strict IoU thresholds, a 2–3 px localization slip on a ~10 px-wide ground-truth box can drop a detection from IoU 0.8 to below 0.6 — penalizing mAP@0.5:0.95 heavily while leaving mAP@0.5 untouched.









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
