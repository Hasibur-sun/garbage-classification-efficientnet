# Garbage Classification Using Deep Learning

Image classification of household waste into 10 categories using transfer learning with **EfficientNet-B6** (PyTorch). The model reaches **94.18% accuracy on a held-out test set**.

## Overview

Sorting waste correctly is a key step in recycling. This project fine-tunes an ImageNet-pretrained EfficientNet-B6 to recognise what kind of waste is in a photo, and includes a small Gradio demo for trying it on your own images.

## Dataset

[Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) from Kaggle (MIT license). The `original` folder is used: **12,259 images** across 10 classes.

| Class | Images |
|---|---|
| clothes | 1892 |
| glass | 1736 |
| plastic | 1597 |
| shoes | 1449 |
| cardboard | 1411 |
| paper | 1336 |
| metal | 930 |
| battery | 756 |
| biological | 699 |
| trash | 453 |

The classes are imbalanced (`clothes` has about 4x as many images as `trash`), so the split is stratified. Images average roughly 291 x 304 px.

## Pipeline

1. **Exploration**: class counts, image resolution distribution, random samples, average RGB intensity.
2. **Cleaning**: removes non-image files and corrupted images that fail PIL verification.
3. **Split**: stratified 70 / 15 / 15 (`random_state=42`): 8,581 train, 1,839 validation, 1,839 test.
4. **Preprocessing**: resize to 224 x 224 and normalise with ImageNet mean/std. Training adds random horizontal and vertical flips, rotation up to 30 degrees, and brightness/contrast jitter.
5. **Model**: EfficientNet-B6 pretrained on ImageNet, with the final layer replaced by a 10-class linear layer.
6. **Training** (Adam, cross-entropy, batch size 16, Tesla T4 GPU):
   - Stage 1: 3 epochs at lr = 1e-3
   - Stage 2 (fine-tuning): 2 epochs at lr = 1e-4
7. **Evaluation**: validation and test accuracy, plus a confusion matrix.
8. **Demo**: Gradio app that shows the top-5 class probabilities for an uploaded image.

## Results

| Stage | Train accuracy | Validation accuracy |
|---|---|---|
| After 3 epochs (lr 1e-3) | 88.63% | 86.73% |
| After fine-tuning, 2 more epochs (lr 1e-4) | 94.34% | 93.20% |

**Final test accuracy: 94.18%** (1,839 held-out images).

## Getting started

The notebook was developed in Google Colab with a GPU.

1. Open `Garbage_Classification.ipynb` in Colab (or Jupyter with a CUDA GPU).
2. Install dependencies: `pip install -r requirements.txt`
3. Provide a Kaggle API token (`kaggle.json`) so the dataset can be downloaded.
4. Run the cells in order. The notebook saves `garbage_eff_b6.pth` (after stage 1) and `garbage_eff_b6_finetuned.pth` (after fine-tuning).

Model weights are not stored in this repository because of their size.

### Using the Gradio demo

After training, run the deployment cell. It rebuilds EfficientNet-B6, loads `garbage_eff_b6_finetuned.pth`, and launches a web UI where you upload an image and see the predicted class probabilities.

## Repository structure

```
.
├── Garbage_Classification.ipynb   # data exploration, training, evaluation, demo
├── requirements.txt
└── README.md
```

## Known limitations

- The saved notebook outputs for the per-class classification report and the Gradio demo show `NameError`s, because those cells were run after the Colab runtime had reset. Re-running the notebook top to bottom fixes this; per-class precision and recall are therefore not reported here.
- Training was short (5 epochs in total), so there is likely room for further improvement.
- Class imbalance is not handled beyond stratified splitting (no class weights or resampling).
- The notebook does not use a `torch.manual_seed`, so exact results will vary slightly between runs.

## Possible improvements

- Train longer with a learning-rate scheduler and early stopping.
- Add class weighting or oversampling for the smaller classes (`trash`, `biological`, `battery`).
- Compare against other backbones (e.g. EfficientNet-B0, ResNet, ConvNeXt).
- Deploy the demo to Hugging Face Spaces.

## Tech stack

Python, PyTorch, torchvision, scikit-learn, pandas, seaborn, matplotlib, Pillow, Gradio.
