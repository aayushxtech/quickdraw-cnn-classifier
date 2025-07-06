# QuickDraw CNN Classifier

This project implements a Convolutional Neural Network (CNN) from scratch using PyTorch to classify vector-based hand-drawn sketches from the [QuickDraw dataset](https://quickdraw.withgoogle.com/data). It includes a complete ML pipeline: preprocessing stroke-based data, vector-to-image conversion, dataset construction, training, evaluation, and prediction.

---

### Features

-  Supports 15 categories from the QuickDraw SketchRNN dataset (e.g., "The Eiffel Tower", "Apple", "Airplane", etc.)
-  Converts vector stroke data into normalized grayscale images using custom rendering logic
-  Implements a deep CNN with convolutional blocks, batch normalization, dropout, and a fully connected classifier
-  Efficient training using GPU acceleration (tested on Google Colab)
-  Achieves competitive accuracy despite using a lightweight architecture without transfer learning
-  Clean PyTorch data pipeline with tensor formatting `(N, C, H, W)` and stratified dataset split

---

### Dataset

- Format: `.npz` files (Google's SketchRNN format)
- Each sample is a sequence of pen strokes: `(Δx, Δy, pen_state)`
- Preprocessing involves:
  - Stroke-to-image rendering using PIL (`ImageDraw`)
  - Centering, scaling, and padding strokes on a 128×128 canvas
  - Label assignment and normalization
- Images are converted to single-channel (grayscale) tensors

---

### Model Architecture

| Layer | Type | Channels |
|-------|------|----------|
| `Conv2D` | 1 → 32 | `3x3`, padding=1
| `BatchNorm2d` | 32 | + ReLU
| `MaxPool2D` | `2x2` | Downsampling
| `Conv2D` | 32 → 64 | `3x3`, padding=1
| `BatchNorm2d` | 64 | + ReLU
| `MaxPool2D` | `2x2` | Downsampling|
| `Conv2D` | 64 → 128 | `3x3`, padding=1
| `BatchNorm2d` | 128 | + ReLU
| `MaxPool2D` | `2x2` | Downsampling 
| `Linear` | 32768 → 256 | FC Layer + ReLU + Dropout 
| `Linear` | 256 → 15 | Output logits 

- **Loss Function**: `CrossEntropyLoss`
- **Optimizer**: `Adam (lr=1e-3)`

---

### Sample Output

> _(Sample classification output with predicted class vs actual)_

![image](https://github.com/user-attachments/assets/d2e7c9a1-9938-42fe-bcf0-ccd425fc985c)

