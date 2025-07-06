# QuickDraw CNN Classifier

This project builds and trains a Convolutional Neural Network (CNN) to classify hand-drawn sketches from the [QuickDraw dataset](https://quickdraw.withgoogle.com/data). It covers the complete ML pipeline: from converting stroke-based vector drawings into grayscale images, to training and evaluating a deep learning model using PyTorch.

### Features
- Uses 15 diverse QuickDraw classes (e.g., Eiffel Tower, Apple, Airplane, Ant, etc.)
- Converts stroke data (.npz) into 128x128 grayscale images
- Builds a CNN from scratch (no transfer learning)
- Supports training with GPU (Colab)
- Achieves reasonable accuracy with simple architecture

### Dataset
Each drawing is a sequence of strokes stored in `.npz` format. We preprocess and convert them into image tensors ready for CNN input.

### Model Summary
- 3 Convolutional layers with BatchNorm and ReLU
- MaxPooling & Dropout regularization
- Fully connected classifier head
- Trained using CrossEntropyLoss & Adam optimizer

### Sample Output
> ![image](https://github.com/user-attachments/assets/d2e7c9a1-9938-42fe-bcf0-ccd425fc985c)

---

Let me know if you'd like me to write the full `README.md` (with usage instructions, requirements, etc.) or help you generate a preview image for the repo!
