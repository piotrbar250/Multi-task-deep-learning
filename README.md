# Multi-Task Learning for Geometric Shape Classification and Counting

PyTorch project for multi-task deep learning on the Geometric Shape Numbers dataset.  
It uses a shared backbone with two heads:
- one for image classification (shape/number class),
- one for regression (counts of geometric shapes in the image).

The training notebook runs three types of experiments:
- classification only,
- regression only,
- several variants of multi-task learning (classification + regression).

The metrics notebook computes multiple evaluations showing that multi-task learning achieves nearly the same accuracy as classification-only training, while significantly reducing training overhead.

## Requirements

- Python 3.8+
- torch
- torchvision
- pandas
- numpy
- pillow
- matplotlib
- scikit-learn

Install with:

```bash
pip install torch torchvision pandas numpy pillow matplotlib scikit-learn
