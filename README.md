# Breast Cancer Classification — TensorFlow/Keras

Binary classification of breast tumors (malignant vs. benign) using a deep neural network built with TensorFlow/Keras.

## Dataset

[UCI Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) — 569 samples, 30 numeric features derived from digitized images of fine needle aspirates (FNA) of breast masses. Target: `0` = Malignant, `1` = Benign.

## Project Structure

The notebook walks through three progressively improved models.

**1. Baseline (overfitting)** — 4-layer dense network trained for 600 epochs with no regularisation, demonstrating clear overfitting via a diverging validation loss curve.

**2. Early Stopping** — Same architecture but training is halted automatically when `val_loss` stops improving (`patience=30`), preventing overfitting without manual epoch tuning.

**3. Dropout + Early Stopping** — Dropout layers (`rate=0.3`) added after the first two dense layers. Both training and validation loss decrease together and plateau, indicating good generalisation.

## Results

The final model (Dropout + Early Stopping) achieves ~97% accuracy on the test set with only ~3 misclassifications out of 114 test samples.

## Stack

`Python` `TensorFlow/Keras` `scikit-learn` `pandas` `NumPy` `Matplotlib` `seaborn`

## How to Run

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
jupyter notebook Classification_Breast_Cancer_Wisconsin_Diagnostic.ipynb
```

> The dataset file `cancer_classification.csv` must be in the same directory as the notebook.

## Author

**Moaaz Ahmed**
- GitHub: [@Moaaz Ahmed](https://github.com/MoaazSalter)