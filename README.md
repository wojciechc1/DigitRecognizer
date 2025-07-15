# Digit Recognizer – Compare Models & Predict by Drawing
## (Manual Linear Regression + Neural Nets + GUI)

A simple but flexible playground for testing different machine learning models on the MNIST digit recognition task — including a handwritten Linear Regression from scratch (no PyTorch autograd), feedforward neural networks, and CNNs.

Comes with a live GUI where you can draw digits and see real-time predictions from trained models.
---

## 🔍 Features

- Manual linear regression from scratch (no libraries, just math!)
- Comparison of multiple models (SimpleNN, CNN)
- Automatic training & testing pipeline with accuracy/loss plots
- Model saving & loading
- GUI: draw a digit and get instant prediction
- CLI-based prediction from image files

---


## 🚀 Getting started (in process)

### 1. Install dependencies 

```bash
    # install dependencies
    pip install -r requirements.txt

    # train neural net
    python train_model.py

    # train manual regression
    python train_manual.py

    # test neural net
    python test/test_nn.py

    # launch GUI
    python gui/main.py
```