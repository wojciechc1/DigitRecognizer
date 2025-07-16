# Digit Recognizer – Compare Models & Predict by Drawing
A simple playground for experimenting with different machine learning models on the MNIST digit recognition task.
Achieved ~85% training accuracy and ~88% test accuracy early in training using a 2-layer neural network implemented from scratch.

## 🔍 Features

- Manual 2-layer neural network built from scratch
- Comparison of multiple models (SimpleNN, CNN)
- Automatic training & testing pipeline with accuracy/loss plots
- Model saving & loading
- GUI: draw a digit and get instant prediction

---


## 🚀 Getting started (in process- refactoring)

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