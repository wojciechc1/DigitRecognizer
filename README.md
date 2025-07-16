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


## 🚀 Getting started

### 1. Install dependencies 

```bash
    # install dependencies
    pip install -r requirements.txt

    # train CNN and SimpleNN 
    python train_model.py

    # GUI - draw and predict
    python predict_image.py

    # test and train nn created from scratch
    python gui/manual_nn_test_train.py
```
