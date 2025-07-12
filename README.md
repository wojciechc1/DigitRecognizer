# 🧠 Digit Recognizer – Model Comparison & GUI

This project is a simple tool to **compare different neural networks** on the MNIST digit recognition task. It also includes a small **GUI** where users can draw digits and see predictions from a trained model.

---

## 🔍 What It Does

- Trains and compares different models (like SimpleNN and CNN)
- Shows accuracy and loss plots for each model
- Saves the trained models
- Lets users draw digits in a GUI and predict them using the trained network

---

## 📁 Project Structure
```bash
digit_recognizer/
│
├── data/ # data for training and testing
├── models/ # Neural network classes
├── test/ 
├── train/ 
├── utils/ # Utils
├── gui/ # GUI for drawing digits
├── saved_models/ # Folder where trained models are saved
├── predict_image.py # predict digit from a file 
├── train_model.py # Train models
├── requirements.txt # requirements
└── README.md
```

## 🚀 How to Use

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train and compare models

```bash
    python train_model.py
```
This trains multiple models and shows plots comparing their accuracy and loss.

### 3. Run the GUI

```bash
    python gui/app.py
```