import tkinter as tk
from PIL import Image, ImageDraw
import torch
from model.model import CNN, SimpleNN, BigMLP
from utils.predict import predict
from utils.utils import load_model

class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Digit Recognizer")

        self.canvas = tk.Canvas(self.master, width=280, height=280, bg="white")
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<B1-Motion>", self.draw)

        self.pred_label1 = tk.Label(self.master, text="Prediction: ", font=("Arial", 24))
        self.pred_label1.pack(pady=20)

        self.pred_label2 = tk.Label(self.master, text="Prediction: ", font=("Arial", 24))
        self.pred_label2.pack(pady=20)

        self.pred_label3 = tk.Label(self.master, text="Prediction: ", font=("Arial", 24))
        self.pred_label3.pack(pady=20)

        self.predict_button = tk.Button(self.master, text="Predict", command=self.update_prediction)
        self.predict_button.pack(pady=5)

        self.clear_button = tk.Button(self.master, text="Clear", command=self.clear)
        self.clear_button.pack()

        self.image = Image.new("L", (280, 280), color=255)
        self.draw_image = ImageDraw.Draw(self.image)

        # model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model1 = load_model(SimpleNN, "../saved_models/model_simple_nn.pth", self.device)
        self.model2 = load_model(CNN, "../saved_models/model_cnn.pth", self.device)
        self.model3 = load_model(BigMLP, "../saved_models/model_big_mlp.pth", self.device)


    def draw(self, event):
        x, y = event.x, event.y
        r = 10
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")
        self.draw_image.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=255)
        self.draw_image = ImageDraw.Draw(self.image)
        self.pred_label1.config(text="Prediction: ")
        self.pred_label2.config(text="Prediction: ")
        self.pred_label3.config(text="Prediction: ")

    def update_prediction(self):
        img = self.image.resize((28, 28)).copy()
        print(img)
        pred1 = predict(self.model1, '', self.device, img)
        pred2 = predict(self.model2, '', self.device, img)
        pred3 = predict(self.model3, '', self.device, img)
        self.pred_label1.config(text=f"Prediction: {pred1}")
        self.pred_label2.config(text=f"Prediction: {pred2}")
        self.pred_label3.config(text=f"Prediction: {pred3}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
