from utils.utils import load_model
from model.model import SimpleNN, CNN, BigMLP
from utils.predict import predict
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ladowanie modeli
    model1 = load_model(SimpleNN, "./saved_models/model_simple_nn.pth", device)
    model2 = load_model(CNN, "./saved_models/model_cnn.pth", device)
    model3 = load_model(BigMLP, "./saved_models/model_big_mlp.pth", device)

    # testowanie obrazu
    image_path = "./data/test_digit.png"  # ścieżka do twojego obrazka
    prediction = predict(model1, image_path, device)
    print(f"Predicted digit by SimpleNN: {prediction}")

    prediction = predict(model2, image_path, device)
    print(f"Predicted digit by CNN: {prediction}")

    prediction = predict(model3, image_path, device)
    print(f"Predicted digit by BigMLP: {prediction}")


if __name__ == "__main__":
    main()