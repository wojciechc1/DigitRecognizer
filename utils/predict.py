import torch
from torchvision import transforms
from PIL import Image


def predict(model, image_path, device):
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(),  # zamień na 1 kanał (czarno-biały)
        transforms.Resize((28, 28)),  # MNIST ma 28x28 px
        transforms.ToTensor(),  # konwertuj do tensora
        transforms.Normalize((0.1307,), (0.3081,))  # normalizacja jak w MNIST
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # [1, 1, 28, 28]

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    return prediction

