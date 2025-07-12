import torch
from torchvision import transforms
from PIL import Image, ImageOps


def predict(model, image_path, device, img = None):
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(),  # zamień na 1 kanał (czarno-biały)
        transforms.Resize((28, 28)),  # MNIST ma 28x28 px
        transforms.ToTensor(),  # konwertuj do tensora
        transforms.Normalize((0.1307,), (0.3081,))  # normalizacja jak w MNIST
    ])

    if img is None:
        image = Image.open(image_path).convert("RGB")
    else:
        image = img
        image = ImageOps.invert(image)

    image = transform(image).unsqueeze(0).to(device)  # [1, 1, 28, 28]

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    return prediction

if __name__ == "__main__":
    image = Image.open('../data/test_digit.png').convert("RGB")
    print(image)

