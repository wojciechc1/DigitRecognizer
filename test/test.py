import torch

def test(model, loader, loss_fn, device):
    model.eval()  # tryb testowania
    total_loss = 0
    correct = 0

    with torch.no_grad():  # bez gradientów (szybciej)
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)

    return avg_loss, accuracy
