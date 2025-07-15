
def test_manual(model, loader, learning_rate = 0.01):
    total_loss = 0.0
    correct = 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.view(inputs.size(0), -1) # Spłaszcz dane
        labels = labels.float() # etykiety na float (do regresji)

        predictions = model.forward(inputs)

        loss = ((predictions - labels) ** 2).mean()

        total_loss += loss.item()

        for p, l in zip(predictions, labels):
            if round(p.item()) == l.item():
                correct += 1

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)

    return avg_loss, accuracy