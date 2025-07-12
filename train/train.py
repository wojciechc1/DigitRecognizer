def train(model, loader, loss_fn, optimizer, device):
    model.train()  # tryb treningu
    total_loss = 0
    correct = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # forward
        outputs = model(images)

        # oblicz stratę (loss)
        loss = loss_fn(outputs, labels)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # zapisz stratę i trafienia
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    # średnia strata i accuracy (trafność)
    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)

    return avg_loss, accuracy