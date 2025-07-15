def train_manual(model, loader, epochs = 10, learning_rate = 0.01):

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0

        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.view(inputs.size(0), -1) # Spłaszcz dane
            labels = labels.float() # etykiety na float (do regresji)

            predictions = model.forward(inputs)

            loss = ((predictions - labels) ** 2).mean()

            error = predictions - labels

            # obliczanie gradientow
            grad_w = 2 * inputs.T @ error / inputs.size(0)  # [784, 1]
            grad_b = 2 * error.mean()

            # aktualizacja parametrow modelu
            model.w -= learning_rate * grad_w.squeeze()  # dostosowywanie nachylenia funkcji
            model.b -= learning_rate * grad_b   # dostosowywanie punktu przeciecia

            total_loss += loss.item()

            for p, l in zip(predictions, labels):
                if round(p.item()) == l.item():
                    correct += 1

        accuracy = correct / len(loader.dataset)
        print(accuracy)