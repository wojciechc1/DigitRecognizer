import torch

def train_manual(model, loader, learning_rate):
    total_loss = 0.0
    correct = 0


    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.view(inputs.size(0), -1) # Spłaszcz dane
        labels = labels.float() # etykiety na float (do regresji)

        # one hot wektory:
        one_hot_labels = []
        for i, label in enumerate(labels):
            one_hot_labels.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            one_hot_labels[i][int(label)] = 1

        one_hot_tensor = torch.tensor(one_hot_labels, dtype=torch.float32)


        predictions, y1, a1 = model.forward(inputs)

        loss = ((predictions - one_hot_tensor) ** 2).mean()

        error = predictions - one_hot_tensor #shape 2, 10

        # obliczanie gradientow
        #grad_w = 2 * inputs.T @ error / inputs.size(0)  # [784, 1]
        #grad_b = 2 * error.mean()

        grad_w2 = (a1.T @ error) / inputs.size(0)  # shape: [64, 10]
        grad_b2 = error.mean() # shape: [10]

        # aktualizacja parametrow modelu
        model.w2 -= learning_rate * grad_w2  # dostosowywanie nachylenia funkcji
        model.b2 -= learning_rate * grad_b2  # dostosowywanie punktu przeciecia

        error_hidden = error @ model.w2.T  # shape: [batch_size, 64]
        relu_grad = (y1 > 0).float()
        error_hidden = error_hidden * relu_grad

        grad_w1 = (inputs.T @ error_hidden) / inputs.size(0)   # shape: [784, 64]
        grad_b1 = error_hidden.mean()  # shape: [64]

        model.w1 -= learning_rate * grad_w1
        model.b1 -= learning_rate * grad_b1

        total_loss += loss.item()

        for p, l in zip(predictions, one_hot_labels):
            predicted_index = torch.argmax(p).item()
            true_index = torch.argmax(torch.tensor(l)).item()

            if predicted_index == true_index:
                correct += 1

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    print('train acc|loss:', accuracy, '|', avg_loss)


    return avg_loss, accuracy