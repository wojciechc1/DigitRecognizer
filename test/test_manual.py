import torch

def test_manual(model, loader):
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


        total_loss += loss.item()

        for p, l in zip(predictions, one_hot_labels):
            predicted_index = torch.argmax(p).item()
            true_index = torch.argmax(torch.tensor(l)).item()

            if predicted_index == true_index:
                correct += 1

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    print('test acc|loss:', accuracy, '|', avg_loss)

    return avg_loss, accuracy