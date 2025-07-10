from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F

import torch

from torch.utils.data import Subset

import matplotlib.pyplot as plt


transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

# mniejsza ilosc danych
train_data = Subset(train_data, range(6000))
test_data = Subset(test_data, range(1000))

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

print("Size of train and test data:", len(train_data), len(test_data))


# tworzenie prostej sieci

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # spłaszcz obraz
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x

model1 = SimpleNN()
model2 = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.01)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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



# Sprawdzanie dokładności przed uczeniem:

test_loss, test_acc = test(model1, test_loader, loss_fn, device)
print(f"Model1 - przed treningiem: Test accuracy = {test_acc:.4f}")

test_loss, test_acc = test(model2, test_loader, loss_fn, device)
print(f"Model2 - przed treningiem: Test accuracy = {test_acc:.4f}")


# listy na metryki
train_accs1, test_accs1 = [], []
train_losses1, test_losses1 = [], []

train_accs2, test_accs2 = [], []
train_losses2, test_losses2 = [], []



for epoch in range(5):
    train_loss1, train_acc1 = train(model1, train_loader, loss_fn, optimizer1, device)
    test_loss1, test_acc1 = test(model1, test_loader, loss_fn, device)

    train_loss2, train_acc2 = train(model2, train_loader, loss_fn, optimizer2, device)
    test_loss2, test_acc2 = test(model2, test_loader, loss_fn, device)

    # zapisujemy wyniki do list
    train_accs1.append(train_acc1)
    test_accs1.append(test_acc1)
    train_losses1.append(train_loss1)
    test_losses1.append(test_loss1)

    train_accs2.append(train_acc2)
    test_accs2.append(test_acc2)
    train_losses2.append(train_loss2)
    test_losses2.append(test_loss2)

    print(f"Epoch {epoch+1}:")
    print(f" SimpleNN -> Acc: {test_acc1:.4f}, Loss: {test_loss1:.4f}")
    print(f" CNN      -> Acc: {test_acc2:.4f}, Loss: {test_loss2:.4f}")


# rysowanie wykresów
epochs = range(1, 6)

plt.figure(figsize=(12,5))

plt.suptitle(f'Train: {len(train_data)}, Test:{ len(test_data)}')

plt.subplot(1,2,1)
plt.plot(epochs, train_accs1, 'b-', label='SimpleNN Train Acc')
plt.plot(epochs, test_accs1, 'b--', label='SimpleNN Test Acc')
plt.plot(epochs, train_accs2, 'r-', label='CNN Train Acc')
plt.plot(epochs, test_accs2, 'r--', label='CNN Test Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, train_losses1, 'b-', label='SimpleNN Train Loss')
plt.plot(epochs, test_losses1, 'b--', label='SimpleNN Test Loss')
plt.plot(epochs, train_losses2, 'r-', label='CNN Train Loss')
plt.plot(epochs, test_losses2, 'r--', label='CNN Test Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()