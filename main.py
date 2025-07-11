from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Subset
import matplotlib.pyplot as plt

from model import SimpleNN, CNN
from data_loader import get_data
from utils import plot_metrics
from test import test
from train import train


# wczytywanie danych
train_loader, test_loader = get_data(6000, 1000, 64)


# tworzenie modelu
model1 = SimpleNN()
model2 = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.01)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    print(f" SimpleNN -> Acc: {test_acc1:.4f}, Loss: {test_loss1:.4f} | {train_acc1}, {train_loss1}")
    print(f" CNN      -> Acc: {test_acc2:.4f}, Loss: {test_loss2:.4f} | {train_acc2}, {train_loss2}")


plot_metrics(train_accs1, test_accs1, train_losses1, test_losses1, train_accs2, test_accs2, train_losses2, test_losses2)