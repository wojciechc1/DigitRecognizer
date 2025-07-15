from train.train_manual import train_manual
from test.test_manual import test_manual
from utils.data_loader import get_data
from model.model import LinearRegressionModel
from utils.utils import plot_metrics, save_model


# config
learning_rate = 0.001
train_data_size = 60000
test_data_size = 10000
batch_size = 64
epochs = 5

# wczytywanie danych
train_loader, test_loader = get_data(train_data_size, test_data_size, batch_size)

# listy na metryki
train_accs1, test_accs1 = [], []
train_losses1, test_losses1 = [], []


model = LinearRegressionModel()

for epoch in range(epochs):
    train_loss1, train_acc1 = train_manual(model, train_loader)
    test_loss1, test_acc1 = test_manual(model, test_loader)


    train_accs1.append(train_acc1)
    test_accs1.append(test_acc1)
    train_losses1.append(train_loss1)
    test_losses1.append(test_loss1)
    print('Epoch: {:04d}'.format(epoch + 1), flush=True)



import matplotlib.pyplot as plt

epochs = range(1, len(train_accs1) + 1)

plt.figure(figsize=(12, 5))

# Wykres dokładności
plt.subplot(1, 2, 1)
plt.plot(epochs, train_accs1, label='Train Accuracy')
plt.plot(epochs, test_accs1, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy over Epochs')
plt.legend()

# Wykres strat
plt.subplot(1, 2, 2)
plt.plot(epochs, train_losses1, label='Train Loss')
plt.plot(epochs, test_losses1, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Loss over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

