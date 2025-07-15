import torch.nn as nn
import torch
from model.model import SimpleNN, CNN, BigMLP
from utils.data_loader import get_data
from utils.utils import plot_metrics, save_model
from test.test import test
from train.train import train
import time

def main():

    start = time.time() # mierzenie czasu dzialania programu

    # config
    learning_rate = 0.001
    train_data_size = 6000
    test_data_size = 10000
    batch_size = 64
    epochs = 3

    # wczytywanie danych
    train_loader, test_loader = get_data(train_data_size, test_data_size, batch_size)


    # tworzenie modelu
    model1 = SimpleNN()
    model2 = CNN()
    model3 = BigMLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)
    optimizer3 = torch.optim.Adam(model2.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Sprawdzanie dokładności przed uczeniem:
    test_loss, test_acc = test(model1, test_loader, loss_fn, device)
    print(f"[MAIN] Model1 - przed treningiem: Test accuracy = {test_acc:.4f}")

    test_loss, test_acc = test(model2, test_loader, loss_fn, device)
    print(f"[MAIN] Model2 - przed treningiem: Test accuracy = {test_acc:.4f}")

    test_loss, test_acc = test(model3, test_loader, loss_fn, device)
    print(f"[MAIN] Model3 - przed treningiem: Test accuracy = {test_acc:.4f}")


    # listy na metryki
    train_accs1, test_accs1 = [], []
    train_losses1, test_losses1 = [], []

    train_accs2, test_accs2 = [], []
    train_losses2, test_losses2 = [], []

    train_accs3, test_accs3 = [], []
    train_losses3, test_losses3 = [], []


    for epoch in range(epochs):
        train_loss1, train_acc1 = train(model1, train_loader, loss_fn, optimizer1, device)
        test_loss1, test_acc1 = test(model1, test_loader, loss_fn, device)

        train_loss2, train_acc2 = train(model2, train_loader, loss_fn, optimizer2, device)
        test_loss2, test_acc2 = test(model2, test_loader, loss_fn, device)

        train_loss3, train_acc3 = train(model3, train_loader, loss_fn, optimizer3, device)
        test_loss3, test_acc3 = test(model3, test_loader, loss_fn, device)

        # zapisujemy wyniki do list
        train_accs1.append(train_acc1)
        test_accs1.append(test_acc1)
        train_losses1.append(train_loss1)
        test_losses1.append(test_loss1)

        train_accs2.append(train_acc2)
        test_accs2.append(test_acc2)
        train_losses2.append(train_loss2)
        test_losses2.append(test_loss2)

        train_accs3.append(train_acc2)
        test_accs3.append(test_acc2)
        train_losses3.append(train_loss2)
        test_losses3.append(test_loss2)

        print(f"[MAIN] Epoch {epoch+1}:")
        print(f" SimpleNN -> Acc: {test_acc1:.4f}, Loss: {test_loss1:.4f} | {train_acc1}, {train_loss1}")
        print(f" CNN      -> Acc: {test_acc2:.4f}, Loss: {test_loss2:.4f} | {train_acc2}, {train_loss2}")
        print(f" BigMLP   -> Acc: {test_acc3:.4f}, Loss: {test_loss3:.4f} | {train_acc3}, {train_loss3}")

    # rysowanie wykresow
    #plot_metrics(train_accs1, test_accs1, train_losses1, test_losses1, train_accs2, test_accs2, train_losses2, test_losses2)

    end = time.time()  # mierzenie czasu dzialania programu

    # Zapisywanie modelu
    #save_model(model1, "saved_models/model_simple_nn.pth")
    #save_model(model2, "saved_models/model_cnn.pth")
    #save_model(model3, "saved_models/model_big_mlp.pth")

    return end - start

if __name__ == "__main__":
    duration = main()
    print(f"[TIME] Czas treningu: {duration:.2f} sekundy")