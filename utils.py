import matplotlib.pyplot as plt
import torch

def plot_metrics(train_accs1, test_accs1, train_losses1, test_losses1, train_accs2, test_accs2, train_losses2, test_losses2):
    epochs = range(1, len(train_accs1)+1)

    plt.figure(figsize=(12, 5))

    #plt.suptitle(f'Train: {len(train_data)}, Test:{len(test_data)}')

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accs1, 'b-', label='SimpleNN Train Acc')
    plt.plot(epochs, test_accs1, 'b--', label='SimpleNN Test Acc')
    plt.plot(epochs, train_accs2, 'r-', label='CNN Train Acc')
    plt.plot(epochs, test_accs2, 'r--', label='CNN Test Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses1, 'b-', label='SimpleNN Train Loss')
    plt.plot(epochs, test_losses1, 'b--', label='SimpleNN Test Loss')
    plt.plot(epochs, train_losses2, 'r-', label='CNN Train Loss')
    plt.plot(epochs, test_losses2, 'r--', label='CNN Test Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    return model
