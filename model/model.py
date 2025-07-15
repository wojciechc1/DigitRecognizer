import torch.nn as nn
import torch.nn.functional as F
import torch


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


class BigMLP(nn.Module):
    def __init__(self):
        super(BigMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)




class LinearRegressionModel():
    def __init__(self):
        self.w = torch.randn(784) # waga - wspolczynnik - a
        self.b = torch.randn(1) # bias - wyraz wolny - b

    def relu(self, y):
        y_relu = []
        for val in y:
            if val.item() < 0:
                tmp = torch.tensor([0.0])
                y_relu.append(tmp)
            else:
                y_relu.append(val.unsqueeze(0))
        return torch.cat(y_relu)

    def forward(self, x):
        # przekazanie funkcji
        #y = self.w * x + self.b
        # iloczyn skalarny xd
        y = x @ self.w + self.b
        yr = self.relu(y)

        return yr