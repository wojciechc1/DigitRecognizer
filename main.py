from train.train_manual import train_manual
from test.test_manual import test_manual
from utils.data_loader import get_data
from model.model import LinearRegressionModel


train_loader, test_loader = get_data(60000, 10000, 64)

model = LinearRegressionModel()

train_manual(model, train_loader)
test_manual(model, test_loader)