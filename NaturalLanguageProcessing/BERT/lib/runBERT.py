from train import train_network
from dataset import Dataset


print("Preparing the dataset for BERT...")
dataset = Dataset(train=True, prepareTSV=True)
dataset = Dataset(train=False, prepareTSV=True)

print("Training the BERT network...")
train_network()
