import torch

num_classes = 2
num_epochs = 20
batch_size = 16
learning_rate = 0.0001
device = 'cuda' if torch.cuda.is_available() else 'cpu'