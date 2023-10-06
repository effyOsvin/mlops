import torch
import torch.nn as nn
from .data_prep import train_data, valid_data
from .runner import CNNRunner

# Last layer (embeddings) size for CNN models
EMBEDDING_SIZE = 128

# Number of classes in the dataset
NUM_CLASSES = 2

ckpt_name_cnn = './mlops/bin/model_cnn.ckpt'

class Flatten(nn.Module):
    def forward(self, x):
        # finally we have it in pytorch
        return torch.flatten(x, start_dim=1)


def train_model():
    model_cnn = nn.Sequential()
    model_cnn.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3))
    model_cnn.add_module('pool1', nn.MaxPool2d(2))
    model_cnn.add_module('relu1', nn.ReLU())
    model_cnn.add_module('conv2', nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3))
    model_cnn.add_module('pool2', nn.MaxPool2d(2))
    model_cnn.add_module('relu2', nn.ReLU())
    model_cnn.add_module('conv3', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3))
    model_cnn.add_module('pool3', nn.MaxPool2d(2))
    model_cnn.add_module('relu3', nn.ReLU())
    model_cnn.add_module('conv4', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3))
    model_cnn.add_module('pool4', nn.MaxPool2d(2))
    model_cnn.add_module('relu4', nn.ReLU())
    model_cnn.add_module('global_max_pooling', nn.AdaptiveMaxPool2d(1))
    model_cnn.add_module('dropout', nn.Dropout(0.3))
    model_cnn.add_module('flat', Flatten())
    model_cnn.add_module('fc', nn.Linear(128, EMBEDDING_SIZE))
    model_cnn.add_module('relu', nn.ReLU())
    model_cnn.add_module('dropout_6', nn.Dropout(0.3))
    model_cnn.add_module('fc_logits', nn.Linear(EMBEDDING_SIZE, NUM_CLASSES, bias=False))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    opt = torch.optim.Adam(model_cnn.parameters(), lr=1e-3)
    opt.zero_grad()

    runner_cnn = CNNRunner(model_cnn, opt, device, ckpt_name_cnn)
    train_batch_gen = train_data()
    val_batch_gen = valid_data()
    runner_cnn.train(train_batch_gen, val_batch_gen, n_epochs=10)
