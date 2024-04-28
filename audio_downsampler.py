import torch.nn as nn


class AudioDownsample(nn.Module):
    def __init__(self):
        super(AudioDownsample, self).__init__()

        input_size = 31 * 1024
        embedding_size = 77 * 1024
        self.proj1 = nn.Linear(input_size, 512)
        self.proj2 = nn.Linear(512, 256)
        self.proj3 = nn.Linear(256, embedding_size)
        self.activation = nn.GELU()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.activation(self.proj1(x))
        x = self.activation(self.proj2(x))
        x = self.proj3(x)

        return x.view(batch_size, 77, 1024)
