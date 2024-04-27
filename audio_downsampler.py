import torch.nn as nn


class AudioDownsample(nn.Module):
    def __init__(self):
        # super().__init__()
        super(AudioDownsample, self).__init__()

        hidden_size = 31 * 1024
        embedding_size = 384
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.proj2 = nn.Linear(hidden_size, embedding_size)
        self.activation = nn.GELU()

    def forward(self, x):
        # x /= torch.max(torch.abs(x))
        x = x.view(-1)
        x = self.activation(self.proj1(x))
        return self.proj2(x)
