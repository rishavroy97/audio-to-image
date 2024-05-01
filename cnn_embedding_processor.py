import torch.nn as nn


class AudioCNNEmbeddingProcessor(nn.Module):
    def __init__(self):
        super(AudioCNNEmbeddingProcessor, self).__init__()

        # Define layers for your neural network
        self.conv1 = nn.Conv1d(in_channels=31, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x shape: [BATCH_SIZE, 31, 1024]

        # Apply first convolutional layer
        x = self.conv1(x)
        x = self.relu(x)

        # Apply second convolutional layer
        x = self.conv2(x)
        x = self.relu(x)

        # Apply batch normalization
        x = self.batchnorm(x)

        # Apply third convolutional layer
        x = self.conv3(x)

        # Output shape: [BATCH_SIZE, 1024, L'] where L' depends on input length and kernel size

        # You may need to reshape or transpose the output to match the desired output shape
        # Reshape x to match the output dimensions [BATCH_SIZE, 77, 1024]
        x = x.permute(0, 2, 1)  # Swap dimensions 1 and 2
        x = x[:, :77, :]  # Trim or pad to match desired output length (77)

        return x
