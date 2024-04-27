import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, HubertModel

from audio_downsampler import AudioDownsample

processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft", cache_dir=".")
audio_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft", cache_dir=".")
text_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training params
NUM_EPOCHS = 1000

# Model
audio_downsample = AudioDownsample().to(device)

# Optimizer
optimizer = torch.optim.AdamW(audio_downsample.parameters(), lr=5e-3, weight_decay=1e-5)

# Scheduler
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=3,
    factor=0.5,
    threshold_mode="rel",
    min_lr=1e-8,
    threshold=0.01,
)

# Loss Calculator
criterion = torch.nn.CrossEntropyLoss()


def train(num_epochs=1000):
    # Training loop
    for epoch in range(num_epochs):
        # TODO: get details from a data loader
        for inputs, labels in dataloader:
            # Forward pass
            # TODO: fix this line
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print loss every few epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print('Training finished!')


if __name__ == '__main__':
    train(NUM_EPOCHS)