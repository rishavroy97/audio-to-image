import argparse
import os

import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from transformers import AutoProcessor, HubertModel

from audio_dataset import AudioDataset
from audio_downsampler import AudioDownsample

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft", cache_dir="./models")
audio_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft", cache_dir="./models").to(device)
text_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', cache_folder='./models')

# Training params
AUDIO_INPUT_CAP = 10000
NUM_EPOCHS = 100
BATCH_SIZE = 32
CSV_FILE = "./vggsound.csv"
DATA_DIR = "./data/audio"
START_EPOCH = 0
CHECKPOINT_DIR = './checkpoints'
BEST_MODEL_DIR = '.models/audio-downsample'

# Loss Calculator
criterion = torch.nn.MSELoss()

# Datasets
train_dataset = AudioDataset(CSV_FILE, DATA_DIR, split='train')
val_dataset = AudioDataset(CSV_FILE, DATA_DIR, split='val')

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)


def forward_pass(audios, labels, down_sampler):
    audios = audios.to(device)
    audios = audios.squeeze(1)
    labels = list(labels)

    processed_audios = processor(audios, return_tensors="pt", sampling_rate=16000).input_values.squeeze(0).to(
        device)
    audio_embeddings = audio_model(processed_audios[:, :AUDIO_INPUT_CAP]).last_hidden_state.to(device)
    audio_ds_output = down_sampler(audio_embeddings).to(device)

    text = text_model.encode(labels)
    prompt_embeddings = torch.tensor(text).to(device)

    return audio_ds_output, prompt_embeddings


def calculate_validation_loss(model, val_loader):
    model.eval()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for audios, labels in val_loader:
            # Forward Pass
            audio_ds_output, prompt_embeddings = forward_pass(audios, labels, model)
            loss = criterion(audio_ds_output, prompt_embeddings)

            total_loss += loss.item() * audios.size(0)
            num_samples += audios.size(0)

    return total_loss / num_samples


def train(start=0, num_epochs=10):
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

    # Load checkpoint
    checkpoint_path = f'{CHECKPOINT_DIR}/checkpoint_epoch{start}.pth'
    if start > 0 and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        audio_downsample.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.AdamW(checkpoint['optimizer_state_dict'])
        loss = checkpoint['train_loss']
        start = checkpoint['epoch']
        print(f'Loaded checkpoint at epoch {start}')

    # Load best loss
    best_model_path = f'{BEST_MODEL_DIR}/best_model.pth'
    best_loss = float('inf')
    loss = float('inf')
    if os.path.exists(best_model_path):
        best_model = torch.load(best_model_path)
        best_loss = best_model['loss']

    for epoch in range(start, num_epochs):

        # Training
        audio_downsample.train()
        for audios, labels in train_dataloader:
            # Forward Pass
            audio_ds_output, prompt_embeddings = forward_pass(audios, labels, audio_downsample)

            loss = criterion(audio_ds_output, prompt_embeddings)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr_scheduler.step(epoch)

        # Validation phase
        val_loss = calculate_validation_loss(audio_downsample, val_dataloader)

        if val_loss < best_loss:
            best_loss = val_loss

            # Save the best model
            if not os.path.exists(BEST_MODEL_DIR):
                os.makedirs(BEST_MODEL_DIR)

            torch.save({
                'epoch': epoch,
                'model_state_dict': audio_downsample.state_dict(),
                'loss': best_loss,
            }, best_model_path)
            print(f'Best model saved at epoch {epoch}')

        if (epoch + 1) % 30 == 0:
            # Print loss every few epochs
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

            # Save checkpoint
            if not os.path.exists(CHECKPOINT_DIR):
                os.makedirs(CHECKPOINT_DIR)

            checkpoint_path = f'{CHECKPOINT_DIR}/checkpoint_epoch{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': audio_downsample.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': loss
            }, checkpoint_path)
            print(f"Saved checkpoint: {epoch}")

    print('Training finished!')
    print(f'Final Loss: {loss.item():.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Model for 1000 Epoch")
    parser.add_argument("--start", type=int, default=START_EPOCH, help="Starting index epoch")
    args = parser.parse_args()

    train(args.start, NUM_EPOCHS)
