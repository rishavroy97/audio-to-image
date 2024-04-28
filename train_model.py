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
DATA_DIR = "/scratch/as18464/data/audio"
START_EPOCH = 0
CHECKPOINT_DIR = './checkpoints'

# Audio Downsampler
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
criterion = torch.nn.MSELoss()

# Dataset
dataset = AudioDataset(CSV_FILE, DATA_DIR)

# Dataloader
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def train(start = 0, num_epochs=10):

    checkpoint_path = f'{CHECKPOINT_DIR}/checkpoint_epoch{epoch}.pth'
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        audio_downsample = AudioDownsample()
        audio_downsample.load_state_dict(checkpoint['model_state_dict']).to(device)
        optimizer = torch.optim.AdamW(checkpoint['optimizer_state_dict'])

    for epoch in range(num_epochs):
        for audios, labels in dataloader:
            # Forward pass
            audios = audios.to(device)
            audios = audios.squeeze(1)
            labels = list(labels)

            processed_audios = processor(audios, return_tensors="pt", sampling_rate=16000).input_values.squeeze(0).to(
                device)
            audio_embeddings = audio_model(processed_audios[:, :AUDIO_INPUT_CAP]).last_hidden_state.to(device)
            audio_ds_output = audio_downsample(audio_embeddings).to(device)

            text = text_model.encode(labels)
            prompt_embeddings = torch.tensor(text).to(device)
            loss = criterion(audio_ds_output, prompt_embeddings)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr_scheduler.step(epoch)

        # Print loss every few epochs
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss
        }, checkpoint_path)

    print('Training finished!')
    print(f'Final Loss: {loss.item():.4f}')


if __name__ == '__main__':
    train(START_EPOCH, NUM_EPOCHS)
