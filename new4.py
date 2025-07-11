import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

# Dataset Class (uses image_feature from JSON)
class EmotionCaptionDataset(Dataset):
    def __init__(self, json_file="final_preprocessed_dataset.json", tokenizer=None, max_len=30):
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Precomputed 2048-d image feature
        image_feature = torch.tensor(item["image_feature"], dtype=torch.float32)

        # One-hot 6-d emotion vector
        emotion_vector = torch.tensor(item["emotion_vector"], dtype=torch.float32)

        # Padded caption token IDs
        caption_ids = item["input_ids"][:self.max_len]
        caption_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(caption_ids))
        caption_ids = torch.tensor(caption_ids, dtype=torch.long)

        return image_feature, emotion_vector, caption_ids

# Model Class
class EmotionCaptioningLSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, emotion_dim=6, image_dim=2048, num_layers=1):
        super(EmotionCaptioningLSTM, self).__init__()

        self.fc_context = nn.Linear(image_dim + emotion_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image_feature, emotion_vector, captions):
        context = torch.cat((image_feature, emotion_vector), dim=1)  # [B, 2054]
        h0 = torch.tanh(self.fc_context(context)).unsqueeze(0)       # [1, B, H]
        c0 = torch.zeros_like(h0)

        embeddings = self.embedding(captions)                        # [B, T, E]
        outputs, _ = self.lstm(embeddings, (h0, c0))                 # [B, T, H]
        logits = self.fc_out(outputs)                                # [B, T, V]

        return logits

# Training Script
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    dataset = EmotionCaptionDataset(
        json_file="final_preprocessed_dataset.json",
        tokenizer=tokenizer,
        max_len=30
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmotionCaptioningLSTM(
        embed_dim=256,
        hidden_dim=512,
        vocab_size=tokenizer.vocab_size
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    os.makedirs("saved_models", exist_ok=True)  # Ensure save directory exists

    for epoch in range(3):
        model.train()
        total_loss = 0

        for image_feat, emotion_vec, captions in tqdm(dataloader):
            image_feat = image_feat.to(device)
            emotion_vec = emotion_vec.to(device)
            captions = captions.to(device)

            optimizer.zero_grad()

            outputs = model(image_feat, emotion_vec, captions[:, :-1])
            target = captions[:, 1:]

            loss = criterion(outputs.reshape(-1, outputs.size(-1)), target.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

        # Save model after each epoch
        save_path = f"saved_models/emotion_caption_model_epoch{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'tokenizer': tokenizer.name_or_path
        }, save_path)

        print(f"Model saved to: {save_path}")
