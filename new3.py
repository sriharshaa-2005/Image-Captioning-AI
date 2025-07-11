import torch
from torch.utils.data import Dataset
import json
from transformers import BertTokenizer

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

        # Load precomputed 2048-d image features
        image_feature = torch.tensor(item["image_feature"], dtype=torch.float32)

        # 6-d emotion one-hot vector
        emotion_vector = torch.tensor(item["emotion_vector"], dtype=torch.float32)

        # Padded caption token IDs
        caption_ids = item["input_ids"][:self.max_len]
        caption_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(caption_ids))
        caption_ids = torch.tensor(caption_ids, dtype=torch.long)

        return image_feature, emotion_vector, caption_ids

# TESTING THE DATASET
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = EmotionCaptionDataset("final_preprocessed_dataset.json", tokenizer)

    print(f"Total samples: {len(dataset)}")
    image_feat, emotion_vec, cap_ids = dataset[0]

    print("\nSample 1:")
    print("Image Feature Shape:", image_feat.shape)       # Should be torch.Size([2048])
    print("Emotion Vector:", emotion_vec.tolist())         # One-hot 6-d
    print("Caption Token IDs:", cap_ids[:10].tolist(), "...")  # First 10 tokens
