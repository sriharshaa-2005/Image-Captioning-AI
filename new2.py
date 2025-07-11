import json
import torch
from transformers import BertTokenizer
from tqdm import tqdm

# Config 
emotion_classes = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
caption_emotion_json = "image_emotion_captions_limited.json"
image_feature_json = "image_features_1000.json"
output_final_dataset = "final_preprocessed_dataset.json"
max_caption_length = 30

# Load input files
with open(caption_emotion_json, 'r', encoding='utf-8') as f:
    caption_data = json.load(f)

with open(image_feature_json, 'r') as f:
    image_features = json.load(f)

# Load tokenizer 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Helper Functions
def one_hot_encode_emotion(emotion):
    vec = [0] * len(emotion_classes)
    if emotion in emotion_classes:
        vec[emotion_classes.index(emotion)] = 1
    return vec

def tokenize_caption(caption):
    tokens = tokenizer(
        caption,
        padding='max_length',
        truncation=True,
        max_length=max_caption_length,
        return_tensors='pt'
    )
    return tokens['input_ids'].squeeze().tolist(), tokens['attention_mask'].squeeze().tolist()

# Build Final Dataset 
final_data = []

print("Building final dataset...")
for image_name, captions in tqdm(caption_data.items()):
    if image_name not in image_features:
        print(f"Skipping missing image feature: {image_name}")
        continue

    for item in captions:
        caption = item['caption']
        emotion = item['emotion']
        
        input_ids, attention_mask = tokenize_caption(caption)
        emotion_vector = one_hot_encode_emotion(emotion)
        feature_vector = image_features[image_name]

        final_data.append({
            "image": image_name,
            "caption": caption,
            "emotion": emotion,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "emotion_vector": emotion_vector,
            "image_feature": feature_vector
        })

# Save
with open(output_final_dataset, 'w') as f:
    json.dump(final_data, f, indent=4)

print(f"Done! Final preprocessed dataset saved to '{output_final_dataset}'")
