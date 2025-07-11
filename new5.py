import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from transformers import BertTokenizer
import numpy as np

# --- CONFIG ---
EMOTION_MAP = {
    "angry": 0,
    "happy": 1,
    "sad": 2,
    "surprise": 3,
    "neutral": 4,
    "fear": 5
}
MODEL_PATH = "saved_models/emotion_caption_model_epoch3.pt"
MAX_LEN = 30

# --- Preprocess Image ---
def extract_image_feature(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer
    resnet.eval()

    with torch.no_grad():
        feature = resnet(image_tensor).squeeze().numpy()  # (2048,)
    return feature

# --- Emotion to one-hot ---
def get_emotion_vector(emotion_label):
    vector = np.zeros(len(EMOTION_MAP), dtype=np.float32)
    idx = EMOTION_MAP[emotion_label]
    vector[idx] = 1.0
    return vector

# --- Model Class ---
class EmotionCaptioningLSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, emotion_dim=6, image_dim=2048, num_layers=1):
        super(EmotionCaptioningLSTM, self).__init__()
        self.fc_context = nn.Linear(image_dim + emotion_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image_feature, emotion_vector, captions):
        context = torch.cat((image_feature, emotion_vector), dim=1)
        h0 = torch.tanh(self.fc_context(context)).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        embeddings = self.embedding(captions)
        outputs, _ = self.lstm(embeddings, (h0, c0))
        logits = self.fc_out(outputs)
        return logits

# --- Load model and tokenizer once ---
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCaptioningLSTM(embed_dim=256, hidden_dim=512, vocab_size=tokenizer.vocab_size)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# --- Caption Generation Function ---
def generate_caption(image_path, emotion_label):
    # Prepare image feature & emotion vector
    img_feat = torch.tensor(extract_image_feature(image_path), dtype=torch.float32).unsqueeze(0).to(device)
    emo_vec = torch.tensor(get_emotion_vector(emotion_label), dtype=torch.float32).unsqueeze(0).to(device)

    caption = [tokenizer.cls_token_id]

    for _ in range(MAX_LEN):
        input_ids = torch.tensor(caption, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_feat, emo_vec, input_ids)
        next_token = torch.argmax(outputs[0, -1, :]).item()
        if next_token == tokenizer.sep_token_id or next_token == tokenizer.pad_token_id:
            break
        caption.append(next_token)

    return tokenizer.decode(caption, skip_special_tokens=True)

# --- Test it ---
if __name__ == "__main__":
    IMAGE_PATH = "test.jpg"
    EMOTION = "happy"

    result = generate_caption(IMAGE_PATH, EMOTION)
    print("Image:", IMAGE_PATH)
    print("Emotion:", EMOTION)
    print("Generated Caption:", result)
