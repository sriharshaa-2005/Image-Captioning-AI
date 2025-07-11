import pandas as pd
from transformers import pipeline
import json
from tqdm import tqdm

def main():
    csv_file = 'captions.txt'
    print("Loading TXT file as CSV...")
    df = pd.read_csv(csv_file, delimiter=',', names=['image', 'caption'], header=0)

    # Limit to 1000 images
    unique_images = df['image'].unique()[:1000]
    df = df[df['image'].isin(unique_images)]

    # Limit captions per image to 2
    limited_df = df.groupby('image').head(2)

    image_captions = limited_df.groupby('image')['caption'].apply(list).to_dict()

    print("Loading emotion classification model...")
    emotion_classifier = pipeline('text-classification', model='nateraw/bert-base-uncased-emotion', framework="pt")


    def get_emotion_label(caption):
        try:
            result = emotion_classifier(caption)[0]
            return result['label']  # One of: anger, fear, joy, love, sadness, surprise
        except Exception as e:
            print(f"Error classifying caption: {caption}\n{e}")
            return "unknown"

    print("Classifying emotions on captions...")
    image_emotion_captions = {}

    for img, captions in tqdm(image_captions.items(), desc="Processing images"):
        labeled_captions = []
        for caption in captions:
            emotion = get_emotion_label(caption)
            labeled_captions.append({'caption': caption, 'emotion': emotion})
        image_emotion_captions[img] = labeled_captions

    output_file = 'image_emotion_captions_limited.json'
    print(f"Saving labeled captions to {output_file} ...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(image_emotion_captions, f, ensure_ascii=False, indent=4)

    print("Done!")

if __name__ == "__main__":
    main()
