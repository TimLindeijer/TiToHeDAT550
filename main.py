# Read data here
# Extract zip file if folder does not already exist
import os
import zipfile
import json
from torch.optim import AdamW
from PIL import Image
import torch
from sklearn.preprocessing import LabelEncoder

from transformers import CLIPProcessor, CLIPModel, RobertaTokenizer, RobertaForSequenceClassification

### ZIP EXCTRACTION ###

folder_path = 'data/CT23_1A_checkworthy_multimodal_english_v2'
zip_file_path = 'data/CT23_1A_checkworthy_multimodal_english_v2.zip'

def zip_extration(folder_path, zip_file_path):
    print('Zip file extraction started')
    if not os.path.exists(folder_path):
        print('Folder does not exist, extracting zip file')
        os.makedirs(folder_path)

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)
    
    print('Zip file extracted')

zip_extration(folder_path, zip_file_path)

train_path = folder_path + '/CT23_1A_checkworthy_multimodal_english_train.jsonl'
test_path = folder_path + '/CT23_1A_checkworthy_multimodal_english_test.jsonl'

def split_json(data):
    text_data = {
        'tweet_id': data['tweet_id'],
        'tweet_url': data['tweet_url'],
        'text': data['tweet_text'] + data['ocr_text'],
        'class_label': data['class_label']
    }

    image_data = {
        'tweet_id': data['tweet_id'],
        'tweet_url': data['tweet_url'],
        'class_label': data['class_label'],
        'image_path': data['image_path'],
        'image_url': data['image_url']
    }

    return text_data, image_data

### READ AND SPLIT DATA ###

# Read data from the folder
def read_data(file_path):
    print('Reading data')
    text_data = []
    image_data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            text, image = split_json(json_obj)
            text_data.append(text)
            image_data.append(image)
    print('Finished reading data')
    return text_data, image_data

train_text_data, train_image_data = read_data(train_path)
# print(f'Text: {train_text_data[0]}')
# print(f'Image: {train_image_data[0]}')

### PREPARE DATA ###
print('Preparing data')
texts = [data['text'] for data in train_text_data]
labels = [data['class_label'] for data in train_text_data]  # You need to have labels for your training data

# images = [Image.open(folder_path + '/' + data['image_path']) for data in train_image_data]
images = []
for data in train_image_data:
    with Image.open(folder_path + '/' + data['image_path']) as img:
        images.append(img.copy())


# Convert labels to numerical values
le = LabelEncoder()
labels_num = le.fit_transform(labels)

print('Setting up models')

# CLIP ViT model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# RoBERTa model
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(labels))


inputs_text = roberta_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
inputs_text["labels"] = torch.tensor(labels_num)

inputs_image = clip_processor(images=images, return_tensors="pt", padding=True, truncation=True)
inputs_image["labels"] = torch.tensor(labels_num)
# Define the optimizers
optimizer_roberta = AdamW(roberta_model.parameters(), lr=1e-5)
optimizer_clip = AdamW(clip_model.parameters(), lr=1e-5)

print('Starting Training')

# Train the models
roberta_model.train()
clip_model.train()
for epoch in range(10):  # Number of epochs is a hyperparameter you can tune
    outputs_roberta = roberta_model(**inputs_text)
    loss_roberta = outputs_roberta.loss
    loss_roberta.backward()
    optimizer_roberta.step()
    optimizer_roberta.zero_grad()

    outputs_clip = clip_model(**inputs_image)
    loss_clip = outputs_clip.loss
    loss_clip.backward()
    optimizer_clip.step()
    optimizer_clip.zero_grad()
    print(f"Epoch {epoch + 1}: RoBERTa loss = {loss_roberta.item()}, CLIP loss = {loss_clip.item()}")
