# Read data here
# Extract zip file if folder does not already exist
import os
import zipfile
import json

from torch.optim import AdamW
from PIL import Image
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import CLIPProcessor, CLIPModel, RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset
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

# Prepare the data
labels = [data['class_label'] for data in train_text_data]  # You need to have labels for your training data

images = []
for data in train_image_data:
    with Image.open(folder_path + '/' + data['image_path']) as img:
        images.append(img.copy())

# Split the data into training and validation sets
train_labels, val_labels, train_images, val_images = train_test_split(
    labels, images, test_size=0.2, random_state=42
)

# Convert labels to numerical values
le = LabelEncoder()
train_labels_num = le.fit_transform(train_labels)
val_labels_num = le.transform(val_labels)

# CLIP ViT model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Process the inputs
inputs_image_train = clip_processor(images=[img.convert("RGBA") for img in train_images], return_tensors="pt")

inputs_image_val = clip_processor(images=[img.convert("RGBA") for img in val_images], return_tensors="pt")

# Define the optimizers
optimizer_clip = AdamW(clip_model.parameters(), lr=1e-5)

# Compute metrics function for evaluation
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

training_args_clip = TrainingArguments(
    output_dir='./results_clip',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=1000,
    evaluation_strategy="epoch",  # Evaluate at the end of every epoch
)

# Create a Trainer for CLIP
trainer_clip = Trainer(
    model=clip_model,
    args=training_args_clip,
    train_dataset=inputs_image_train,
    eval_dataset=inputs_image_val,
    compute_metrics=compute_metrics,
)


# Train the models
trainer_clip.train()

# Evaluate the models
eval_result_clip = trainer_clip.evaluate()

# Print the evaluation results
print(f"Eval result for CLIP: {eval_result_clip}")