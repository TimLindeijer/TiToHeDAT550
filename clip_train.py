# Read data here
# Extract zip file if folder does not already exist
import os
import zipfile
import json

from PIL import Image
import requests
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import CLIPProcessor, CLIPModel, CLIPConfig
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW


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
    image_data = {
        'text': data['tweet_text'] + data['ocr_text'],
        'label': data['class_label'],
        'image_path': data['image_path']
        # 'url': data['image_url']
    }

    return image_data

### READ AND SPLIT DATA ###

# Read data from the folder
def read_data(file_path):
    print('Reading data')
    image_data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            image = split_json(json_obj)
            image_data.append(image)
    print('Finished reading data')
    return image_data

train_image_data = read_data(train_path)

# Convert labels to numerical values
le = LabelEncoder()

# Fit the encoder on the class labels and transform them
labels = [data['label'] for data in train_image_data]
le.fit(labels)
numerical_labels = le.transform(labels)

# Replace the class_label in each data dictionary with its numerical equivalent
for data, num_label in zip(train_image_data, numerical_labels):
    data['label'] = num_label
# print(f'Text: {train_text_data[0]}')
# print(f'Image: {train_image_data[0]}')

### PREPARE DATA ###
print('Preparing data')

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, data, folder_path):
        self.data = data
        self.folder_path = folder_path
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.folder_path, item["image_path"])
        try:
            image = Image.open(image_path).convert('RGBA')
        except Exception as e:
            print(f"Error loading image at path: {image_path}")
            print(e)
            # Return None or a placeholder image
            image = Image.new('RGBA', (224, 224))  # Placeholder image
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        input_ids = self.processor(text=item["text"], return_tensors="pt", padding="max_length", max_length=77, truncation=True).input_ids
        label = torch.tensor(item["label"])
        return {
            "pixel_values": pixel_values.squeeze(),
            "input_ids": input_ids.squeeze(),
            "label": label
        }




# Split the data into training and validation sets
train_data, val_data = train_test_split(
    train_image_data, test_size=0.2, random_state=42
)

train_dataset = CLIPDataset(train_data, folder_path)
val_dataset = CLIPDataset(val_data, folder_path)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

# CLIP ViT model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Define the fine-tuning configuration
config = CLIPConfig(
    text_config=model.text_model.config.to_dict(),
    vision_config=model.vision_model.config.to_dict(),
    projection_dim=512,
    logit_scale_init_value=2.6592,
)

# Instantiate the fine-tuned CLIP model
fine_tuned_model = CLIPModel(config)

# Freeze the pre-trained model parameters
for param in fine_tuned_model.parameters():
    param.requires_grad = False

# Define the fine-tuning head
# fine_tuned_model.classification_head = torch.nn.Linear(config.projection_dim, 2)
fine_tuned_model.classification_head = torch.nn.Linear(1024, 1)
# Define the optimizer and training loop
optimizer = AdamW(fine_tuned_model.classification_head.parameters(), lr=5e-5)
loss_fn = torch.nn.BCEWithLogitsLoss()

for epoch in range(10):
    fine_tuned_model.train()
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = fine_tuned_model(pixel_values=batch["pixel_values"], input_ids=batch["input_ids"])
        combined_embeds = torch.cat((outputs.image_embeds, outputs.text_embeds), dim=1)
        logits = fine_tuned_model.classification_head(combined_embeds)
        
        loss = loss_fn(logits.view(-1), batch["label"].float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Print average training loss per epoch
    print(f"Epoch {epoch+1}, Loss: {train_loss / len(train_loader)}")

    # Evaluation phase
    fine_tuned_model.eval()
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = fine_tuned_model(pixel_values=batch["pixel_values"], input_ids=batch["input_ids"])
            combined_embeds = torch.cat((outputs.image_embeds, outputs.text_embeds), dim=1)
            logits = fine_tuned_model.classification_head(combined_embeds)
        
            preds = torch.sigmoid(logits.view(-1)) > 0.5  # Get binary predictions
            total_correct += (preds == batch["label"]).sum().item()
            total_count += preds.size(0)
    
    # Print validation accuracy
    print(f"Validation Accuracy: {total_correct / total_count}")