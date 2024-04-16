# Read data here
# Extract zip file if folder does not already exist
import os
import zipfile
import json

from PIL import Image
import requests
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import CLIPProcessor, CLIPModel, CLIPConfig, Trainer, TrainingArguments
import torch.nn.functional as F
import torch

### ZIP EXTRACTION ###

folder_path = 'data/CT23_1A_checkworthy_multimodal_english_v2'
zip_file_path = 'data/CT23_1A_checkworthy_multimodal_english_v2.zip'

def zip_extraction(folder_path, zip_file_path):
    print('Zip file extraction started')
    if not os.path.exists(folder_path):
        print('Folder does not exist, extracting zip file')
        os.makedirs(folder_path)

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)
    
    print('Zip file extracted')

zip_extraction(folder_path, zip_file_path)

train_path = folder_path + '/CT23_1A_checkworthy_multimodal_english_train.jsonl'
test_path = folder_path + '/CT23_1A_checkworthy_multimodal_english_test.jsonl'

def split_json(data):
    image_data = {
        'text': data['tweet_text'] + data['ocr_text'],
        'label': data['class_label'],
        'image_path': data['image_path']
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

# Define the fine-tuned CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")

# Freeze the pre-trained model parameters
for param in model.parameters():
    param.requires_grad = False

# Define the fine-tuning head
model.classification_head = torch.nn.Linear(1024, 1)

# Define the optimizer
optimizer = AdamW(model.classification_head.parameters(), lr=1e-3)

# Define the loss function
loss_fn = torch.nn.BCEWithLogitsLoss()

# Define the training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    output_dir='./results_roberta',
    logging_dir='./logs',
    logging_steps=100,
    learning_rate=0.001,  # Set your desired initial learning rate here
)


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")  # Remove "labels" from inputs and store it separately
        outputs = model(**inputs)
        
        # Get logits for each modality
        logits_image = outputs.logits_per_image
        logits_text = outputs.logits_per_text
        
        # Combine logits if necessary
        combined_logits = torch.cat((logits_image, logits_text), dim=1)
        
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
        # Wrap combined_logits with torch.autograd.Variable to allow setting requires_grad
        combined_logits = torch.autograd.Variable(combined_logits, requires_grad=True)
        
        # Ensure labels tensor has the same size as the first dimension of combined_logits
        labels = labels.unsqueeze(1).expand(-1, combined_logits.size(1)).float()
        
        # Set requires_grad to True for labels
        labels.requires_grad = True
        
        # Calculate loss
        loss = loss_fn(combined_logits, labels)
        return loss

# Initialize model and trainer as before

# Ensure model parameters require gradients
for param in model.parameters():
    param.requires_grad = True




def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()