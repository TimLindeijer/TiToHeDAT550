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
texts = [data['text'] for data in train_text_data]
labels = [data['class_label'] for data in train_text_data]  # You need to have labels for your training data

# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Convert labels to numerical values
le = LabelEncoder()
train_labels_num = le.fit_transform(train_labels)
val_labels_num = le.transform(val_labels)

# RoBERTa model
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(labels))

# Process the inputs
inputs_text_train = roberta_tokenizer(train_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

inputs_text_val = roberta_tokenizer(val_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Define the optimizers
optimizer_roberta = AdamW(roberta_model.parameters(), lr=1e-5)

# Compute metrics function for evaluation
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Define training arguments
training_args_roberta = TrainingArguments(
    output_dir='./results_roberta',
    num_train_epochs=90,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=1000,
    evaluation_strategy="epoch",  # Evaluate at the end of every epoch
)

# Define a dataset
class TextImageDataset(Dataset):
    def __init__(self, encodings_text, labels):
        self.encodings_text = encodings_text
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: torch.clone(val[idx]) for key, val in self.encodings_text.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextImageDataset(inputs_text_train, train_labels_num)
val_dataset = TextImageDataset(inputs_text_val, val_labels_num)


# Create a Trainer for RoBERTa
trainer_roberta = Trainer(
    model=roberta_model,
    args=training_args_roberta,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)


# Train the models
trainer_roberta.train()

# Evaluate the models
eval_result_roberta = trainer_roberta.evaluate()

# Print the evaluation results
print(f"Eval result for RoBERTa: {eval_result_roberta}")