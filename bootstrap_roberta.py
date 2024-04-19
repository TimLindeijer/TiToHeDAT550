import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import json
import numpy as np

# Define a dataset class
class TextDataset(Dataset):
    def __init__(self, encodings_text, labels):
        self.encodings_text = encodings_text
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: torch.clone(val[idx]) for key, val in self.encodings_text.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Compute metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    correct = np.sum(preds == labels)
    total = len(labels)
    return {"accuracy": correct / total}

def split_json(data):
    text_data = {
        'tweet_id': data['tweet_id'],
        'tweet_url': data['tweet_url'],
        'text': data['tweet_text'] + data['ocr_text'],
        'class_label': data['class_label']
    }

    return text_data

### READ AND SPLIT DATA ###

# Read data from the folder
def read_data(file_path):
    print('Reading data')
    text_data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            text = split_json(json_obj)
            text_data.append(text)
    print('Finished reading data')
    return text_data

test_text_data = read_data(train_path)

def main(test_data_path, model_path):
    # Load the test data
    with open(test_data_path, 'r') as file:
        test_data = [json.loads(line) for line in file]

    # Load the tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained(model_path)

    # Process the inputs
    inputs_text_test = tokenizer([data['text'] for data in test_data], return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Create the test dataset
    test_labels = [data['class_label'] for data in test_data]
    test_dataset = TextDataset(inputs_text_test, test_labels)

    # Evaluate the model on test data
    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics
    )

    eval_result = trainer.evaluate(test_dataset)

    # Print the evaluation results
    print(f"Evaluation result: {eval_result}")

if __name__ == "__main__":
    test_data_path = 'data/CT23_1A_checkworthy_multimodal_english_v2/CT23_1A_checkworthy_multimodal_english_test.jsonl'
    model_path = './results_roberta'  # Path to the trained model
    main(test_data_path, model_path)
