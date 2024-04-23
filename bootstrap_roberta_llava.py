import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd



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

def read_descriptions(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    # Remove .jpg to get the tweet id
    df["filename"] = df["filename"].str.replace(".jpg", "")
    # Convert the DataFrame to a dictionary where keys are image IDs and values are descriptions
    descriptions_dict = df.set_index('filename')['description'].to_dict()
    return descriptions_dict

def split_json(data, descriptions_dict):
    # Get the image ID from the image path and remove the '.jpg' extension
    description = descriptions_dict.get(data['tweet_id'], '')  # Empty string if image ID not found
    text_data = {
        'tweet_id': data['tweet_id'],
        'tweet_url': data['tweet_url'],
        'text': data['tweet_text'] + data['ocr_text'] + description,
        'class_label': data['class_label']
    }

    return text_data

### READ AND SPLIT DATA ###

# Read data from the folder
def read_data(file_path, csv_file):
    print('Reading data')
    text_data = []
    descriptions_dict = read_descriptions(csv_file)
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            text = split_json(json_obj, descriptions_dict)
            text_data.append(text)
    print('Finished reading data')
    return text_data


def main(test_data_path, model_path, csv_file):

    test_text_data = read_data(test_data_path, csv_file)
    # Load the tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained(model_path)

    # Process the inputs
    inputs_text_test = tokenizer([data['text'] for data in test_text_data], return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Create the test dataset
    test_labels = [data['class_label'] for data in test_text_data]
    le = LabelEncoder()
    test_labels_num = le.fit_transform(test_labels)
    test_dataset = TextDataset(inputs_text_test, test_labels_num)

    # Evaluate the model on test data
    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics
    )

    n_bootstrap_samples = 100
    all_labels = np.array([])
    all_preds = np.array([])
    bootstrap_accuracies = []
    # Initialize lists to store precision, recall, and F1 scores for each bootstrap sample
    bootstrap_precisions = []
    bootstrap_recalls = []
    bootstrap_f1_scores = []
    wandb.init(project="dat550-multimodal", name="roberta-llava-bootstrap-100")
    for _ in range(n_bootstrap_samples):
        # Sample with replacement from the test set using stratified sampling
        bootstrap_sample = []
        for class_label in np.unique(test_labels_num):
            class_indices = np.where(test_labels_num == class_label)[0]
            bootstrap_indices = np.random.choice(class_indices, size=len(class_indices), replace=True)
            bootstrap_sample.extend([test_text_data[i] for i in bootstrap_indices])

        # Process the inputs
        inputs_text_bootstrap = tokenizer([data['text'] for data in bootstrap_sample], return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Create the bootstrap dataset
        bootstrap_labels = [data['class_label'] for data in bootstrap_sample]
        bootstrap_labels_num = le.transform(bootstrap_labels)
        bootstrap_labels_num = bootstrap_labels_num.astype(np.int64)
        bootstrap_dataset = TextDataset(inputs_text_bootstrap, bootstrap_labels_num)

        # Evaluate the model on the bootstrap sample
        eval_result = trainer.evaluate(bootstrap_dataset)

        # Predict the labels for the bootstrap sample
        predictions, _, _ = trainer.predict(bootstrap_dataset)

        # Convert predictions to class labels
        predicted_labels = np.argmax(predictions, axis=-1)

        # Compute precision, recall, and F1 score for this bootstrap sample
        precision = precision_score(bootstrap_labels_num, predicted_labels, average='weighted')
        recall = recall_score(bootstrap_labels_num, predicted_labels, average='weighted')
        f1 = f1_score(bootstrap_labels_num, predicted_labels, average='weighted')
        # Log the accuracy to wandb
        wandb.log({"Bootstrap Accuracy": eval_result['eval_accuracy']})
        # Log precision, recall, and F1 score to wandb
        wandb.log({"Precision": precision, "Recall": recall, "F1 Score": f1})

        # Print and log the evaluation metrics
        print(f"Accuracy: {eval_result['eval_accuracy']}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 score: {f1}")

        # Append the metrics to the lists
        bootstrap_accuracies.append(eval_result['eval_accuracy'])
        bootstrap_precisions.append(precision)
        bootstrap_recalls.append(recall)
        bootstrap_f1_scores.append(f1)


    # Compute the mean and standard deviation of the evaluation metrics
    mean_accuracy = np.mean(bootstrap_accuracies)
    mean_precision = np.mean(bootstrap_precisions)
    mean_recall = np.mean(bootstrap_recalls)
    mean_f1_score = np.mean(bootstrap_f1_scores)
    std_accuracy = np.std(bootstrap_accuracies)
    std_precision = np.std(bootstrap_precisions)
    std_recall = np.std(bootstrap_recalls)
    std_f1_score = np.std(bootstrap_f1_scores)

    print(f"Mean Accuracy: {mean_accuracy}, Std Accuracy: {std_accuracy}")
    print(f"Mean Precision: {mean_precision}, Std Precision: {std_precision}")
    print(f"Mean Recall: {mean_recall}, Std Recall: {std_recall}")
    print(f"Mean F1 Score: {mean_f1_score}, Std F1 Score: {std_f1_score}")
    wandb.finish()

if __name__ == "__main__":
    folder_path = 'data/CT23_1A_checkworthy_multimodal_english_v2'
    test_path = folder_path + '/CT23_1A_checkworthy_multimodal_english_dev_test.jsonl'
    model_path = './results_roberta/checkpoint-10500'  # Path to the trained model
    csv_file = 'TiToHe_descriptions_rest.csv'
    main(test_path, model_path, csv_file)
