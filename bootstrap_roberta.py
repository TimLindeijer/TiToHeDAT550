import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score



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
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            text = split_json(json_obj)
            text_data.append(text)
    print('Finished reading data')
    return text_data


def main(test_data_path, model_path):

    test_text_data = read_data(test_data_path)

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
    bootstrap_precisions = []
    bootstrap_recalls = []
    bootstrap_f1_scores = []
    print("Running bootstrap test")
    wandb.init(project="dat550-multimodal", name="roberta-only-10epochs-bootstrap-100")
    for _ in range(n_bootstrap_samples):
        # Sample with replacement from the test set
        bootstrap_sample = [random.choice(test_text_data) for _ in range(len(test_text_data))]

        # Process the inputs
        inputs_text_bootstrap = tokenizer([data['text'] for data in bootstrap_sample], return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Create the bootstrap dataset
        bootstrap_labels = [data['class_label'] for data in bootstrap_sample]
        bootstrap_labels_num = le.transform(bootstrap_labels)  # Use transform instead of fit_transform to keep the same encoding
        bootstrap_labels_num = bootstrap_labels_num.astype(np.int64)
        bootstrap_dataset = TextDataset(inputs_text_bootstrap, bootstrap_labels_num)

        # Evaluate the model on the bootstrap sample
        eval_result = trainer.evaluate(bootstrap_dataset)

        # Predict the labels for the bootstrap sample
        predictions, _, _ = trainer.predict(bootstrap_dataset)

        # Add the labels to all_labels
        all_labels = np.concatenate((all_labels, bootstrap_labels_num))

        # Convert the predictions to a numpy array and add them to all_preds
        all_preds = np.concatenate((all_preds, np.argmax(predictions, axis=-1)))

        # Add the accuracy to the list of bootstrap accuracies
        bootstrap_accuracies.append(eval_result['eval_accuracy'])
        # Convert lists to numpy arrays
        all_labels = torch.Tensor(all_labels).long()
        all_preds = np.array(all_preds)

        # Compute precision, recall, and F1 score
        precision = precision_score(all_labels.numpy(), all_preds, zero_division=0)
        recall = recall_score(all_labels.numpy(), all_preds, zero_division=0)
        f1 = f1_score(all_labels.numpy(), all_preds, zero_division=0)

        print(f"Accuracy: {eval_result['eval_accuracy']}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 score: {f1}")

        bootstrap_precisions.append(precision)
        bootstrap_recalls.append(recall)
        bootstrap_f1_scores.append(f1)

        # Log the accuracy to wandb
        wandb.log({"Bootstrap Accuracy": eval_result['eval_accuracy']})
        # Log precision, recall, and F1 score to wandb
        wandb.log({"Precision": precision, "Recall": recall, "F1 Score": f1})

    # Compute the mean and standard deviation of the bootstrap accuracies
    mean_accuracy = np.mean(bootstrap_accuracies)
    std_accuracy = np.std(bootstrap_accuracies)

    mean_precision = np.mean(bootstrap_precisions)
    std_precision = np.std(bootstrap_precisions)

    mean_recall = np.mean(bootstrap_recalls)
    std_recall = np.std(bootstrap_recalls)

    mean_f1 = np.mean(bootstrap_f1_scores)
    std_f1 = np.std(bootstrap_f1_scores)

    print(f"Mean accuracy: {mean_accuracy}")
    print(f"Standard deviation of accuracy: {std_accuracy}")

    print(f"Mean precision: {mean_precision}")
    print(f"Standard deviation of precision: {std_precision}")

    print(f"Mean recall: {mean_recall}")
    print(f"Standard deviation of recall: {std_recall}")

    print(f"Mean F1 score: {mean_f1}")
    print(f"Standard deviation of F1 score: {std_f1}")

if __name__ == "__main__":
    folder_path = 'data/CT23_1A_checkworthy_multimodal_english_v2'
    test_path = folder_path + '/CT23_1A_checkworthy_multimodal_english_dev_test.jsonl'
    model_path = './results_roberta/checkpoint-10500'  # Path to the trained model
    main(test_path, model_path)
