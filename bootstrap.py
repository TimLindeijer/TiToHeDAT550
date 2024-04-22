# bootstrap_test.py
import json
import random
import os
from PIL import Image
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import CLIPProcessor, CLIPModel, CLIPConfig

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
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            image = split_json(json_obj)
            image_data.append(image)
    print('Finished reading data')
    return image_data

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

def bootstrap_test(test_path, model_path, folder_path):
    print("Starting bootstrap test")
    # CLIP ViT model

    # Load the fine-tuned model
    fine_tuned_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    state_dict = torch.load(model_path)
    fine_tuned_model.load_state_dict(state_dict, strict=False)  # Set strict=False to ignore unexpected keys

    # Recreate the classification head used during fine-tuning
    fine_tuned_model.classification_head = torch.nn.Linear(1024, 1)
    fine_tuned_model.eval()

    # Read test data
    test_image_data = read_data(test_path)

    # Convert labels to numerical values for test data
    le = LabelEncoder()
    test_labels = [data['label'] for data in test_image_data]
    test_numerical_labels = le.fit_transform(test_labels)

    # Replace the class_label in each data dictionary with its numerical equivalent
    for data, num_label in zip(test_image_data, test_numerical_labels):
        data['label'] = num_label

    # Create a DataLoader for the test data
    test_dataset = CLIPDataset(test_image_data, folder_path)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    # Bootstrap test
    n_bootstrap_samples = 100
    bootstrap_accuracies = []
    bootstrap_precisions = []
    bootstrap_recalls = []
    bootstrap_f1_scores = []
    print("Running bootstrap test")
    wandb.init(project="dat550-multimodal", name="clip-only-90epochs-bootstrap-wo-conf")
    for _ in range(n_bootstrap_samples):
        # Sample with replacement from the test set
        bootstrap_sample = [random.choice(test_image_data) for _ in range(len(test_image_data))]
        bootstrap_dataset = CLIPDataset(bootstrap_sample, folder_path)
        bootstrap_loader = torch.utils.data.DataLoader(bootstrap_dataset, batch_size=32)

        # Evaluate the model on the bootstrap sample
        total_correct = 0
        total_count = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in bootstrap_loader:
                outputs = fine_tuned_model(pixel_values=batch["pixel_values"], input_ids=batch["input_ids"])
                combined_embeds = torch.cat((outputs.image_embeds, outputs.text_embeds), dim=1)
                logits = fine_tuned_model.classification_head(combined_embeds)
            
                preds = torch.sigmoid(logits.view(-1)) > 0.5  # Get binary predictions
                total_correct += (preds == batch["label"]).sum().item()
                total_count += preds.size(0)
                all_labels.extend(batch["label"].numpy())
                all_preds.extend(preds.numpy())

        # Compute the accuracy and add it to the list of bootstrap accuracies
        bootstrap_accuracy = total_correct / total_count
        print(f"Accuracy: {bootstrap_accuracy}")
        wandb.log({"Bootstrap Accuracy": bootstrap_accuracy})
        bootstrap_accuracies.append(bootstrap_accuracy)
        
        # Convert lists to numpy arrays
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        # Compute precision, recall, and F1 score
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        bootstrap_precisions.append(precision)
        bootstrap_recalls.append(recall)
        bootstrap_f1_scores.append(f1)
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

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
    wandb.finish()
if __name__ == "__main__":
    folder_path = 'data/CT23_1A_checkworthy_multimodal_english_v2'
    test_path = folder_path + '/CT23_1A_checkworthy_multimodal_english_dev_test.jsonl'
    model_path = "clip_only_90_epochs_kfold_wo_conf/fine_tuned_model_epoch.pth"
    bootstrap_test(test_path, model_path, folder_path)