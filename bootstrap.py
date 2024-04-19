# bootstrap_test.py
import json
import random
import os
from PIL import Image

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
    with open(file_path, 'r') as file:
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
    # CLIP ViT model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    # Define the fine-tuning configuration
    config = CLIPConfig(
        text_config=model.text_model.config.to_dict(),
        vision_config=model.vision_model.config.to_dict(),
        projection_dim=512,
        logit_scale_init_value=2.6592,
    )

    # Load the model
    fine_tuned_model = CLIPModel(config)
    fine_tuned_model.load_state_dict(torch.load(model_path))
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
    n_bootstrap_samples = 1000
    bootstrap_accuracies = []

    for _ in range(n_bootstrap_samples):
        # Sample with replacement from the test set
        bootstrap_sample = [random.choice(test_image_data) for _ in range(len(test_image_data))]
        bootstrap_dataset = CLIPDataset(bootstrap_sample, folder_path)
        bootstrap_loader = torch.utils.data.DataLoader(bootstrap_dataset, batch_size=32)

        # Evaluate the model on the bootstrap sample
        total_correct = 0
        total_count = 0
        with torch.no_grad():
            for batch in bootstrap_loader:
                outputs = fine_tuned_model(pixel_values=batch["pixel_values"], input_ids=batch["input_ids"])
                combined_embeds = torch.cat((outputs.image_embeds, outputs.text_embeds), dim=1)
                logits = fine_tuned_model.classification_head(combined_embeds)
            
                preds = torch.sigmoid(logits.view(-1)) > 0.5  # Get binary predictions
                total_correct += (preds == batch["label"]).sum().item()
                total_count += preds.size(0)

        # Compute the accuracy and add it to the list of bootstrap accuracies
        bootstrap_accuracy = total_correct / total_count
        bootstrap_accuracies.append(bootstrap_accuracy)

    # Compute the mean and standard deviation of the bootstrap accuracies
    mean_accuracy = np.mean(bootstrap_accuracies)
    std_accuracy = np.std(bootstrap_accuracies)

    print(f"Mean accuracy: {mean_accuracy}")
    print(f"Standard deviation of accuracy: {std_accuracy}")

if __name__ == "__main__":
    folder_path = 'data/CT23_1A_checkworthy_multimodal_english_v2'
    test_path = folder_path + '/CT23_1A_checkworthy_multimodal_english_test.jsonl'
    model_path = "clip_only_90_epochs_kfold/fine_tuned_model_epoch.pth"
    folder_path = "path_to_your_folder"
    bootstrap_test(test_path, model_path, folder_path)