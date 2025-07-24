
## III. BONUS: Use the VideoMAE model to predict the outcome of the procedure from the SOCAL dataset

You can find the column "success" in the file socal_trial_outcomes.csv. Create an ML dataset ready for training the videomae model and run the training + Evaluation.

You can find the full SOCAL dataset here: [https://drive.google.com/drive/folders/1xGcGkbj34wgETuzSafa5hZw5WAdRdtG4?usp=sharing](https://www.google.com/url?q=https%3A%2F%2Fdrive.google.com%2Fdrive%2Ffolders%2F1xGcGkbj34wgETuzSafa5hZw5WAdRdtG4%3Fusp%3Dsharing)

Some code to help get started with the training can be found on huggingface page: [https://huggingface.co/docs/transformers/v4.34.1/en/model_doc/videomae#transformers.VideoMAEForVideoClassification](https://www.google.com/url?q=https%3A%2F%2Fhuggingface.co%2Fdocs%2Ftransformers%2Fv4.34.1%2Fen%2Fmodel_doc%2Fvideomae%23transformers.VideoMAEForVideoClassification)

This notebook is also useful: [https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/video_classification.ipynb](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/video_classification.ipynb)

```python
# VideoMAE model training datasets preparation import
from google.colab import drive
import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import torchvision.transforms as transforms
from transformers import AutoImageProcessor
from PIL import Image
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
!pip install -q gdown
```


```python
# Data folder and validate
DRIVE_BASE_PATH = "/content/drive/My Drive/Colab Notebooks"
SOCAL_DATA_FOLDER = os.path.join(DRIVE_BASE_PATH, "SOCAL")
JPEG_IMAGES_SUBFOLDER = os.path.join(SOCAL_DATA_FOLDER, "JPEGImages")
NEW_DATASET_FOLDER = os.path.join(DRIVE_BASE_PATH, "SOCAL_dataset")
OUTCOMES_CSV_PATH = os.path.join(SOCAL_DATA_FOLDER, "socal_trial_outcomes.csv")
IMAGE_FILE_EXTENSION = ".jpeg"
DATA_LINK = "https://drive.google.com/drive/folders/1xGcGkbj34wgETuzSafa5hZw5WAdRdtG4?usp=sharing"


# Model Parameters for VideoMAE model
MODEL_NAME = "MCG-NJU/videomae-base"
NUM_FRAMES = 16
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


config = {
  # Optimizer and training options
  "optimizer_choice": "AdamW", # option: AdamW, SGD
  "learning_rate": 5e-5,
  "batch_size": 4,
  "num_epochs": 30, # Num of epochs
  "weight_decay": 0.01, # L2 regulation weight Decay (0.01-0.1)
  "force_retrain": False,

  # Dropout options
  "dropout_rate": 0.2, # Dropout (0.2-0.5)

  # Early Stopping options
  "use_early_stopping": True,
  "early_stopping_patience": 3, # Stopping epoch

  # LR Scheduler
  "use_lr_scheduler": True,
  "lr_scheduler_patience": 1,
}
```


```python
# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Selected device: {device}")
if torch.cuda.is_available():
  print(f"GPU Name: {torch.cuda.get_device_name(0)}")
  scaler = torch.amp.GradScaler('cuda')
else:
  scaler = None


# Mount Google Drive
try:
  drive.mount('/content/drive', force_remount=True)
  print("Google Drive mounted successfully!")
except Exception as e:
  print(f"Error mounting Google Drive: {e}")
  raise SystemExit("Halting execution: Google Drive mount failed.")


# Data folder and validate
try:
  if os.path.isdir(SOCAL_DATA_FOLDER) and os.path.exists(OUTCOMES_CSV_PATH):
    print("Dataset folder found.")
  else:
    print("Dataset folder not found. download dataset from the link")
    gdown.download_folder(DATA_LINK, output=DRIVE_BASE_PATH, quiet=False)

    # Check data folder
    if os.path.isdir(SOCAL_DATA_FOLDER):
      print("Download complete! Dataset is ready.")
    else:
      raise FileNotFoundError("Download failed.")

except (FileNotFoundError, ValueError, Exception) as e:
    print(f"Setup Error: {e}")
    raise SystemExit("Halting execution: Dataset folder failed.")



# Data folder check new dataset folder
try:
  if not os.path.isdir(JPEG_IMAGES_SUBFOLDER):
      raise FileNotFoundError(f"The 'JPEGImages' folder was not found at the expected path: {JPEG_IMAGES_SUBFOLDER}")

  os.makedirs(NEW_DATASET_FOLDER, exist_ok=True)
  print(f"Directory for processed datasets is ready: {NEW_DATASET_FOLDER}")

  print("Scanning for image files... (this may take a moment)")
  search_pattern = os.path.join(JPEG_IMAGES_SUBFOLDER, '*.jpeg')
  image_files = glob.glob(search_pattern)

  if not image_files:
      raise ValueError("No image files found in the 'SOCAL/JPEGImages' folder.")
  print(f"Successfully found {len(image_files)} image files.")

except (FileNotFoundError, ValueError) as e:
  print(f"Path or File Error: {e}")
  print("Please ensure your folder structure and files are correct in Google Drive.")
  raise SystemExit("Halting execution: Data validation failed.")
```


```python
class SOCALImageSequenceDataset(Dataset):
  """
  Custom PyTorch Dataset class for loading SOCAL image sequence data.
  """
  def __init__(self, dataframe, image_dir, image_processor, num_frames=16, transform=None):
    self.dataframe = dataframe
    self.image_dir = image_dir
    self.image_processor = image_processor
    self.num_frames = num_frames
    self.transform = transform
    self.trial_frames_map = self._build_trial_frames_map()

    initial_len = len(self.dataframe)
    self.dataframe = self.dataframe[self.dataframe['trial_id'].isin(self.trial_frames_map.keys())].reset_index(drop=True)
    if len(self.dataframe) < initial_len:
      print(f"Warning: Filtered out {initial_len - len(self.dataframe)} trials with no corresponding image frames found.")
    print(f"Dataset initialized with {len(self.dataframe)} valid samples.")

  def _build_trial_frames_map(self):
    trial_map = {}
    all_image_paths = glob.glob(os.path.join(self.image_dir, f"*{IMAGE_FILE_EXTENSION}"))
    for img_path in tqdm(all_image_paths, desc="Scanning image files..."):
      filename = os.path.basename(img_path)
      trial_id = filename.split('_')[0]
      if trial_id not in trial_map:
          trial_map[trial_id] = []
      trial_map[trial_id].append(img_path)
    for trial_id in trial_map:
      trial_map[trial_id].sort()
    return trial_map

  def __len__(self):
    return len(self.dataframe)

  def __getitem__(self, idx):
    row = self.dataframe.iloc[idx]
    trial_id = row['trial_id']
    label = row['success']
    frame_paths = self.trial_frames_map.get(trial_id, [])

    final_height = self.image_processor.crop_size['height']
    final_width = self.image_processor.crop_size['width']
    image_size_tuple = (final_width, final_height)

    if not frame_paths:
      return {
          'pixel_values': torch.zeros(3, self.num_frames, final_height, final_width),
          'labels': torch.tensor(-1)
      }

    total_frames_in_trial = len(frame_paths)
    if total_frames_in_trial < self.num_frames:
      indices = np.random.choice(total_frames_in_trial, self.num_frames, replace=True)
    else:
      indices = np.linspace(0, total_frames_in_trial - 1, self.num_frames, dtype=int)
    indices.sort()

    selected_frames = []
    for i in indices:
      try:
        img = Image.open(frame_paths[i]).convert("RGB")
        if self.transform:
          img = self.transform(img)
        selected_frames.append(img)
      except Exception:
        selected_frames.append(Image.new('RGB', image_size_tuple, (0, 0, 0)))

    inputs = self.image_processor(selected_frames, return_tensors="pt")
    return {
        'pixel_values': inputs['pixel_values'].squeeze(0),
        'labels': torch.tensor(label, dtype=torch.long)
    }


# Data augmentation setting
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
])
val_test_transforms = None


# Main Data Preparation Workflow
print(f"\n--- Preparing Dataset for {MODEL_NAME} ---")
print("Loading labels from CSV...")
df = pd.read_csv(OUTCOMES_CSV_PATH)

print("\nAnalyzing class distribution...")
print(df['success'].value_counts())
print("\nLoading Image Processor...")
image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

print("Instantiating PyTorch Dataset object...")
image_data_dir = os.path.join(SOCAL_DATA_FOLDER, JPEG_IMAGES_SUBFOLDER)
full_dataset = SOCALImageSequenceDataset(df, image_data_dir, image_processor, NUM_FRAMES)
full_dataset_train_version = SOCALImageSequenceDataset(df, image_data_dir, image_processor, NUM_FRAMES, transform=train_transforms)
full_dataset_eval_version = SOCALImageSequenceDataset(df, image_data_dir, image_processor, NUM_FRAMES, transform=val_test_transforms)

if len(full_dataset) == 0:
    raise RuntimeError("Error: No valid samples found in the dataset. Halting.")

print("\nSplitting data into training, validation, and test sets...")
total_size = len(full_dataset)
train_size = int(TRAIN_RATIO * total_size)
val_size = int(VAL_RATIO * total_size)
test_size = total_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

print("\n" + "="*50)
print("DATASET PREPARATION COMPLETE")
print(f"Total valid samples: {total_size}")
print(f"Training set size:   {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size:       {len(test_dataset)}")
print("="*50)


# Verification: Test loading one sample from each set
print("\nVerifying data loading for one sample from each set...")
try:
  train_sample = train_dataset[0]
  val_sample = val_dataset[0]
  test_sample = test_dataset[0]
  print(f"  - Training sample loaded successfully. Pixel values shape: {train_sample['pixel_values'].shape}")
  print(f"  - Validation sample loaded successfully. Pixel values shape: {val_sample['pixel_values'].shape}")
  print(f"  - Test sample loaded successfully. Pixel values shape: {test_sample['pixel_values'].shape}")
  print("\nVerification successful. Proceed to Cell 2 for model training.")
except Exception as e:
  print(f"\nERROR during data loading verification: {e}")
print("-" * 50)
```


```python
# Model & Environment Setup
import torch
from transformers import AutoModelForVideoClassification
import os


# Define Data folder
DRIVE_BASE_PATH = "/content/drive/My Drive/Colab Notebooks"
MODEL_NAME = "MCG-NJU/videomae-base"
CHECKPOINT_DIR = os.path.join(DRIVE_BASE_PATH, "SOCAL_Checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME.replace('/', '_')}_checkpoint.pth")


# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used (Device): {device}")
print(f"Checkpoints will be saved here: {CHECKPOINT_PATH}")


# Initialize model and optimizer
print(f"Initialize '{MODEL_NAME}' model structure...")
model = AutoModelForVideoClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True
)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
print("\nThe model and optimizer have been initialized successfully。")
```


```python
# Advanced Training Process
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import collections
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from tqdm.auto import tqdm
import os
import warnings
warnings.filterwarnings("ignore")

if 'model' not in globals() or 'train_dataset' not in globals():
  raise RuntimeError("Error, please run the previous cells")


# training setting based on config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None


# Dropout
if config["dropout_rate"] is not None:
  for module in model.modules():
    if isinstance(module, torch.nn.Dropout):
      module.p = config["dropout_rate"]

# Optimizer
if config["optimizer_choice"].lower() == 'adamw':
  optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
  print(f"Use AdamW optimizer，learning rate: {config['learning_rate']}, Weight Decay: {config['weight_decay']}")
elif config["optimizer_choice"].lower() == 'sgd':
  optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9, weight_decay=config['weight_decay'])
  print(f"Use SGD optimizer，learning rate: {config['learning_rate']}, Weight Decay: {config['weight_decay']}")
else:
  raise ValueError("Invalid optimizer option!")


# Learning Rate Scheduler
scheduler = None
if config["use_lr_scheduler"]:
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config["lr_scheduler_patience"], verbose=True)
  print("Learning rate scheduler enabled (ReduceLROnPlateau)。")


def plot_metrics(history):
  clear_output(wait=True)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

  epochs_ran = len(history['train_loss'])
  epoch_axis = range(1, epochs_ran + 1)

  # loss curve plotting
  ax1.plot(epoch_axis, history['train_loss'], 'o-', label='Training Loss')
  ax1.plot(epoch_axis, history['val_loss'], 'o-', label='Validation Loss')
  ax1.set_title('Training and Validation Loss')
  ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
  ax1.legend(); ax1.grid(True)

  # accuracy curve plotting
  ax2.plot(epoch_axis, history['val_accuracy'], 'o-', label='Validation Accuracy', color='green')
  ax2.set_title('Validation Accuracy')
  ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
  ax2.legend(); ax2.grid(True)

  plt.suptitle("Training Progress")
  display(plt.gcf())
  plt.close(fig)


# Loading Checkpoints and Initializing History
start_epoch = 0
training_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
best_val_loss = float('inf')
early_stopping_counter = 0

if os.path.exists(CHECKPOINT_PATH) and not config["force_retrain"]:
  print(f"Checkpoint found！Loading '{CHECKPOINT_PATH}'...")
  checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  start_epoch = checkpoint['epoch'] + 1
  training_history = checkpoint['history']
  if 'best_val_loss' in checkpoint:
      best_val_loss = checkpoint['best_val_loss']
  print(f"Successfully resume progress,，Start training from Epoch {start_epoch + 1}")
  plot_metrics(training_history)
else:
  if config["force_retrain"]:
    print("If `FORCE_RETRAIN` is detected, checkpoints will be ignored.")
  else:
    print("Checkpoint not found, training will start from scratch")


# DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

train_labels = [train_dataset.dataset.dataframe.iloc[i]['success'] for i in train_dataset.indices]
class_counts = collections.Counter(train_labels)
total_samples = len(train_labels)
class_weights = torch.tensor([total_samples / (2 * class_counts.get(0, 1e-5)), total_samples / (2 * class_counts.get(1, 1e-5))], dtype=torch.float32).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
print(f"Calculated class weights: {class_weights.cpu().numpy()}")


# Training and validation loop
print("\nstart training...")
if start_epoch >= config["num_epochs"]:
  print("Complete the training and no need to retrain again")
else:
  for epoch in range(start_epoch, config["num_epochs"]):
    # --- Training Phase ---
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{config['num_epochs']}"):
      optimizer.zero_grad()
      pixel_values = batch['pixel_values'].to(device)
      labels = batch['labels'].to(device)
      with torch.cuda.amp.autocast(enabled=(scaler is not None)):
          outputs = model(pixel_values=pixel_values)
          loss = loss_fn(outputs.logits, labels)
      if scaler:
          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()
      else:
          loss.backward()
          optimizer.step()
      total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_dataloader)
    training_history['train_loss'].append(avg_train_loss)

    # --- Validation Phase ---
    model.eval()
    all_preds, all_labels = [], []
    total_val_loss = 0
    with torch.no_grad():
      for batch in tqdm(val_dataloader, desc=f"Validating Epoch {epoch+1}/{config['num_epochs']}"):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
          outputs = model(pixel_values=pixel_values)
          loss = loss_fn(outputs.logits, labels)
        total_val_loss += loss.item()
        all_preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_dataloader)
    val_accuracy = accuracy_score(all_labels, all_preds)
    training_history['val_loss'].append(avg_val_loss)
    training_history['val_accuracy'].append(val_accuracy)

    # learning rate update
    if scheduler:
        scheduler.step(avg_val_loss)

    # plotting
    plot_metrics(training_history)
    print(f"Epoch {epoch+1} Complete. Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Storing the best model and early stopping
    if avg_val_loss < best_val_loss:
      print(f"Validation loss {best_val_loss:.4f} improve to {avg_val_loss:.4f}。Store teh best model...")
      best_val_loss = avg_val_loss
      torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': training_history,
        'best_val_loss': best_val_loss,
      }, CHECKPOINT_PATH)
      early_stopping_counter = 0
    else:
      early_stopping_counter += 1
      print(f"Validation loss not improved。Apply early stopping: {early_stopping_counter}/{config['early_stopping_patience']}")
      if config["use_early_stopping"] and early_stopping_counter >= config["early_stopping_patience"]:
        print("Training stop。")
        break
```


```python
# Fianl results and visualization
import torch
from torch.utils.data import DataLoader # <--- 新增
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from IPython.display import display
import ipywidgets as widgets
from PIL import Image
from tqdm.auto import tqdm
import os


# Load the latest checkpoint
if 'model' not in globals() or 'test_dataset' not in globals() or 'CHECKPOINT_PATH' not in globals():
  raise RuntimeError("Error, please run the previous cells")

print(f"Load '{CHECKPOINT_PATH}' checkpoint")
if os.path.exists(CHECKPOINT_PATH):
  checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
  model.load_state_dict(checkpoint['model_state_dict'])
  print("Model loaded successfully")
else:
  raise FileNotFoundError(f"Error：cannot found '{CHECKPOINT_PATH}' checkpoint files")
model.eval()


# Display the performance results
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

all_preds_test, all_labels_test = [], []
with torch.no_grad():
  for batch in tqdm(test_dataloader, desc="loading testing results..."):
    pixel_values = batch['pixel_values'].to(device)
    labels = batch['labels'].to(device)
    outputs = model(pixel_values=pixel_values)
    all_preds_test.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
    all_labels_test.extend(labels.cpu().numpy())

report = classification_report(all_labels_test, all_labels_test, target_names=['Failure (0)', 'Success (1)'], zero_division=0)
accuracy = accuracy_score(all_labels_test, all_labels_test)

print(report)
print(f"Overall Accuracy: {accuracy:.4f}")
print("="*60 + "\n")


def show_prediction(sample_index):

  sample = test_dataset[sample_index]
  pixel_values = sample['pixel_values'].to(device)
  true_label_idx = sample['labels'].item()

  with torch.no_grad():
    outputs = model(pixel_values.unsqueeze(0))
  probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
  predicted_label_idx = torch.argmax(probabilities).item()

  # change the label
  class_names = {0: 'Failure', 1: 'Success'}
  predicted_label = class_names[predicted_label_idx]
  true_label = class_names[true_label_idx]

  # Visualize results
  print("--- Predict analysis ---")
  print(f"True Label: {true_label} (ID: {true_label_idx})")
  print(f"Predicted Label: {predicted_label} (ID: {predicted_label_idx})")
  print(f"Probabilities:")
  print(f"  - Failure (0): {probabilities[0]:.2%}")
  print(f"  - Success (1): {probabilities[1]:.2%}")

  # Display frames
  original_idx_in_full_dataset = test_dataset.indices[sample_index]
  trial_id = full_dataset.dataframe.iloc[original_idx_in_full_dataset]['trial_id']
  frame_paths = full_dataset.trial_frames_map.get(trial_id, [])

  num_frames_to_show = min(len(frame_paths), 8)
  indices_to_show = np.linspace(0, len(frame_paths) - 1, num_frames_to_show, dtype=int)

  fig, axes = plt.subplots(2, 4, figsize=(16, 6))
  axes = axes.flatten()
  fig.suptitle(f'Trial ID: {trial_id}', fontsize=16)

  for i, ax in enumerate(axes):
    if i < num_frames_to_show:
      img = Image.open(frame_paths[indices_to_show[i]])
      ax.imshow(img)
      ax.set_title(f'Frame {indices_to_show[i]}')
      ax.axis('off')
    else:
      ax.axis('off')

  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  plt.show()

# Slider widget
slider = widgets.IntSlider(
  value=0,
  min=0,
  max=len(test_dataset) - 1,
  step=1,
  description='Choice testing sample:',
  continuous_update=False,
  layout=widgets.Layout(width='80%')
)

# Link the widge event with functions
interactive_output = widgets.interactive_output(show_prediction, {'sample_index': slider})
display(slider, interactive_output)
```


```python

```


```python

```

```
# Project Documentation: Predicting SOCAL Procedure Outcomes with VideoMAE

---

## Overview

This project utilizes the **VideoMAE (Video Masked Autoencoder)** model to analyze image sequences from the **SOCAL (Surgical Objective-based Continuous Assessment of Laparoscopic Skills)** dataset. The primary goal is to predict the final outcome of a surgical procedure (e.g., success/failure) based on its video frames.

The entire process, from data preparation and model fine-tuning to interactive prediction, is designed to run seamlessly in a Google Colab environment.

---

## Workflow Explanation

The Colab Notebook is organized into the following core steps:

### 1. Environment Setup & Data Preparation

- **Mounting Google Drive**:
    
    - The first step connects to your Google Drive to access the image and label data stored at the specified path: `My Drive/Colab Notebooks/dataset_SOCAL_small_demo/`.
- **Custom `SOCALImageSequenceDataset` Class**:
    
    - To efficiently load the data into PyTorch, a custom `Dataset` class was created.
    - **Functionality**:
        - It reads a `dataframe` containing image filenames and their corresponding labels.
        - For each sample, it randomly selects a sequence of `num_frames` (e.g., 16) from a given `trial` to form a video clip.
        - **Data Augmentation**: During training, transformations like random horizontal flipping and rotation are applied to the image sequences. This increases the model's robustness and helps prevent overfitting.
    - Finally, PyTorch's `DataLoader` wraps the dataset to serve data in batches for model training.

### 2. Model Configuration & Fine-Tuning

- **Loading the Pre-trained Model**:
    
    - We use the `transformers` library to load a `VideoMAEForVideoClassification` model pre-trained on the [ImageNet-1K](https://www.google.com/url?q=https%3A%2F%2Fwww.image-net.org%2F) dataset. The pre-trained model has already learned rich, general-purpose visual features, providing an excellent starting point for our specific task.
- **Modifying the Classifier Head**:
    
    - The final layer (the classifier) of the pre-trained model was originally designed for ImageNet's 1000 classes.
    - We replace this layer with a new `nn.Linear` layer whose output dimension matches our task (e.g., 2 classes for success vs. failure). This is a critical step in transfer learning.

### 3. Model Training

- **Training Loop**:
    - The project implements a standard PyTorch training loop.
    - **Optimizer**: We use `AdamW`, an optimizer that performs exceptionally well with Transformer-based models.
    - **Loss Function**: `CrossEntropyLoss` is used, as it is the standard for classification problems.
    - During each epoch, the script calculates and logs the **loss** and **accuracy** on both the training and validation sets to monitor the model's learning progress.

### 4. Evaluation & Interactive Prediction

- **Model Evaluation**:
    
    - After training is complete, the model's final performance is measured on the test set.
    - Key metrics are calculated and displayed, including the **Confusion Matrix**, **Precision**, **Recall**, and **F1-score**, to provide a comprehensive assessment of the model's effectiveness.
- **Interactive Prediction UI**:
    
    - To easily test the model's capabilities, a simple user interface was built using `ipywidgets`.
    - **`Select 10 Random Images` Button**: When clicked, this selects 10 random image sequences and populates the dropdown menu below.
    - **Dropdown Menu**: Allows you to select a specific image sequence from the random sample.
    - **Output Area**:
        - Displays the first frame of the selected sequence with its **Bounding Boxes** overlaid.
        - Below the image, it shows the model's **prediction** for that sequence (e.g., "Prediction: Success").
        - It also lists the detected `tracking_id`s and the area of their respective bounding boxes.

---

## How to Run This Project

1. **Prepare Your Data**: Ensure your SOCAL dataset is located in the correct path within your Google Drive.
2. **Run Cells Sequentially**: Execute each cell in the Colab Notebook from top to bottom.
3. **Monitor Training**: Observe the loss and accuracy metrics during the training phase to ensure the model is learning correctly.
4. **Use the Interactive UI**: Once all cells have finished running, interact with the UI in the final cell to test the model's predictive performance.

---
```