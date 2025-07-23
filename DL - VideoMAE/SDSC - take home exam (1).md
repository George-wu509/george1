
# SDSC - Senior ML engineer - At Home assignment

~1-2 hours

---

**Do not edit this notebook directly.** Make a copy of it to your Drive and then share it with [margaux@surgicalvideo.io](mailto:margaux@surgicalvideo.io) and [andy@surgicalvideo.io](mailto:andy@surgicalvideo.io) when you are done with the assignment. Thank you!

## I.Python implementation

---

Download or access the shared google drive folder: [https://drive.google.com/drive/folders/1zt03aV2OozARZvIEtVvhdSMvbA61jTxY?usp=sharing](https://www.google.com/url?q=https%3A%2F%2Fdrive.google.com%2Fdrive%2Ffolders%2F1zt03aV2OozARZvIEtVvhdSMvbA61jTxY%3Fusp%3Dsharing). This is a smaller version prepared for this demo of the Simulated Outcomes Following Carotid Artery Laceration (SOCAL) dataset ([https://figshare.com/articles/dataset/Simulated_Outcomes_following_Carotid_Artery_Laceration/15132468](https://www.google.com/url?q=https%3A%2F%2Ffigshare.com%2Farticles%2Fdataset%2FSimulated_Outcomes_following_Carotid_Artery_Laceration%2F15132468))

```python
from google.colab import drive
drive.mount('/content/drive')
```

Put the imports you need here
```python
import glob
import os
import random
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import gdown
!pip install -q gdown
```

#### 1. Check how many images there are in the dataset
```python
# Define Data folder
DRIVE_BASE_PATH = "/content/drive/My Drive/Colab Notebooks"
DATA_FOLDER = os.path.join(DRIVE_BASE_PATH, "dataset_SOCAL_small_demo")
IMAGE_FOLDER = os.path.join(DATA_FOLDER, 'images')
LABEL_FOLDER = os.path.join(DATA_FOLDER, 'labels')
DATA_LINK = 'https://drive.google.com/drive/folders/1zt03aV2OozARZvIEtVvhdSMvbA61jTxY'


# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Selected device: {device}")
if torch.cuda.is_available():
  print(f"GPU Name: {torch.cuda.get_device_name(0)}")


# Mount Google Drive
try:
  drive.mount('/content/drive', force_remount=True)
  print("Google Drive mounted successfully!")
except Exception as e:
  print(f"Error mounting Google Drive: {e}")
  raise SystemExit("Halting execution: Google Drive mount failed.")


# Data folder and validate
try:
  if os.path.isdir(DATA_FOLDER):
    print("Dataset folder found.")
  else:
    print("Dataset folder not found. download dataset from the link")
    gdown.download_folder(DATA_LINK, output=DRIVE_BASE_PATH, quiet=False)

    # Check data folder
    if os.path.isdir(DATA_FOLDERh):
      print("Download complete! Dataset is ready.")
    else:
      raise FileNotFoundError("Download failed.")

  # Check the image and label folders
  if not os.path.exists(IMAGE_FOLDER):
      raise FileNotFoundError(f"The 'images' folder was not found at: {IMAGE_FOLDER}")
  if not os.path.exists(LABEL_FOLDER):
      raise FileNotFoundError(f"The 'labels' folder was not found at: {LABEL_FOLDER}")
  print(f"Dataset path validated: '{DATA_FOLDER}'")

  # Get all image file names
  image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
  if not image_files:
      raise ValueError("No image files found in the image folder.")

except (FileNotFoundError, ValueError, Exception) as e:
    print(f"Setup Error: {e}")
    raise SystemExit("Halting execution: Dataset folder failed.")


# Count the images and labels
matched_count = 0
unmatched_count = 0

for img_name in image_files:
  label_name = os.path.splitext(img_name)[0] + '.txt'
  label_path = os.path.join(LABEL_FOLDER, label_name)

  if os.path.exists(label_path):
    matched_count += 1
  else:
    unmatched_count += 1


# Display Dataset Summary
print("\n" + "="*30)
print("--- Dataset Summary ---")
print(f"Total images found: {len(image_files)}")
print(f"Images with a matching label file: {matched_count}")
print(f"Images without a matching label file: {unmatched_count}")
print("="*30)
print("\nSetup and analysis complete.")
```

#### 2. Display bounding boxes on images
Create a function that selects 10 random from the dataset, displays the images and overlays the bounding boxes on the frames.

```python
import cv2
import random
import os
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output


# Define color palette for different tracking_id
COLOR_PALETTE = [
  (0, 255, 0),  # Green
  (0, 0, 255),  # Red
  (255, 0, 0),  # Blue
  (0, 255, 255), # Cyan
  (255, 0, 255), # Magenta
  (255, 255, 0), # Yellow
  (0, 128, 255), # Orange
  (128, 0, 128)  # Purple
]


# Load bounding boxes data function
def load_bounding_boxes(image_name, LABEL_FOLDER):
  label_name = os.path.splitext(image_name)[0] + '.txt'
  label_path = os.path.join(LABEL_FOLDER, label_name)
  bboxes = []
  if not os.path.exists(label_path):
    return bboxes
  with open(label_path, 'r') as f:
    for line in f.readlines():
      parts = line.strip().split()
      if len(parts) == 5:
        try:
          bboxes.append({
            'tracking_id': int(parts[0]),
            'bbox_norm': tuple(map(float, parts[1:]))
          })
        except ValueError:
          pass
    return bboxes


# Define widgets button to select random images from dataset
random_button = widgets.Button(
  description="Select 10 Random Images",
  button_style='success',
  tooltip='Click to select 10 new random images from the dataset',
  icon='random'
)
image_dropdown = widgets.Dropdown(
  options=[],
  description='Select Image:',
  disabled=True,
)
output_area = widgets.Output()


# Define widgets event to display image, bbox and info
def on_image_select(change):
  if not change['new']:
    return

  # add images and load bbox info
  img_name = change['new']
  img_path = os.path.join(IMAGE_FOLDER, img_name)
  img = cv2.imread(img_path)
  bboxes_info = load_bounding_boxes(img_name, LABEL_FOLDER)

  # open the widge area to display image
  with output_area:
    clear_output(wait=True)
    if img is None:
      print(f"Error: Could not load image '{img_name}'")
      return

    # Load image and assign color palette
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    unique_ids = sorted(list(set(bbox['tracking_id'] for bbox in bboxes_info)))
    color_map = {uid: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, uid in enumerate(unique_ids)}
    box_details = []


    # Setup bbox for image
    for bbox_data in bboxes_info:
      tracking_id = bbox_data['tracking_id']
      x_c, y_c, width, height = bbox_data['bbox_norm']
      x_min, y_min = int((x_c - width/2)*w), int((y_c - height/2)*h)
      x_max, y_max = int((x_c + width/2)*w), int((y_c + height/2)*h)

      # Setup bbox size and color
      box_color = color_map.get(tracking_id)
      area = (x_max - x_min) * (y_max - y_min)
      box_details.append({'id': tracking_id, 'area': area})
      cv2.rectangle(img_rgb, (x_min, y_min), (x_max, y_max), box_color, 2)
      cv2.putText(img_rgb, str(tracking_id), (x_min + 5, y_min + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2, cv2.LINE_AA)

    # Display image and axis
    plt.figure(figsize=(12, 10))
    plt.imshow(img_rgb)
    plt.title(f"Displaying: {img_name}")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.show()

    # disply image info details on the area
    print("\n--- Bounding Box Details ---")
    if not box_details:
      print("No bounding boxes found for this image.")
    else:
      sorted_details = sorted(box_details, key=lambda x: x['id'])
      for detail in sorted_details:
        print(f"  - Tracking ID: {detail['id']:<5} Area: {detail['area']:,} pixels²")


# Random select images and put image list into menu
def on_button_click(b):
  images_with_labels = [
    name for name in image_files
    if os.path.exists(os.path.join(LABEL_FOLDER, os.path.splitext(name)[0] + '.txt'))
  ]
  sample_size = min(10, len(images_with_labels))
  if sample_size == 0:
    with output_area:
        clear_output()
        print("Error: No images with corresponding labels found.")
    return

  # Setup menu
  selected_images = random.sample(images_with_labels, sample_size)
  selected_images.sort()
  image_dropdown.unobserve(on_image_select, names='value')
  image_dropdown.options = selected_images
  image_dropdown.value = None
  image_dropdown.value = selected_images[0]
  image_dropdown.disabled = False
  image_dropdown.observe(on_image_select, names='value')


# Link widge event to functions
random_button.on_click(on_button_click)
image_dropdown.observe(on_image_select, names='value')

if 'image_files' in locals():
  ui = widgets.VBox([random_button, image_dropdown, output_area])
  display(ui)
  print("UI Loaded. Clicking button to populate initial image list...")
  random_button.click()
else:
  print("Error: Prerequisite variables not found. Please run the previous cell.")
```

```
# 📓 Documentation: Bounding Box Visualization Notebook

## Overview
This notebook provides a complete workflow to set up an object detection dataset and interactively visualize images with their corresponding bounding boxes. The script automatically handles data download and provides simple UI controls for exploration.

***

### 🚀 **1. Prerequisites**

* **GPU Runtime:** For best performance, ensure your Colab runtime is set to use a GPU. In the menu, go to **Runtime** → **Change runtime type** → **T4 GPU**.

***

###  **2. Workflow Guide**

The notebook is divided into two main cells. Run them in order.

#### **Cell 1: Dataset Setup & Validation**
This first cell prepares your environment and ensures the dataset is ready.

* **Key Actions:**
    * **Mounts Google Drive:** You will be asked to authorize access.
    * **Checks for Dataset:** It looks for the `dataset_SOCAL_small_demo` folder in your `My Drive/Colab Notebooks/`.
    * **Automatic Download:** If the dataset folder is not found, the script will automatically download and place it in the correct location for you.
    * **Validates Content:** It verifies that the required `images` and `labels` subfolders exist.
    * **Prints a Summary:** At the end, it displays a summary of how many total images were found and how many have a corresponding bounding box label file.

#### **Cell 2: Interactive Bounding Box Viewer**
This cell launches the main visualization tool. It allows you to view images and their object labels.

* **How to Use the Tool:**
    1.  **Click the "Select 10 Random Images" button.** This will find 10 random images from your dataset that have corresponding label files and load their names into the dropdown menu below.
    2.  **Use the "Select Image" dropdown menu.** Choose an image from the list to display it. The first image from the random selection will be shown automatically.

* **Understanding the Output:**
    * **The Image Plot:** An image will be displayed with colored rectangles drawn on it.
        * Each **colored box** is a **bounding box** identifying an object.
        * The **number** on each box is its unique **tracking_id**. Different IDs are assigned different colors for clarity.
    * **Bounding Box Details:** Below the image, a text summary lists each bounding box found, its `Tracking ID`, and its total pixel `Area`.
```
