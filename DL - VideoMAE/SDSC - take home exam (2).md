
## II. Embeddings with VideoMAE-2

---

Download the small demo dataset of surgical videos we have prepared. This dataset has 4 videos from 2 procedure types: Pituitary Tumor Surgery and Cholecystectomy.

The Cholecystectomy videos come from the cholec80 dataset is an endoscopic video dataset containing 80 videos of cholecystectomy surgeries performed by 13 surgeons. The videos are captured at 25 fps and downsampled to 1 fps for processing. The whole dataset is labeled with the phase and tool presence annotations. The phases have been defined by a senior surgeon in Strasbourg hospital, France. Since the tools are sometimes hardly visible in the images and thus difficult to be recognized visually, a tool is defined as present in an image if at least half of the tool tip is visible.

---

From google drive, either copy the dataset surgical_videos_demo to your drive or find the path to the shared folder. The link to the folder is: [https://drive.google.com/drive/folders/1wm5JtGBNETbFx5PHexLIEq85O6YVkIIQ?usp=sharing](https://www.google.com/url?q=https%3A%2F%2Fdrive.google.com%2Fdrive%2Ffolders%2F1wm5JtGBNETbFx5PHexLIEq85O6YVkIIQ%3Fusp%3Dsharing)

---

#### 1. Check that you have 4 videos in the dataset, explore the structure of the dataset
```python
import os
import sys
from google.colab import drive
import gdown
!pip install -q gdown
!pip install av


# Define Data folder
DRIVE_BASE_PATH = "/content/drive/My Drive/Colab Notebooks"
SURGICAL_DATA_FOLDER = os.path.join(DRIVE_BASE_PATH, "surgical_videos_demo")
DATA_LINK = 'https://drive.google.com/drive/folders/1wm5JtGBNETbFx5PHexLIEq85O6YVkIIQ?usp=sharing'


# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Selected device: {device}")
if torch.cuda.is_available():
  print(f"GPU Name: {torch.cuda.get_device_name(0)}")


# Mount Google Drive
try:
  drive.mount('/content/drive')
  print("Google Drive mounted successfully.")
except Exception as e:
  print(f"Error mounting Google Drive: {e}")
  raise SystemExit("Halting execution: Google Drive mount failed.")


# Data folder and validate
try:
  if os.path.isdir(SURGICAL_DATA_FOLDER):
    print("Dataset folder found.")
  else:
    print("Dataset folder not found. download dataset from the link")
    gdown.download_folder(DATA_LINK, output=DRIVE_BASE_PATH, quiet=False)

    # Check data folder
    if os.path.isdir(SURGICAL_DATA_FOLDER):
      print("Download complete! Dataset is ready.")
    else:
      raise FileNotFoundError("Download failed.")

except (FileNotFoundError, ValueError, Exception) as e:
    print(f"Setup Error: {e}")
    raise SystemExit("Halting execution: Dataset folder failed.")


# Explore vidoe files from folders
video_files_info = []
try:
  for phase_folder in sorted(os.listdir(SURGICAL_DATA_FOLDER)):
    phase_folder_path = os.path.join(SURGICAL_DATA_FOLDER, phase_folder)
    if os.path.isdir(phase_folder_path):
      print(f"Entering folder: {phase_folder}")
      for video_name in sorted(os.listdir(phase_folder_path)):
        if video_name.lower().endswith('.mp4'):
          video_path = os.path.join(phase_folder_path, video_name)
          video_files_info.append({
            'path': video_path,
            'name': video_name,
            'phase': phase_folder})
          print(f"  - Found video: {video_name}")

  # count video files and validate
  num_videos = len(video_files_info)
  print(f"\nTotal videos found: {num_videos}")
  if num_videos == 4:
    print("Correct number of videos (4) found in the dataset.")
  else:
    print(f"Warning: Expected 4 videos, but found {num_videos}.")

except FileNotFoundError:
  print(f"Error: Dataset path '{SURGICAL_DATA_FOLDER}' seems to be empty or structured incorrectly.")
```

#### 2. Display some random frames from the videos
```python
import ipywidgets as widgets
from IPython.display import display, clear_output, Video
import random
import matplotlib.pyplot as plt
import av


# Setup widgets menu and button to display video frames
if 'video_files_info' not in globals() or not video_files_info:
  print("Error: The 'video_files_info' list is empty.")
else:

  formatted_video_options = [f"{info['phase']}/{info['name']}" for info in video_files_info]
  video_map = {f"{info['phase']}/{info['name']}": info for info in video_files_info}

  # Setup widgets menu and button
  video_dropdown = widgets.Dropdown(
    options=formatted_video_options,
    description='Select Video:',
    style={'description_width': 'initial'},
    layout={'width': '80%'}
  )
  display_frame_button = widgets.Button(
    description="Display Random Frame",
    button_style='success',
    tooltip='Click to show a random frame from the selected video',
    icon='image',
    layout=widgets.Layout(width='220px', height='50px')
  )
  output_area = widgets.Output()

  # Define widget event to display video frame
  def on_display_frame_button_clicked(button_click):
    selected_key = video_dropdown.value
    video_info = video_map[selected_key]
    video_path = video_info['path']

    with output_area:
      clear_output(wait=True)
      print(f"Processing '{selected_key}' to get a random frame...")
      container = None
      try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        total_frames = stream.frames if stream.frames > 0 else int(stream.duration * stream.average_rate)

        if total_frames > 0:
          random_frame_idx = random.randint(0, total_frames - 1)
          time_base = stream.time_base
          offset = int(random_frame_idx * time_base.denominator / stream.average_rate)
          container.seek(offset, backward=True, any_frame=False, stream=stream)
          frame = next(container.decode(video=0))

          plt.figure(figsize=(8, 8))
          plt.imshow(frame.to_image())
          plt.title(f"{video_info['phase']}\n{video_info['name']}\nFrame Index (approx): {random_frame_idx}", fontsize=10)
          plt.axis('off')
          plt.show()
        else:
          print("This video has no valid frames to display.")
      except Exception as e:
        print(f"Could not process video '{video_info['name']}'. Reason: {e}")
      finally:
        if container:
          container.close()

  # Link widge event to functions
  display_frame_button.on_click(on_display_frame_button_clicked)
  display(widgets.VBox([video_dropdown, display_frame_button, output_area]))

```

#### 3. Display the VideoMAE-2 embeddings
Use the model VideoMAE-2 from huggingface: [https://huggingface.co/docs/transformers/model_doc/videomae](https://www.google.com/url?q=https%3A%2F%2Fhuggingface.co%2Fdocs%2Ftransformers%2Fmodel_doc%2Fvideomae) to visualize the embeddings graph of the videos from the surgical videos dataset. See the Examples section in the huggingface page to find useful functions.

Some help: You need to sample 16 frames as mentioned in the videomae paper: "Our backbone is 16-frame vanilla ViT-B".

You can have a sample rate: sample frames every x frames.
```
!pip install transformers
```

```
# imports you might need for that section
import numpy as np
import torch
from transformers import AutoImageProcessor, VideoMAEModel
import torchvision.transforms as transforms
from tqdm.notebook import tqdm
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
try:
  import umap.umap_ as umap
except ImportError:
  print("Warning: 'umap-learn' package not found. UMAP method will be unavailable.")
  umap = None
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import gc
import warnings
```

```
from transformers import AutoImageProcessor, VideoMAEModel

image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model_videomae_base = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base", output_hidden_states=True)
```

Use the model_videomae_base to generate the embeddings of the videos from the dataset and then visualize them in a graph.
```python
# Embedding & Sampling Parameters
NUM_FRAMES_PER_CLIP = 16
SAMPLING_OPTION = 'random_clip'  # Options: 'random_clip', 'sparse', 'sliding_window'
NUM_RANDOM_CLIPS = 3

# Visualization Parameters
DIM_REDUCTION_METHOD = 'PCA'  # Options: 'PCA', 't-SNE', 'UMAP', 'MDS'
```

```python
def get_video_clips(video_path, num_frames_per_clip=16, sampling_option='random_clip', stride=8, num_random_clips=3):
  try:
    # Open the container just to get metadata
    with av.open(video_path) as container:
      stream = container.streams.video[0]
      total_frames = stream.frames if stream.frames > 0 else int(stream.duration * stream.average_rate)
      if total_frames < num_frames_per_clip:
          return []
  except Exception as e:
    print(f"Warning: Could not open or read metadata from {os.path.basename(video_path)}. Reason: {e}")
    return []

  # Calculate the starting frame indices for all clips
  clip_start_indices = []
  if sampling_option == 'sliding_window':
    clip_start_indices = list(range(0, total_frames - num_frames_per_clip + 1, stride))
  elif sampling_option == 'random_clip':
    possible_starts = list(range(0, total_frames - num_frames_per_clip + 1))
    num_to_sample = min(len(possible_starts), num_random_clips)
    if num_to_sample > 0:
        clip_start_indices = random.sample(possible_starts, num_to_sample)
  if not clip_start_indices and sampling_option != 'sparse':
    return []

  # For each clip, open the video, seek, and extract
  final_clips = []
  for start_idx in clip_start_indices:
    clip_frames = []
    container = None
    try:
      container = av.open(video_path)
      stream = container.streams.video[0]

      time_base = stream.time_base
      seek_timestamp = int(start_idx * stream.average_rate.denominator * time_base.denominator / (stream.average_rate.numerator * time_base.numerator))
      container.seek(seek_timestamp, backward=True, any_frame=False, stream=stream)

      for i, frame in enumerate(container.decode(video=0)):
        if i >= num_frames_per_clip:
            break
        clip_frames.append(frame.to_image())

      if len(clip_frames) == num_frames_per_clip:
        final_clips.append(clip_frames)

    except Exception as e:
      print(f"Warning: Failed to extract a clip from {os.path.basename(video_path)} at frame {start_idx}. Reason: {e}")
    finally:
      if container:
        container.close()

  # For sparse sampling method
  if sampling_option == 'sparse':
    container = None
    try:
      container = av.open(video_path)
      indices = np.linspace(0, total_frames - 1, num_frames_per_clip, dtype=int).tolist()
      indices_set = set(indices)
      frames = {}
      for i, frame in enumerate(container.decode(video=0)):
        if i in indices_set:
          frames[i] = frame.to_image()
        if len(frames) == len(indices_set):
          break

      ordered_frames = [frames[i] for i in indices if i in frames]
      if len(ordered_frames) == num_frames_per_clip:
        final_clips.append(ordered_frames)

    except Exception as e:
      print(f"Warning: Failed to extract sparse clip from {os.path.basename(video_path)}. Reason: {e}")

    finally:
      if container:
        container.close()

  return final_clips

def visualize_embeddings(embeddings_array, labels, title_prefix="", point_size=100, text_labels=None):
  if len(embeddings_array) < 2:
    print(f"\nNot enough samples to plot for '{title_prefix}'.")
    return
  scaler = StandardScaler(); scaled_embeddings = scaler.fit_transform(embeddings_array)
  print(f"\nReducing dimensionality for {title_prefix} embeddings using {DIM_REDUCTION_METHOD}...")
  reducer = None
  try:
    if DIM_REDUCTION_METHOD == 'PCA':
      reducer = PCA(n_components=2, random_state=42)

    elif DIM_REDUCTION_METHOD == 't-SNE':
      perplexity = min(30, len(scaled_embeddings) - 1)

      if perplexity > 0: reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    elif DIM_REDUCTION_METHOD == 'UMAP' and umap:
      n_neighbors = min(15, len(scaled_embeddings) - 1)
      if n_neighbors > 0: reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)

    elif DIM_REDUCTION_METHOD == 'MDS':
      reducer = MDS(n_components=2, random_state=42, normalized_stress='auto')

    if not reducer:
      print(f"Could not initialize {DIM_REDUCTION_METHOD}, falling back to PCA.")
      reducer = PCA(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(scaled_embeddings)

  except Exception as e:
      print(f"Error during reduction: {e}"); return

  # Embedding visualization setting
  plt.figure(figsize=(12, 10)); color_map = {'pituitary_tumor_surgery': 'hotpink', 'Cholecystectomy': 'limegreen'}; legend_elements = []
  for i, label in enumerate(labels):
    color = color_map.get(label, 'gray')
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color=color, s=point_size, alpha=0.8, edgecolors='w')
    if text_labels and i < len(text_labels): plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1] + 0.1, text_labels[i], fontsize=8, ha='center')
    if label not in [l.get_label() for l in legend_elements]:
      legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10))
  title = f"VideoMAE-2 {title_prefix} Embeddings\n(Method: {DIM_REDUCTION_METHOD}, Sampling: {SAMPLING_OPTION})"
  plt.title(title, fontsize=14, weight='bold'); plt.xlabel(f"{DIM_REDUCTION_METHOD} Dim 1"); plt.ylabel(f"{DIM_REDUCTION_METHOD} Dim 2")
  plt.legend(handles=sorted(legend_elements, key=lambda x: x.get_label()), title="Surgical Phase", loc='best'); plt.grid(True, linestyle='--'); plt.show()
```

```python
if not video_files_info:
  print("Error: The 'video_files_info' list is empty..")
else:
  # Load Model
  model_videomae_base.eval()
  model_videomae_base.to(device)

  # Extract Embeddings
  print(f"\n--- Extracting Embeddings ({SAMPLING_OPTION} strategy) ---")
  all_clip_embeddings_info = []; video_clip_embeddings_map = defaultdict(list)
  with torch.no_grad():
    for video_info in tqdm(video_files_info, desc="Processing videos"):
      video_clips = get_video_clips(video_info['path'], NUM_FRAMES_PER_CLIP, SAMPLING_OPTION, 3, NUM_RANDOM_CLIPS)
      if not video_clips:
        print(f"Info: No clips were extracted from {video_info['name']}.")
        continue
      for clip_idx, frame_list in enumerate(video_clips):
        try:
          inputs = image_processor(frame_list, return_tensors="pt").to(device)
          outputs = model_videomae_base(**inputs)
          embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
          all_clip_embeddings_info.append({'embedding': embedding, 'video_name': video_info['name'], 'video_phase': video_info['phase'], 'clip_idx': clip_idx})
          video_clip_embeddings_map[video_info['name']].append(embedding)
        except Exception as e:
          print(f"Error processing clip from '{video_info['name']}': {e}")
  print(f"\nExtraction complete. Found embeddings for {len(all_clip_embeddings_info)} clips.")


  # Visualize Embeddings
  print("\n--- Visualizing Embeddings ---")
  # Plot 1: plot every emvedding points
  if all_clip_embeddings_info:
    clip_embeddings = np.array([info['embedding'] for info in all_clip_embeddings_info])
    clip_labels = [info['video_phase'] for info in all_clip_embeddings_info]
    clip_text = [f"{info['video_name'].replace('.mp4', '')}_c{info['clip_idx']}" for info in all_clip_embeddings_info]
    visualize_embeddings(clip_embeddings, clip_labels, title_prefix="Clip-Level", point_size=100, text_labels=clip_text)
  # Plot 2: plot average emvedding for each videos
  if video_clip_embeddings_map:
    avg_embeddings, avg_labels, avg_names = [], [], []
    for name, embs in video_clip_embeddings_map.items():
      if embs:
        avg_embeddings.append(np.mean(embs, axis=0))
        info = next((v for v in video_files_info if v['name'] == name), None)
        if info:
          avg_labels.append(info['phase'])
          avg_names.append(name.replace('.mp4', ''))
    if avg_embeddings:
      visualize_embeddings(np.array(avg_embeddings), avg_labels, title_prefix="Video-Level (Averaged)", point_size=350, text_labels=avg_names)

  print("\nProcess finished.")
```

```
## Overview
This notebook provides an interactive workflow to extract feature embeddings from videos using the powerful **VideoMAE-2 model**. It then uses dimensionality reduction techniques to visualize these embeddings, allowing you to see how your videos relate to one another in a 2D space. Videos that are visually similar will appear closer together in the final plots.

This tool is designed for researchers, students, or anyone interested in programmatic video analysis and computer vision.

***

## ⚙️ 1. Prerequisites & Setup
Before running the code, please complete these two essential setup steps:

1.  **Set Hardware Accelerator to GPU:**
    * In the Colab menu, go to **Runtime** → **Change runtime type**.
    * Select **T4 GPU** from the "Hardware accelerator" dropdown menu. This will significantly speed up the analysis.

2.  **Prepare Your Data in Google Drive:**
    * You need a folder containing your video files (e.g., `surgical_videos_demo`).
    * Upload this folder to your Google Drive. For best results, place it in the root of **My Drive** or inside the `Colab Notebooks` folder.
    * **Crucially, update the path in `Cell 0`** to match the location of your folder. For example:
        * `dataset_base_path = '/content/drive/MyDrive/surgical_videos_demo'`
        * `dataset_base_path = '/content/drive/MyDrive/Colab Notebooks/my_project/videos'`

***

## 🔬 2. Workflow: Cell by Cell

The notebook is divided into several cells, which should be run in order.

### Cell 0: Initial Setup
This cell prepares the environment. It performs the following actions:
* Installs necessary Python libraries (`av`, `umap-learn`).
* Imports all required modules for the notebook.
* Mounts your Google Drive, giving the notebook access to your files. You will be asked for authorization.

### Cell 1: Dataset Exploration & Verification
This cell checks if your data is accessible.
* **Action:** It scans the directory specified by `dataset_base_path`.
* **Output:** It will print a list of all `.mp4` files it finds, grouped by their parent folder. It also confirms the total number of videos found.
* **Goal:** To ensure the notebook can see your data before proceeding.

### Cell 2: Interactive Video Previewer
This cell provides tools to visually inspect your videos before running the main analysis.
* **Select Video Dropdown:** Choose a video from the list. The options are formatted as `folder/filename.mp4`.
* **Display Random Frame Button:** Click this to see a single, randomly selected frame from the chosen video. This is useful for a quick quality check.
* **Play Full Video Button:** Click this to load an embedded video player and watch the entire selected video.

### Cell 3: Embedding Extraction & Visualization
This is the main analysis cell where the core processing happens.

#### Key Concepts
* **Embeddings:** Think of an embedding as a "digital fingerprint" or a "DNA sequence" for a video clip. It's a list of numbers (a vector) that represents the visual content of the clip. The VideoMAE-2 model is an expert at creating these fingerprints.
* **Dimensionality Reduction:** An embedding has hundreds of dimensions, which we cannot visualize. This process is like flattening a 3D globe onto a 2D paper map. It takes the high-dimensional "fingerprint" and intelligently projects it onto a 2D plot, trying its best to preserve the original relationships.

#### Customizable Parameters
You can change these variables at the top of the cell to control the analysis:
* `SAMPLING_OPTION`: How to select clips from each video.
    * `'random_clip'`: (Default) Picks a few clips from random locations. Fast and provides a good overview.
    * `'sliding_window'`: Analyzes the entire video by taking overlapping clips. Very thorough but slower.
    * `'sparse'`: Takes a single clip with frames evenly sampled from the entire video.
* `DIM_REDUCTION_METHOD`: The algorithm used to create the 2D "map".
    * `'PCA'`: (Default) Very fast, great for a first look.
    * `'UMAP'`: (Recommended for quality) Excellent at showing both local clusters and the global structure. Requires `umap-learn` to be installed.
    * `'t-SNE'`: A classic method, great at revealing tight clusters.

#### Interpreting the Output Plots
Two plots will be generated at the end:
1.  **Clip-Level Embeddings Plot:**
    * **Each dot is a single clip** (e.g., 16 frames) from a video.
    * If dots from the same video (`video01_c0`, `video01_c1`, etc.) are clustered together, it means the video is visually consistent over time.
    * If they are spread out, the video contains a wide variety of scenes.
2.  **Video-Level (Averaged) Embeddings Plot:**
    * **Each dot is an entire video** (represented by the average of all its clip embeddings).
    * **This is the most important plot for comparison.** Videos that are close together on this plot are considered visually similar by the model. For example, all "Cholecystectomy" videos should hopefully cluster together, separate from the "pituitary_tumor_surgery" videos.
```