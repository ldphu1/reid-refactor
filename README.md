
# Introduction

Here is my python source code for Person Re-Identification (Re-ID) - a robust system for matching human identities across different camera views. With my code, you could:
* Extract discriminative 512-dimensional feature embeddings from human images using a custom ResNet-50 network (`model.py`)
* Train the model using a Batch Hard Triplet and Cross-Entropy Loss to effectively distinguish different identities (`train.py`)
* Build a feature gallery from a database of known identities (`build_gallery.py`)
* Run an inference app which detects people using YOLOv8 and identifies/tracks them across frames in a single video file (`reid_video_demo.py`)

<div align="center">
    <p>
    <img src="https://github.com/user-attachments/assets/a5286544-f871-42ae-b8e9-68f906be6f56" width="75%" ></img>
    <img src="https://github.com/user-attachments/assets/439ac77a-9eb6-4ee1-95ca-05fc04124e05" width="50%" ></img>
    </p>
  <p><i>Some examples of my model's output</i></p>
</div>

# Setup
1. Installation

Clone the repository and install the required packages:

    ```bash
        git clone [https://github.com/your-username/reid-refactor.git](https://github.com/your-username/reid-refactor.git)
        cd reid-refactor 
        pip install -r requirements.txt
    ```
3. Dataset Preparation
Place your datasets in the data/ directory. For example, if using Market-1501:
    ```bash
        data/
        └── Market-1501-v15.09.15/
            ├── bounding_box_train/
            ├── bounding_box_test/
            └── query/
    ```
# Usage Guide
1. Training
    ```bash
    python -m scripts.train 
    ```
3. Building Gallery
    ```bash
    python -m scripts.build_gallery 
    ```
    **Building a Custom Gallery for Specific Videos**
    
      Instead of using the default Market-1501 dataset, you can easily create custom galleries tailored for your specific videos. 
    
      Prepare the Gallery Folder: Create a new folder (e.g., `my_custom_gallery/`) and place the reference images of the people you want to track inside it.
      * Image Format: All images must be in `.jpg` format.
      
      * Naming Convention: The filename **must** start with the Person ID followed by an underscore `_`. The script parses the ID using the string before the first `_`.
    
         *Correct Examples:* `0001_front.jpg`, `0002_camera1.jpg`, `JohnDoe_1.jpg`.
      
         *Incorrect Examples:* `front_0001.jpg`, `image1.png`.
4. Running Demo
    ```bash
    python -m scripts.reid_video_demo --config configs/default.yaml --video_path path/to/video.mp4
    ```

# Dataset

The dataset used for training my model is the **[Market-1501](https://www.kaggle.com/datasets/sachinsarkar/market1501)** dataset.
The structure requires the standard Market-1501 splits: `bounding_box_train/` for training triplets, and `query/` along with `bounding_box_test/` for evaluation.

# Trained models

Due to GitHub's file size limits, the trained weights are hosted externally. 
 **[Download here](https://drive.google.com/drive/folders/1GIc0b7MpvtXpEHc4TFDwY3hDQz6u2Ylm?usp=sharing)**

*Note: After downloading, please place the `best_model.pth` and `gallery_market1501.pt` inside the `weights/` folder before running any scripts.*

# Model Architecture

The model is based on a pretrained ResNet-50 backbone with a dual-head design:

Components:
* Backbone: ResNet-50 (ImageNet pretrained)
* Global pooling: AdaptiveAvgPool2d(1)
* Embedding layer: Linear(2048 → 512)
* BatchNorm1d
* Dropout (p=0.5)
* L2 normalization
* Classifier
# Experiments

I trained the model for 60 epochs using the Adam optimizer, combining Batch Hard Triplet Loss and Cross-Entropy Loss (with label smoothing). The model's performance was monitored using TensorBoard (saved in `weights/`). During training, the model's Rank-1, Rank-5, and mAP are evaluated on the query set every 5 epochs. 

<p align = "center">
<img width="1041" height="658" alt="image" src="https://github.com/user-attachments/assets/ac644a64-125b-4232-a554-5f8de2ba53fe" />
</p>

As shown in the charts above, the loss converges smoothly, and the model achieves impressive final results on the Market-1501 dataset: **Rank-1 accuracy of ~89.2%**, **Rank-5 accuracy of ~95.7%** and **mAP of ~75.5%**. The checkpoint with the highest Rank-1 score is automatically saved as (`best_model.pth`).
## Repository Structure
    ```
    reid-refactor/
    ├── configs/            # YAML configuration files (Hyperparameters)
    ├── data/               # Raw and processed datasets (e.g., Market-1501)
    ├── models/             # Trained model checkpoints (.pth, .pt)
    ├── notebooks/          # Jupyter notebooks for EDA and testing
    ├── scripts/            # Entry point scripts for execution
    │   ├── train.py
    │   ├── build_gallery.py
    │   └── reid_video_demo.py
    ├── src/                # Core source code (reusable modules)
    │   ├── dataset.py      # Custom Dataset and Dataloader
    │   ├── loss.py         # Loss functions (e.g., Triplet Loss)
    │   ├── metrics.py      # Evaluation metrics (mAP, Rank-1)
    │   └── models/         # Neural Network architectures
    │       └── extractor.py
    ├── requirements.txt    # Project dependencies
    └── README.md
    ```
# Requirements

* python 3.8+
* pytorch
* torchvision
* ultralytics (YOLOv8)
* opencv-python (cv2)
* numpy
* pillow
* tqdm
* tensorboard
