# Custom VGG16 Document Image Classification

## Description

The **Custom VGG16 Document Image Classification** project implements a deep learning solution using the VGG16 architecture for classifying document images into 9 predefined categories. The project uses PyTorch for model training and evaluation, with a focus on fine-tuning the VGG16 model to achieve high accuracy in document classification. The dataset is divided into training and testing sets, and various data augmentation techniques are employed to improve model generalization.

## Features

- **VGG16 Model Architecture**: Utilizes the well-known VGG16 model with custom layers for document image classification.
- **Data Augmentation**: Implements random resized cropping and horizontal flipping for training images to enhance model robustness.
- **Training and Testing**: The project includes a training loop that tracks both training and test accuracy over multiple epochs.
- **Cross-Platform Compatibility**: The project is implemented in Python and supports both CPU and GPU for training.
- **Evaluation**: Provides detailed accuracy metrics after each epoch, including testing accuracy on the test set.

## Technologies Used

- **Programming Language**: Python
- **Deep Learning Frameworks**: PyTorch, TensorFlow (optional for comparison)
- **Libraries**: 
  - PyTorch (torch, torch.nn, torchvision)
  - TensorFlow (optional for model comparison)
  - NumPy
  - Matplotlib (for visualizations)
  - OpenCV (optional for image preprocessing)
- **Hardware**: GPU (CUDA-enabled) or CPU for model training

## Data

The dataset used in this project consists of document images classified into 9 different categories. The images are stored in two directories:
- **train**: Contains the training images.
- **test**: Contains the test images.

The images are loaded using the `ImageFolder` class from PyTorch’s `torchvision.datasets`, with transformations applied for both training and testing.

## Installation

To set up this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AnoopCA/Custom_VGG16_Document_Image_Classification.git
   cd custom-vgg16-document-image-classification
   
   ```