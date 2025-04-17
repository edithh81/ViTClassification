# Vision Transformer for MNIST Digit Recognition

This project implements a Vision Transformer (ViT) model for handwritten digit recognition using the MNIST dataset. The model achieves approximately 98% accuracy on the test set and is deployed with Streamlit for interactive usage.

## Project Overview

Vision Transformers have shown remarkable performance on various computer vision tasks by applying transformer architecture to image analysis. This project demonstrates how to:

- Implement a Vision Transformer from scratch using PyTorch
- Train the model on the MNIST handwritten digit dataset
- Deploy the model via a Streamlit web application for real-time inference

## Model Architecture

The model follows the Vision Transformer architecture with the following key components:

1. **Patch Embedding**: Divides the image into patches and embeds them with a convolutional layer
2. **Positional Encoding**: Adds position information to enable the model to understand the spatial relationships
3. **Transformer Encoder**: Processes the embeddings through self-attention and feed-forward networks
4. **Classification Head**: Performs global average pooling and uses a fully connected layer for final classification

Key parameters:
- Embedding dimension: 512
- Number of attention heads: 4/8 (varies by model version)
- Transformer layers: 4
- Patch size: 7×7
- Feed-forward dimension: 1024
- Dropout rate: 0.1/0.2

## Performance

The model achieves:
- Training accuracy: ~98%
- Validation accuracy: ~97.5%
- Test accuracy: ~96.3%

## Installation

```bash
# Clone the repository
git clone https://github.com/edithh81/ViTClassification.git
cd ViTClassification

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.6+
- PyTorch 1.7+
- Torchvision
- NumPy
- Matplotlib
- Streamlit
- Streamlit-drawable-canvas
- OpenCV (cv2)
- PIL

## Project Structure

```
ViTClassification/
│
├── notebooks/              # Jupyter notebooks for development and experimentation
│   └── viT.ipynb          # Training and evaluation notebook
│
├── model/                  # Directory for storing trained models
│   ├── best_model.pt      # Best performing model
│   └── viT.py             # Model architecture definition
│
├── data/                   # Data directory (automatically populated by torchvision)
│
├── app.py                  # Streamlit application for interactive inference
│
└── README.md               # Project documentation
```

## Usage

### Training the Model

The model training is implemented in the Jupyter notebook. To retrain the model:

1. Open `notebooks/viT.ipynb` in Jupyter Lab/Notebook
2. Follow the notebook cells for data preparation, model configuration, training, and evaluation
3. Trained models will be saved to the `model` directory

### Running the Streamlit Application

To run the interactive web application:

```bash
streamlit run app.py
```

The application will open in your default web browser and provides two options:
1. **Draw a digit**: Use the canvas to draw a digit and get real-time predictions
2. **Upload an image**: Upload an image of a handwritten digit for prediction

## Implementation Details

The Vision Transformer implementation includes:

1. **TransformerEncoder**: Implements the self-attention mechanism and feed-forward networks
2. **PatchPositionEmbedding**: Converts the image into patches and adds positional information
3. **ViT**: Main model class that combines all components for end-to-end inference

The model processes input images by:
- Converting the image into fixed-size patches (7×7)
- Embedding the patches into vectors
- Adding positional embeddings
- Processing through transformer layers
- Applying global average pooling
- Final classification through fully connected layers

## License

[Add your license information here]

## Acknowledgements

- The Vision Transformer architecture is based on the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al.
- The MNIST dataset is provided by Yann LeCun et al.
