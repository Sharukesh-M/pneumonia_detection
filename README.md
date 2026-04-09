# Pneumonia Detection using CNN

This project implements a Convolutional Neural Network (CNN) to detect pneumonia from chest X-ray images. 

### [🚀 Live Demo - Project Website](https://Sharukesh-M.github.io/pneumonia_detection/)

## Project Structure

- `model.py`: Defines the CNN architecture using TensorFlow/Keras.
- `train.py`: Script to train the model on the pneumonia dataset.
- `gui.py`: A local Desktop application for making predictions.
- `download_dataset.py`: Script to download the dataset from Kaggle.
- `index.html`: A frontend presentation of the project.
- `pneumonia_cnn_model.h5`: Pre-trained model weights (excluded from Git due to size).

## Getting Started

1. **Install dependencies**:
   ```bash
   pip install tensorflow opencv-python scikit-learn kagglehub
   ```

2. **Download the dataset**:
   Run the following command to download the dataset:
   ```bash
   python download_dataset.py
   ```
   *Note: Ensure you have your Kaggle credentials set up if required.*

3. **Train the model**:
   ```bash
   python train.py
   ```

4. **Run the GUI**:
   ```bash
   python gui.py
   ```

## Model Architecture

The model is a Sequential CNN consisting of:
- 4 Convolutional blocks with Batch Normalization and Max Pooling.
- Fully connected Dense layer with 512 units.
- Sigmoid output for binary classification (Normal vs Pneumonia).

## Acknowledgments

Dataset provided by Paul Mooney on [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
