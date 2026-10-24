# Chest X-Ray Image Classifier

## Overview

A machine-learning project that uses transfer learning to classify chest X-ray images into NORMAL and PNEUMONIA categories.

I developed this project to explore how artificial intelligence can be applied to healthcare and gain practical experience building and evaluating neural networks.

## Technologies Used

- Python
- PyTorch and Torchvision
- Google Colab
- Hugging Face Datasets
- MobileNetV3 Small
- Scikit-learn
- Pillow

## Dataset

The project uses the `keremberke/chest-xray-classification` dataset, containing:

- **4,077** training images
- **1,165** validation images
- **582** test images

The images are resized to 224 × 224 pixels and normalized before being processed by the neural network.

## Methodology

1. Loaded and explored the chest X-ray dataset.
2. Resized and normalized the images.
3. Adapted a pretrained MobileNetV3 Small model for two-class classification.
4. Froze the pretrained model's parameters and trained its new classification layer for three epochs.
5. Used cross-entropy loss and the Adam optimizer.
6. Evaluated performance on separate validation and test datasets.
7. Created an image-upload feature to demonstrate predictions.

## Results

| Metric | Result |
|---|---|
| Validation accuracy | 92.27% |
| Test accuracy | 91.58% |
| Correct test predictions | 533/582 |
| Pneumonia precision | 96% |
| Pneumonia recall | 91% |
| Pneumonia F1-score | 94% |

### Confusion Matrix

| Actual / Predicted | NORMAL | PNEUMONIA |
|---|---|---|
| NORMAL | 157 | 14 |
| PNEUMONIA | 35 | 376 |

The model correctly classified 533 of 582 test images.

However, it incorrectly classified 35 pneumonia-labeled images as normal, highlighting the importance of reducing false negatives in medical AI.

## Image Upload and Prediction

The project includes an interactive prediction feature within Google Colab.

Users can upload a chest X-ray image, which is automatically preprocessed and passed through the trained model.

The model then displays its predicted category: NORMAL or PNEUMONIA.

These predictions are for educational demonstration only.

## Limitations

The model was evaluated on a limited dataset and has not undergone independent clinical validation.

Its performance may differ on images from other hospitals, imaging devices, or patient populations.

The model may produce incorrect predictions and should not be used for medical diagnosis or medical decision-making.

## Future Improvements

1. Compare different neural network architectures, such as ResNet and EfficientNet.
2. Investigate incorrectly classified images to understand the model's weaknesses.
3. Implement Grad-CAM to visualize which image regions influence predictions.
4. Evaluate performance using additional independent datasets.
5. Develop a more accessible user interface.
6. Explore techniques for improving generalization and reducing false negatives.

## Project Files

- `Medical_Image_Classifier.ipynb` — Complete Google Colab notebook containing the code, model training, evaluation, and prediction feature.
- `medical_image_classifier.pth` — Saved trained model weights.

## Conclusion

This project provided practical experience in applying deep learning to medical image classification.

Through developing the classifier, I gained experience in dataset preparation, transfer learning, neural network training, performance evaluation, and interpreting model limitations.

The project strengthened my interest in artificial intelligence and its potential applications in healthcare.

## Project Status

## Dataset Source

Dataset: [Chest X-Ray Classification](https://huggingface.co/datasets/keremberke/chest-xray-classification)

The dataset contains labeled chest X-ray images in two categories: NORMAL and PNEUMONIA.

Dataset ownership and licensing belong to the original dataset providers. Review the dataset's license and usage conditions before redistributing its images.

## How to Run the Project

1. Open `Medical_Image_Classifier.ipynb` in Google Colab.
2. Enable a GPU runtime (Runtime → Change runtime type → T4 GPU).
3. Install the required dependencies as shown in the notebook.
4. Run the notebook cells in order to load the dataset, prepare the images, and train the model.
5. Evaluate the model using the validation and test datasets.
6. Use the final image-upload cell to test the classifier on an example chest X-ray.

Note: Running the training cells again will retrain the model. The saved `.pth` file contains previously trained model weights.

## Reproducibility

The project was developed in Google Colab using PyTorch, Torchvision, and Hugging Face Datasets.

The dataset was loaded using `datasets==3.6.0`.

Results may vary between training runs because of randomness in model training and data shuffling.

**Working educational prototype.**

This project is not a clinically validated medical diagnostic system.