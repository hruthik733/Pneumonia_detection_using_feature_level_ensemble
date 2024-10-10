# Pneumonia Detection Using Feature-Level Ensemble of CNNs

## Project Overview
This project uses a feature-level ensemble method by combining features from multiple pre-trained CNN models (VGG19, EfficientNetB0, and others) to enhance pneumonia detection accuracy from chest X-rays. Grad-CAM is used to generate visual heatmaps for better interpretability, helping identify the regions in the X-rays affected by pneumonia.

## Objectives
- To automatically detect pneumonia from chest X-rays with high accuracy.
- To improve model performance by ensembling features from different architectures.
- To make the model's predictions more interpretable using Grad-CAM.

## Technologies Used
- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow, Keras
- **Pre-trained Models**: VGG19, EfficientNetB0
- **Visualization**: Grad-CAM
- **Development Environment**: Google Colab
Sample input image:
<img src="https://github.com/user-attachments/assets/cc57d3df-58db-4abd-b2df-00788c1c7cbd" alt="IM-0041-0001" width="400"/>
Sample heat map image:
<img src="https://github.com/user-attachments/assets/41da4669-31ff-436a-a755-0d13e7554f54" alt="Screenshot 2024-10-08 132442" width="400"/>


