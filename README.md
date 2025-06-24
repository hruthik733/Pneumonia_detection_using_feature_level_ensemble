# 🚀 Live Pneumonia Detection Web App

Experience real-time pneumonia detection using chest X-ray images — deployed and ready for use!

🧠 **Powered by** an ensemble of deep learning models with **Grad-CAM visualizations** for transparent decision-making.

⚡ **Fast Results:** Upload a chest X-ray and get an AI-powered diagnosis in just **3–4 seconds**.

<h3 align="center">🧪✨ <strong>Try the App Now</strong> ✨🧪</h3>

<p align="center">
  <a href="https://huggingface.co/spaces/hp733/pneumonia-xray-detector" target="_blank">
    <img src="https://img.shields.io/badge/👉%20Launch%20Pneumonia%20Detector-%20Click%20Here-ff69b4?style=for-the-badge&logo=huggingface&logoColor=white" alt="Launch App on Hugging Face"/>
  </a>
</p>

---

# Pneumonia Detection Using Feature-Level Ensemble of CNNs

## Project Overview
This project focuses on detecting pneumonia from chest X-ray images using a feature-level ensemble method that combines the strengths of multiple pre-trained Convolutional Neural Network (CNN) models. By leveraging models like VGG19 and EfficientNetB0, we aim to significantly enhance the accuracy of pneumonia detection. Additionally, we integrate Grad-CAM to generate visual heatmaps, allowing for more interpretable predictions by highlighting the areas of the lungs affected by pneumonia. The ultimate goal of this project is to build an efficient diagnostic tool that can assist healthcare professionals, particularly in resource-constrained environments, in making faster and more accurate decisions.

##Key Highlights
Feature-Level Ensembling: Combines features extracted from multiple CNN models, leveraging their individual strengths for improved classification accuracy.
Interpretability with Grad-CAM: Provides visual explanations of the model's decisions by localizing pneumonia-affected areas in chest X-ray images, making the results more transparent for medical professionals.
Impact on Healthcare: This automated tool aims to bridge the gap in diagnostic capabilities, especially in rural or resource-limited areas where access to radiologists is minimal.

## Objectives
- Achieve high accuracy in pneumonia detection by combining features from multiple pre-trained CNN architectures.
- Improve model performance through feature-level ensembling, enabling better generalization to diverse cases.
- Ensure interpretability of the model’s predictions by generating heatmaps with Grad-CAM, aiding clinicians in identifying pneumonia-infected regions.
- Develop a model that can be easily integrated into healthcare systems for early diagnosis, which is critical for patient treatment outcomes.

## Technologies Used
- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow, Keras
- **Pre-trained Models**: VGG19, EfficientNetB0, DenseNet121
- **Visualization**: Grad-CAM
- **Development Environment**: Google Colab

## Ensemble Model Architecture
Our ensemble approach involves extracting and concatenating features from multiple CNN models at a specific layer (usually the final convolutional layers) before feeding them into a fully connected layer for classification. This approach ensures that the strengths of each architecture are utilized, leading to improved performance over individual models.
### Architecture Diagram
![{AB9E1D25-D9E3-4F0D-9CE4-2B3AFF26D5B4}](https://github.com/user-attachments/assets/b34b429d-6502-443c-982b-17ea4aef919b)
<table>
  <tr>
    <td align="center">
      <strong>Input X-ray Image</strong><br>
      <img src="https://github.com/user-attachments/assets/4c3ea2f3-3087-4007-9bf4-43759b82356e" alt="Input X-ray" width="300"/>
    </td>
    <td align="center">
      <strong>Predicted Output (Heatmap)</strong><br>
      <img src="https://github.com/user-attachments/assets/0e922614-6181-4099-8600-11c12a03f39b" alt="Heatmap Output" width="300"/>
    </td>
  </tr>
</table>

## Future Work
- Model Optimization: Explore additional models for ensembling, such as ResNet or InceptionV3, and fine-tune hyperparameters to improve detection accuracy further.
- Testing on Multiple Datasets: Validate the model on other medical image datasets (e.g., lung cancer X-rays) to ensure robustness across different respiratory diseases.
- Deployment: Integrate the model into a user-friendly web interface using Flask or Django for real-world clinical use.


## Final Ensemble Model

Here is the final ensemble model. You can check it out here: [Final Ensemble Model](https://drive.google.com/file/d/1otsIiZJ0dxHbyxcCZMYL772bRyQGVdck/view?usp=drive_link).

https://github.com/user-attachments/assets/3134196e-d6b6-4092-9317-51d50af2f460





