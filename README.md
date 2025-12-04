# Pneumonia Detection Using Feature-Level Ensemble of CNNs
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange"/>
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-red"/>
  <img src="https://img.shields.io/badge/Model-Feature%20Level%20Ensemble-yellow"/>
  <img src="https://img.shields.io/badge/Explainability-GradCAM-brightgreen"/>
  <img src="https://img.shields.io/badge/Dataset-Chest%20X--Ray%20(NIH%20%2B%20Kaggle)-lightgrey"/>
  <img src="https://img.shields.io/badge/Deployment-HuggingFace%20Spaces-ffcc00"/>
  <img src="https://img.shields.io/badge/Status-Production%20Ready-success"/>
  <img src="https://img.shields.io/badge/License-MIT-blueviolet"/>
</p>

---

## 🚀 LIVE PNEUMONIA DETECTION WEB APP

> 🩺 **Real-time Pneumonia Detection using Chest X-ray Images**  
> ⚡ Powered by deep learning ensemble + Grad-CAM for explainability  
> 🕒 **Get results in <3 seconds! Just upload and diagnose.**

---

<h2 align="center">🧪🔬 <strong>TRY IT LIVE</strong> 🔬🧪</h2>

<p align="center">
  <a href="https://huggingface.co/spaces/hp733/Pneumonia_detection_using_Feature-level_Ensemble_Learning" target="_blank">
    <img src="https://img.shields.io/badge/CLICK%20TO%20LAUNCH-LIVE%20APP-ff3366?style=for-the-badge&logo=huggingface&logoColor=white" alt="Launch App on Hugging Face"/>
  </a>
</p>

---

## 🧠 Project Overview

This project focuses on detecting pneumonia from chest X-ray images using a **feature-level ensemble** method that combines the strengths of multiple pre-trained Convolutional Neural Network (CNN) models. By leveraging architectures like **VGG19**, **EfficientNetB0**, and **DenseNet121**, we aim to significantly enhance the accuracy and generalizability of pneumonia detection by **leveraging the unique strengths of each architecture**.

To improve interpretability, we integrate **Grad-CAM** to generate visual heatmaps, which help users understand the decision-making process of the model. These visualizations highlight areas of the chest X-ray that the model focused on while making a prediction, thus improving transparency and trust.

> ⚠️ **Note**: The Grad-CAM heatmaps reflect the **model's attention**, not exact disease localization. Since the model is trained on whole X-ray images (not pixel-level segmented data), the heatmaps highlight regions that **influenced the prediction**, which may correlate with clinical features but do not guarantee accurate localization of pneumonia-affected areas.  

---

## ✨ Key Highlights

- 🔗 **Feature-Level Ensemble**: Combines feature representations from multiple CNN models to boost accuracy and robustness.
- 🔍 **Explainability with Grad-CAM**: Generates attention heatmaps to visualize what parts of the image contributed most to the model’s decision.
- 🩺 **Clinical Relevance**: Supports rapid screening in areas lacking medical infrastructure, aiding early detection and treatment of pneumonia.
- 🌐 **Deployed Web App**: Live inference app hosted on Hugging Face Spaces allows instant user interaction and real-time diagnosis.

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
- **Deployment: Hugging Face Spaces (with Gradio)

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


## 📄 Publication

[![Published in Springer](https://img.shields.io/badge/Published%20In-Springer%20Nature-brightgreen)](https://doi.org/10.1007/978-981-96-7253-0_26)

This work was published in **Springer Nature** as part of the  
*Proceedings of the 9th International Conference on Micro-Electronics, Electromagnetics and Telecommunication (ICMEET 2024), NIT Mizoram.*

🔗 **DOI:** [https://doi.org/10.1007/978-981-96-7253-0_26](https://doi.org/10.1007/978-981-96-7253-0_26)

---

### 📚 Citation
```bibtex
@inproceedings{hruthik2024pneumonia,
  title={Pneumonia Detection in Chest X-Rays Using Feature-Level Ensemble Learning},
  author={Hruthik Krishna, P and Pranathi, K and Ganesh, K and Surya, K},
  booktitle={International Conference on Microelectronics, Electromagnetics and Telecommunication},
  pages={309--322},
  year={2024},
  organization={Springer}
}
