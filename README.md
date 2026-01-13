
# Deepfake Face Detection

<img width="1875" height="797" alt="Screenshot 2025-08-20 203139" src="https://github.com/user-attachments/assets/7d339486-96c4-4fed-8488-68cfde07a0e9" />

## Project Overview
This repository contains the code and resources for a **Deepfake Face Detection** system, developed as a final graduation project for the NTI summer training program. In an era where synthetic media blurs reality, this project aims to develop a robust deep learning solution for accurately discerning real from AI-generated (deepfake) facial images, fostering a more trustworthy digital environment.

## Team Members
- Dina Abdullah
- Wahb Mohamed
- Maram Mohamed

## Dataset
This project utilizes the **DeepFake Detection Challenge (DFDC)** dataset from Kaggle.
- **Total Images:** 95,634 frames
- **Class Distribution:**
  - **FAKE (AI-generated):** 79,341 images
  - **REAL (Authentic):** 16,293 images

The dataset includes image files and a metadata CSV file with crucial information such as video source, frame dimensions, and labels.

## Project Structure
1. **Data Preprocessing**
2. **Model Selection & Training**
3. **Evaluation & Testing**
4. **Future Enhancements**

### 1. Data Preprocessing
To ensure model robustness and handle challenges like class imbalance, the following preprocessing steps were applied:
- **Addressing Imbalance:** A balanced subset of 16,000 images per class was created. The data was split into:
  - 80% Training
  - 10% Validation
  - 10% Testing
- **Resizing:** Images were resized to `224x224x3` (RGB) to reduce computational load while maintaining compatibility with the model.
- **Normalization:** Pixel values were scaled to the range `[0, 1]` to stabilize training and mitigate exploding gradients.
- **Augmentation:** Techniques like rotation, flipping, and shifting were applied (on the training set only) to improve generalization.

### 2. Model Selection
Several state-of-the-art CNN architectures were evaluated, including **ResNet**, **EfficientNet**, and **Xception**. **EfficientNet-B4** was selected for its optimal balance between high accuracy (~82.6% top-1) and computational efficiency (19M parameters), outperforming larger models like NASNet-A (89M parameters) with similar accuracy.

**Key Paper:** [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

### 3. Training & Optimization
- **Framework:** TensorFlow
- **Environment:** Google Colab (for GPU acceleration)
- **Loss Function:** Binary Cross-Entropy
- **Optimizer:** Adam
- **Techniques:**
  - Early Stopping (to prevent overfitting)
  - Model Checkpointing (saved to Google Drive after each epoch to resume training after disconnections)
  - Learning Rate Reduction

**Challenges Overcome:**
- Extensive training time and Colab runtime limitations were mitigated through persistent checkpointing and multi-account training continuation.

### 4. Evaluation
The model achieved a **90% accuracy** on the test set, demonstrating strong performance in distinguishing between real and deepfake faces.

### 5. Model Uploading 
as the model size ~ 200 mb so we've uploaded it on hugging face and here is the link if you're interested 
**model link :** https://huggingface.co/Wahb12111/Deepfake_Image_Classification

## Future Enhancements
1. Integrate **attention mechanisms** to focus on critical deepfake artifacts.
2. Apply the model in real-time scenarios like **online meetings and exam proctoring** to detect cheating.
3. Incorporate **temporal features** from video sequences for improved detection.
4. Train on larger datasets and explore more advanced models like **EfficientNet-B7**.

## Usage
*(Note: This section is a placeholder. You should add instructions on how to run your code, install dependencies, and use your model once you push the code to the repository.)*

Example:
```bash
# Clone the repository
git clone https://github.com/your-username/deepfake-face-detection.git

# Install dependencies
pip install -r requirements.txt

# Run the training script
python train.py
```

## Acknowledgments
We thank the **DeepFake Detection Challenge (DFDC)** on Kaggle for providing the dataset and **NTI** for the summer training opportunity.

## Contact
For questions or further information, please feel free to contact the team members.

---

Let me know if you'd like to add a specific license, contribution guidelines, or a more detailed code structure section once your code is uploaded
