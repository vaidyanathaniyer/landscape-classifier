# Landscape Classifier 🌿🛰️

## Table of Contents
- [Problem Statement](#problem-statement) 🎯
- [Project Goals](#project-goals) 🌟
- [Project Objectives and Scope](#project-objectives-and-scope) 🔍
- [Working of the Model](#working-of-the-model) 💻
- [Resources Used](#resources-used) 📚
- [Action of Completion](#action-of-completion) ✅
- [Potential Limitations and Stay Back](#potential-limitations-and-stay-back) ⚠️
- [Summary](#summary) 📝

## Problem Statement
Implementation of a Machine Learning Model for Landscape Classification.

## Project Goals
- Develop a machine learning model capable of accurately classifying different landcover types in satellite images.
- Create a Command-Line Interface (CLI) for image processing and analysis.
- Utilize the UCMerced_LandUse dataset for training and validation.
- Generate an output image with highlighted areas representing various landcover types.
- Demonstrate the potential application of the technology in aviation for real-time analysis of topography during flight.

## Project Objectives and Scope
- Train a Convolutional Neural Network (CNN) model using the provided dataset.
- Implement a CLI for users to input satellite images for analysis.
- Classify landcover types such as water, sand, forest, wetland, roadways, snow/ice, and manmade areas.
- Demonstrate the potential application of the technology in the aviation industry for aerial scanning and real-time topography analysis during flight.

## Working of the Model
1. **Dataset Preparation**:
   - The UCMerced_LandUse dataset is utilized for training and validation.
   - Images are loaded and preprocessed, including resizing and normalization.
   
2. **Model Architecture**:
   - A CNN model is created using Keras with layers for convolution, activation, pooling, flattening, and dense.
   - The model is compiled with categorical cross-entropy loss and Adam optimizer.

3. **Training**:
   - The model is trained on the dataset with 21 landcover classes.
   - Training is performed for 50 epochs with a batch size of 64.

4. **Image Prediction**:
   - The CLI accepts an input image path.
   - Preprocessing is applied to the input image.
   - The trained model predicts the class probabilities for the input image.

5. **Output Generation**:
   - The predicted class probabilities are used to generate an output image.
   - The output image highlights different landcover types using color-coded sections.

## Resources Used
### Python Libraries:
- scikit-learn
- Keras
- OpenCV
- PIL
- NumPy
- argparse

### Dataset:
- [UCMerced_LandUse](http://weegee.vision.ucmerced.edu/datasets/landuse.html)

## Action of Completion
- Developed a CLI-based landscape classifier using a CNN model.
- Successfully trained the model on the UCMerced_LandUse dataset.
- Implemented preprocessing, prediction, and output generation functionalities.
- Demonstrated the capability to classify various landcover types in satellite images.

## Potential Limitations and Stay Back
- **Dataset Size**: Limited dataset size may affect the model's ability to generalize to unseen data.
- **Model Performance**: The model's performance may vary depending on the complexity of the landscape and quality of satellite images.
- **Real-time Analysis**: Implementation in real-time applications may require optimization for efficiency and speed.
- **Aviation Integration**: Integration into aviation systems would require rigorous testing and compliance with safety regulations.

## Summary
The landscape classifier project aims to develop a machine learning model capable of accurately classifying landcover types in satellite images. By leveraging a CNN architecture and the UCMerced_LandUse dataset, the model achieves reliable classification results. The CLI interface allows users to input satellite images for analysis, making it suitable for various applications, including aviation. The technology holds immense potential for revolutionizing aerial scanning and real-time topography analysis during flight, paving the way for advancements in the aviation industry.
