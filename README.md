# Vision Assignment: Image Inpainting

## Overview
This project focuses on face recognition and image inpainting technology. It uses a Convolutional Neural Network (CNN) for face recognition and a Stable Diffusion model for image inpainting to obscure facial features for privacy protection.


## Data Source
The data for training the face recognition model was obtained from the Labeled Faces in the Wild (LFW) dataset. The LFW dataset contains over 13,000 images of faces collected from the web, each labeled with the name of the person pictured. The dataset can be accessed (http://vis-www.cs.umass.edu/lfw/).

## Model and Data Justification
Convolutional Neural Network (CNN) was chosen for face recognition due to its proven effectiveness in image processing tasks. CNNs are particularly well-suited for this type of data because they can automatically learn spatial hierarchies of features from images, which is important for recognizing facial patterns. Additionally, the model used a PCA-SVM pipeline as an alternative approach, applying PCA for dimensionality reduction and SVM for classification, which is known for its robustness in high-dimensional spaces.

## Commented Examples
- **Input**: An uploaded image containing a face.
- **Output**: An inpainted image with the detected face area modified by the model.
- **Observation**: The model successfully detects and inpaints the face area, providing a clear visualization of the changes. The results are as expected, with the inpainting model generating plausible alterations to the detected face area.

## Testing
The confusion matrix and classification report provide insights into the model's performance:

### Classification Report

| Class             | Precision | Recall | F1-Score | Support |
|-------------------|-----------|--------|----------|---------|
| Ariel Sharon      | 0.78      | 0.54   | 0.64     | 13      |
| Colin Powell      | 0.80      | 0.88   | 0.84     | 60      |
| Donald Rumsfeld   | 0.94      | 0.63   | 0.76     | 27      |
| George W Bush     | 0.85      | 0.98   | 0.91     | 146     |
| Gerhard Schroeder | 0.95      | 0.80   | 0.87     | 25      |
| Hugo Chavez       | 1.00      | 0.47   | 0.64     | 15      |
| Tony Blair        | 0.94      | 0.83   | 0.88     | 36      |

### Overall Metrics

- **Accuracy**: 0.86
- **Macro Average**: Precision 0.89, Recall 0.73, F1-Score 0.79
- **Weighted Average**: Precision 0.87, Recall 0.86, F1-Score 0.85

The confusion matrix shows that the model performs particularly well in identifying "George W Bush" with a recall of 98%, but has more difficulty with "Hugo Chavez", indicated by a recall of 47%. Overall, the model demonstrates strong performance, though some classes could benefit from further refinement to improve identification accuracy.


## Code and Instructions to Run It
The code can be found in Image_Inpaint_System.py.


