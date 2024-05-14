import cv2
import numpy as np
import gradio as gr
import logging
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import torch
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion Inpainting model from Diffusers
pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting").to(device)

# Load LFW dataset
def load_data():
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    X = lfw_people.images
    y = lfw_people.target
    target_names = lfw_people.target_names
    num_classes = target_names.shape[0]
    return X, y, num_classes, target_names

X, y, num_classes, target_names = load_data()

# Prepare data for CNN
X_cnn = X / 255.0
X_cnn = X_cnn.reshape(X_cnn.shape[0], X_cnn.shape[1], X_cnn.shape[2], 1)
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y, test_size=0.2, random_state=42)

# Build and train the CNN model
model = Sequential([
    Flatten(input_shape=(X_cnn.shape[1], X_cnn.shape[2])),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_cnn, y_train_cnn, epochs=10, validation_data=(X_test_cnn, y_test_cnn))

# Prepare data for SVM with PCA
X_pca = X.reshape(X.shape[0], -1)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.25, random_state=42)

# Compute PCA
n_components = 150
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train_pca)
X_train_pca = pca.transform(X_train_pca)
X_test_pca = pca.transform(X_test_pca)
eigenfaces = pca.components_.reshape((n_components, X.shape[1], X.shape[2]))

# Train SVM with GridSearch
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train_pca)

# Face detection function
def detect_faces(image, classifier_path='haarcascade_frontalface_default.xml'):
    face_cascade = cv2.CascadeClassifier(classifier_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    return faces, gray_image

def inpaint_image(image):
    # Resize the image to reduce processing time
    image = cv2.resize(image, (512, 512))

    faces, gray_image = detect_faces(image)
    if len(faces) == 0:
        return "No face detected", None

    x, y, w, h = faces[0]  # Use the first detected face
    mask = np.zeros_like(gray_image)
    mask[y:y+h, x:x+w] = 255

    # Convert images to PIL format
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask)

    # Inpaint the image
    with torch.no_grad():
        result = pipe(prompt="wear a mask", image=image_pil, mask_image=mask_pil).images[0]

    # Convert back to numpy array
    result = np.array(result)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    # Draw rectangle around the face in the original image
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return result, image

# SVM prediction function
def predict_svm(image):
    faces, gray_image = detect_faces(image)
    if len(faces) == 0:
        return "No face detected", None
    
    x, y, w, h = faces[0]  # Use the first detected face
    face_image = cv2.resize(gray_image[y:y+h, x:x+w], (X.shape[1], X.shape[2]))
    
    # SVM prediction
    face_image_pca = face_image.reshape(1, -1)
    face_image_pca = pca.transform(face_image_pca)
    predicted_class_svm = clf.predict(face_image_pca)[0]
    
    # Draw rectangle around the face
    face_image_display = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(face_image_display, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return target_names[predicted_class_svm], face_image_display

# Evaluate SVM performance
y_pred = clf.predict(X_test_pca)

# Confusion matrix
conf_matrix = confusion_matrix(y_test_pca, y_pred)
plt.figure(figsize=(10, 7))
plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names)

# Labeling the plot
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Add text annotations
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.show()

# Classification report
class_report = classification_report(y_test_pca, y_pred, target_names=target_names)
print(class_report)

# Additional metrics
accuracy = accuracy_score(y_test_pca, y_pred)
precision = precision_score(y_test_pca, y_pred, average='weighted')
recall = recall_score(y_test_pca, y_pred, average='weighted')
f1 = f1_score(y_test_pca, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Gradio interface
iface = gr.Interface(
    fn=inpaint_image,
    inputs=gr.Image(type="numpy", label="Upload an image"),
    outputs=[gr.Image(type="numpy", label="Inpainted Image"), gr.Image(type="numpy", label="Original Image with Face Detection")],
    title="Face Inpainting System",
    description="Upload an image to inpaint the detected face using a Stable Diffusion model.",
    live=True
)

iface.queue()
iface.launch(share=True)
