import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
def show_differences(real_img_path, fake_img_path):
    real_img = cv2.imread(real_img_path)
    fake_img = cv2.imread(fake_img_path)

    real_gray = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
    fake_gray = cv2.cvtColor(fake_img, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(real_gray, fake_gray)

    cv2.imshow('Real Image', real_img)
    cv2.imshow('Fake Image', fake_img)
    cv2.imshow('Absolute Difference', diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

real_image_path = 'fr2.jpeg'
fake_image_path = 'fr1.jpeg'
show_differences(real_image_path, fake_image_path)
# Function to load images from folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize image to 128x128 pixels
            images.append(img)
    return images

# Load real and AI-generated images
real_images = cv2.imread("fr2.jpeg")
ai_images = cv2.imread("fr1.jpeg")

# Create labels: 0 for real images, 1 for AI-generated images
real_labels = np.zeros(len(real_images))
ai_labels = np.ones(len(ai_images))

# Combine images and labels
all_images = np.concatenate((real_images, ai_images), axis=0)
all_labels = np.concatenate((real_labels, ai_labels), axis=0)

# Flatten images
all_images_flat = np.array([img.flatten() for img in all_images])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_images_flat, all_labels, test_size=0.2, random_state=42)

# Create SVM classifier
svm_classifier = SVC(kernel='linear')

# Train SVM classifier
svm_classifier.fit(X_train, y_train)

# Predict labels for test set
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
