import cv2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 128))  # Resize the image to a smaller size
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(img)
    return hog_features.flatten()


# Path to the folder containing training images
folder_path = './train'

# Lists to store image paths and corresponding features
image_paths = []
image_features = []

# Iterate over images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Get image path
        image_path = os.path.join(folder_path, filename)
        # Extract features and append to the lists
        features = extract_features(image_path)
        image_paths.append(image_path)
        image_features.append(features)

# Convert list to numpy array
image_features = np.array(image_features)


def match_image(captured_img_path, threshold=0.72):
    # Extract features from captured image
    captured_features = extract_features(captured_img_path).reshape(1, -1)
    # Initialize maximum similarity and index
    max_similarity = -1
    max_index = -1
    # Iterate over features of images in the folder
    for i, features in enumerate(image_features):
        features = features.reshape(1, -1)
        # Compute cosine similarity between features
        similarity = cosine_similarity(features, captured_features)[0][0]
        # Update maximum similarity and index if current similarity is larger
        if similarity > max_similarity:
            max_similarity = similarity
            max_index = i
    # Check if a match is found above the threshold
    if max_similarity >= threshold:
        matched_image_path = image_paths[max_index]
        return True, matched_image_path, max_similarity
    else:
        return False, None, max_similarity


def program3(i):
    captured_img_path = './captured/' + str(i) + '.jpg'
    # Match the captured image with images in the folder
    match_found, matched_image_path, similarity = match_image(captured_img_path)
    if match_found:
        print(f'Matched image: {matched_image_path} with similarity: {similarity}')
        # Display both images
        captured_img = cv2.imread(captured_img_path)
        matched_img = cv2.imread(matched_image_path)

        # Resize images to the same size for side-by-side display
        captured_img = cv2.resize(captured_img, (300, 300))
        matched_img = cv2.resize(matched_img, (300, 300))

        # Concatenate images horizontally
        combined_img = np.hstack((captured_img, matched_img))

        # Display the concatenated image
        cv2.imshow('Captured Image (Left) and Matched Image (Right)', combined_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return True
    else:
        print(f"No match found. Highest similarity: {similarity}")
        return False

# Example usage
# result = program3(1)
# print("Match result:", result)
