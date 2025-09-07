#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import shutil

from skimage import io, transform
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


#Create a folder containing the target heatmaps, organized by the building's construction year


# In[ ]:


# Set the path to the directory containing the subfolders with image files
folder_path = '/put/the/path/where/you/save/all/the/IIS/heatmap/'

# Set the DataFrame column name containing the image filenames
filename_column = 'filename'

# Set the path to the new folder to copy the image files
new_folder_path = '/creat/a/new/path/for/re-saving/selected/only/images/'

# Read the DataFrame from a CSV file or create it programmatically


# Get the list of image filenames from the DataFrame column
image_filenames = age00s[filename_column].tolist()

# Create the new folder if it doesn't exist
os.makedirs(new_folder_path, exist_ok=True)

# Loop through the subfolders in the given directory
for root, dirs, files in os.walk(folder_path):
    # Loop through the files in the current subfolder
    for file in files:
        # Check if the filename is in the list of image filenames
        if file in image_filenames:
            # Construct the source path to the image file
            source_path = os.path.join(root, file)
            
            # Construct the destination path to the new folder
            destination_path = os.path.join(new_folder_path, file)
            
            # Copy the image file to the new folder
            shutil.copyfile(source_path, destination_path)

# Print a message indicating the successful copying of image files
print("Image files have been successfully copied to the new folder.")


# In[ ]:


#run the code to generate the specific year period folder, which will be created beforehand.


# In[ ]:


# Set the path to the folder containing the images
folder_path = '/Users/yaoyanhua/Documents/Urban_housing_layout/final_results/shanghai/00s'

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False, input_shape=(None, None, 3))

# Initialize lists to store image data and filenames
image_data = []
filenames = []

# Loop through the images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load and preprocess the image
        image_path = os.path.join(folder_path, filename)
        image = io.imread(image_path)
        image = preprocess_input(image)
        
        # Resize the image to a compatible shape
        image = transform.resize(image, (224, 224, 3))
        
        # Append the image and filename to the lists
        image_data.append(image)
        filenames.append(filename)

# Convert the image data list to a numpy array
image_data = np.array(image_data)

# Extract features using the VGG16 model
features = model.predict(image_data)

# Reshape the features array
features = features.reshape(features.shape[0], -1)

# Compute pairwise similarity between the feature vectors
similarity_matrix = cosine_similarity(features)

# Calculate the representativeness scores using the PageRank algorithm
graph = nx.from_numpy_array(similarity_matrix)
pagerank_scores = nx.pagerank(graph)

# Sort the images based on their representativeness scores in descending order
sorted_indices = sorted(range(len(pagerank_scores)), key=pagerank_scores.__getitem__, reverse=True)

# Select the top 5 most representative images
top_5_indices = sorted_indices[:5]
top_5_filenames = [filenames[i] for i in top_5_indices]

# Display the top 5 most representative images
for filename in top_5_filenames:
    image_path = os.path.join(folder_path, filename)
    image = io.imread(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Display the filenames of the top 5 most representative images
print("Top 5 Most Representative Images:")
for filename in top_5_filenames:
    print(filename)


# In[ ]:


#根据模型所识别出的最具典型性的IIS热力图查找其原始平面图，并进行定性分析与讨论。

