from keras.utils import img_to_array 
from keras.applications.vgg16 import preprocess_input 
from keras.utils import load_img 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle

from ultralytics import YOLO
import os
import cv2
import math

# %%
for CITY in ["beijing"]:
    print(CITY)
    for unique_labels in [5, 8, 12, 18, 25, 30]:
        print(unique_labels)
        df_list = pd.read_csv(f"/home/mdxuser/Desktop/Ph.D/MIT_SCL/floor_plan_extraction/results/{CITY}/{CITY}_results.csv")

        df_list_property_id = df_list["propertyID"].tolist()

        path = rf"/home/mdxuser/Desktop/Ph.D/MIT_SCL/floor_plan_extraction/results/{CITY}/results_hmap_all_nt/"
        # change the working directory to the path where the images are located
        os.chdir(path)

        # this list holds all the image filename
        hmaps = []

        # creates a ScandirIterator aliased as files
        with os.scandir(path) as files:
        # loops through each file in the directory
            for file in files:
                if file.name.endswith('.png'):
                # adds only the image files to the flowers list
                    if int(file.name.split('_')[0]) in df_list_property_id:
                        hmaps.append(file.name)

        # %%
        len(hmaps)


        # %%
        p = rf"/home/mdxuser/Desktop/Ph.D/MIT_SCL/floor_plan_extraction/results/{CITY}/results_hmap_all_nt/"


        with open(f"/home/mdxuser/Desktop/Ph.D/MIT_SCL/floor_plan_extraction/results/{CITY}/features_15.pkl", 'rb') as f:
            data = pickle.load(f)

        # %%
        # get a list of the filenames
        filenames = np.array(list(data.keys()))

        # get a list of just the features
        feat = np.array(list(data.values()))

        # %%
        feat = feat.reshape(-1, 4096)
        # reduce the amount of dimensions in the feature vector
        pca = PCA(n_components=18, random_state=22)

        pca.fit(feat)
        x_visual = pca.transform(feat)

        # %%
        x_visual.shape

        # %%
        feature_np = np.zeros((x_visual.shape[0], 1, 18))

        # %%
        feature_np.shape

        # %%
        for idx, vis_feat in enumerate(x_visual):

            # living_room_mean_dist = df_list[df_list["propertyID"] == int(hmaps[idx].split('_')[0])]["living_room_mean_dist"].iloc[0]
            # bedroom_mean_dist = df_list[df_list["propertyID"] == int(hmaps[idx].split('_')[0])]["bedroom_mean_dist"].iloc[0]
            # kitchen_mean_dist = df_list[df_list["propertyID"] == int(hmaps[idx].split('_')[0])]["kitchen_mean_dist"].iloc[0]
            # balcony_mean_dist = df_list[df_list["propertyID"] == int(hmaps[idx].split('_')[0])]["balcony_mean_dist"].iloc[0]
            # bathroom_mean_dist = df_list[df_list["propertyID"] == int(hmaps[idx].split('_')[0])]["bathroom_mean_dist"].iloc[0]
            # dining_romm_mean_dist = df_list[df_list["propertyID"] == int(hmaps[idx].split('_')[0])]["dining_romm_mean_dist"].iloc[0]
            
            
            feature_np[idx, 0, :18] = x_visual[idx]
            # feature_np[idx, 0, 12] = living_room_mean_dist
            # feature_np[idx, 0, 13] = bedroom_mean_dist
            # feature_np[idx, 0, 14] = kitchen_mean_dist
            # feature_np[idx, 0, 15] = balcony_mean_dist
            # feature_np[idx, 0, 16] = bathroom_mean_dist
            # feature_np[idx, 0, 17] = dining_romm_mean_dist

        # %%
        feat = feature_np.reshape(-1, 18)
        # reduce the amount of dimensions in the feature vector
        pca = PCA(n_components=10, random_state=22)

        # %%
        pca

        # %%
        pca.fit(feat)
        x = pca.transform(feat)

        # %%
        x

        # %%
        x.shape

        # %%
        # cluster feature vectors
        kmeans = KMeans(n_clusters=unique_labels, random_state=22)
        kmeans.fit(x)

        # %%
        from sklearn.metrics import pairwise_distances_argmin_min

        # %%
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, x)

        # %%
        # holds the cluster id and the images { id: [images] }
        groups = {}

        for file, cluster in zip(filenames,kmeans.labels_):
            #print(file)
            if cluster not in groups.keys():
                groups[cluster] = []
                groups[cluster].append(file)
            else:
                groups[cluster].append(file)

        # %%
        idx = 0
        for key in groups:
            #os.makedirs(f"{CLUSTER_PATH}{key}", exist_ok=True)
            files = groups[key]
            for file in files:
                #print(int(file.split('.')[0].split('_')[0]))
                #print(df_list[df_list['propertyID'] == int(file.split('.')[0].split('_')[0])].index)
                for index in df_list[df_list['propertyID'] == int(file.split('.')[0].split('_')[0])].index:
                    df_list.at[index, "cluster"] = key

        # %%
        df_list

        # %%
        CLUSTER_PATH = f"/home/mdxuser/Desktop/Ph.D/MIT_SCL/floor_plan_extraction/results/{CITY}/clustering/{unique_labels}_clusters/"
        os.makedirs(f"{CLUSTER_PATH}", exist_ok=True)

        # %%
        df_list.to_csv(f"/home/mdxuser/Desktop/Ph.D/MIT_SCL/floor_plan_extraction/results/{CITY}/clustering/{unique_labels}_clusters/cluster_labeled.csv", index=False, sep=",")

        # %%
        year_range_df = pd.DataFrame()

        # %%
        import shutil
        BASE_PATH = f"/home/mdxuser/Desktop/Ph.D/MIT_SCL/floor_plan_extraction/results/{CITY}/results_hmap_all_nt/"
        CLUSTER_PATH = f"/home/mdxuser/Desktop/Ph.D/MIT_SCL/floor_plan_extraction/results/{CITY}/clustering/{unique_labels}_clusters/"
        count = 0

        idx = 0
        for key in groups:
            print(key)
            os.makedirs(f"{CLUSTER_PATH}{key}", exist_ok=True)
            files = groups[key]
            for file in files:
                print(int(file.split('.')[0].split('_')[0]))
                try:
                    year = df_list[df_list["propertyID"] == int(file.split('.')[0].split('_')[0])]["built_year"]
                    year = year.iloc[0]
                    year_range_df.at[idx, key] = year
                except Exception as e:
                    year = "UNK"
                    count += 1
                idx += 1
                shutil.copyfile(f"{BASE_PATH}{file}", f"{CLUSTER_PATH}{key}/{file.split('_')[0]}_{year}.png")

        # %%
        MOST_REPR = True
        if MOST_REPR:
            idx = 0
            import shutil
            for file, cluster in zip(filenames,kmeans.labels_):
                
                if idx in closest:
                    print(idx, file, cluster)
                    shutil.copyfile(f"{BASE_PATH}{file}", f"{CLUSTER_PATH}/{file.split('_')[0]}_{cluster}.png")
                idx += 1

        # %%
        year_range_df.to_csv(f"/home/mdxuser/Desktop/Ph.D/MIT_SCL/floor_plan_extraction/results/{CITY}/clustering/{unique_labels}_clusters/cluster_year_range.csv", index=False, sep=",")
        del feature_np
    
