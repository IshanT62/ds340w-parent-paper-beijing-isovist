# %%
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import shapely
from shapely.geometry import LineString, Point
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor

# %%
def roi_countour(img):
    contour_img = np.zeros_like(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE

    largest_contour_area = 0
    for cnt in contours:
        if (cv2.contourArea(cnt) > largest_contour_area):
            largest_contour_area = cv2.contourArea(cnt)
            largest_contour = cnt

    epsilon = 0.001*cv2.arcLength(largest_contour,True)
    approx = cv2.approxPolyDP(largest_contour,epsilon,True)
    cv2.fillPoly(contour_img, [approx], (255,0,0))
    
    final = cv2.drawContours(img, [approx], 0, [0, 0, 255],2)
    return contour_img, largest_contour, final, approx

# %%
def check_inside(cnt, point):
    return cv2.pointPolygonTest(cnt, point, False) 


# %%
FOLDER_PATH = "./dataset/beijing_selected/" # /home/mdxuser/Desktop/Ph.D/MIT_SCL/floor_plan_extraction/dataset/seoul/
_, _, FLOOR_PLANS = next(os.walk(f"{FOLDER_PATH}"))


# %%
floor_plan_paths = []
for floor_plan in FLOOR_PLANS:
    floor_plan_path = FOLDER_PATH + floor_plan
    floor_plan_paths.append(floor_plan_path)
    
def process_complete(floor_plan_path):  
    img = cv2.imread(f"{floor_plan_path}")
    floor_plan = os.path.basename(floor_plan_path)
    contour_img, roi_contours, drawn_cont, approx = roi_countour(img)
    
    SAMPLED = False
    
    roi_contours_mask = []
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            if check_inside(approx, (i,j)) == 0:
                roi_contours_mask.append([i, j])
    
    _25_percent = int(len(roi_contours_mask)/3) 
    roi_contours_mask = random.sample(roi_contours_mask, _25_percent)

    non_zero_pixels = cv2.findNonZero(contour_img[:,:,0])
    non_zero_pixels_list = []
    for each in non_zero_pixels:
        non_zero_pixels_list.append((each[0][0], each[0][1]))
        
    angle_radians = []
    for i in range(0, 370, 10):
        rad = (i * math.pi)/180
        angle_radians.append(rad)
    
    list_all_lines = []
    length_segment = 1000
    line_img = np.zeros_like(img)
    if SAMPLED:
        roi_contours_mask = roi_contours
        
    for pnt in roi_contours_mask: #instead of roi_contours
        #print(pnt[0])
        if SAMPLED:
            x1, y1 = pnt[0][0], pnt[0][1]
        else:
            x1, y1 = pnt[0], pnt[1]
        A = (x1, y1)
        for angle in angle_radians:
            x2 = int(x1 + length_segment*math.cos(angle))
            y2 = int(y1 + length_segment*math.sin(angle))
            B = (x2, y2)
            list_all_lines.append([A, B])
            #cv2.line(img, (x1, y1), (x2, y2), (0,0, 255), 1) 
            cv2.line(line_img, (x1, y1), (x2, y2), 255, 2)
            #result = cv2.pointPolygonTest(cnt, (pnt[0][0], pnt[0][1]), False)
    contour_line_img = cv2.bitwise_and(line_img, contour_img)
    heat_map_arr = np.zeros_like(img)
    
    
    dict_int = {}
    for i in range(len(list_all_lines)):
        print(i, len(list_all_lines))
        for j in range(len(list_all_lines)):
            if i != j:
                line1 = LineString([list_all_lines[i][0], list_all_lines[i][1]])
                line2 = LineString([list_all_lines[j][0], list_all_lines[j][1]])
                int_pt = line1.intersection(line2)
                try:
                    point_of_intersection = int(int_pt.x), int(int_pt.y)
                    if point_of_intersection in dict_int.keys():
                        dict_int[point_of_intersection] += 1
                    else:
                        dict_int[point_of_intersection] = 1
                except Exception as e:
                    pass
                
    heat_map_arr = np.zeros_like(img)
    filtered_dict_int = {}
    for pixels in non_zero_pixels_list:
        if pixels in dict_int.keys():
            heat_map_arr[pixels[1], pixels[0]] = dict_int[pixels]
            filtered_dict_int[pixels] = dict_int[pixels]
    plt.axis('off')
    sns.heatmap(heat_map_arr[:,:,0], cbar=False)
    plt.savefig(f"./results_hmap_all_nt/{floor_plan.split('.jpg')[0]}.png", bbox_inches='tight', pad_inches = 0, dpi=201.5)
    
with ProcessPoolExecutor(max_workers = 10) as executor:
    executor.map(process_complete, floor_plan_paths)