import numpy as np
import pandas as pd
import cv2
import os
import sys

if len(sys.argv) != 4:
    print("Usage: python3 main.py <part_id> <input_dir> <output_dir>")
    sys.exit(1)

part_id = int(sys.argv[1])
input_dir = sys.argv[2]
output_dir = sys.argv[3]


unit_kernel = np.ones((3,3))

dilate_kernel = np.ones((1,5))

erode_kernel = np.ones((3,1))

uniform_kernel=np.ones((3,3))/9

gaussian_kernel=np.array([[1,2,1],[2,4,2],[1,2,1]])/16

# vertical edges
# sobelx_kernel=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])/8

# horizontal edges
# sobely_kernel=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])/8

# laplacian_kernel=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])


def convolution(image, kernel):

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    output_height = image_height
    output_width = image_width
    
    pad_width = (kernel_width - 1)//2

    image = np.pad(image, ((pad_width, pad_width), (pad_width, pad_width)), mode='constant', constant_values=255)
    
    kernel = np.fliplr(np.flipud(kernel))
    
    result = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            result[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return result


def dilate(image, kernel):

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    output_height = image_height
    output_width = image_width
    
    pad_height = (kernel_height - 1)//2
    pad_width = (kernel_width - 1)//2

    image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=255)

    result = np.zeros((output_height, output_width))
    
#     kernel = np.fliplr(np.flipud(kernel))

    for i in range(output_height):
        for j in range(output_width):
            result[i, j] = np.max(image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return result


def erode(image, kernel):
    
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    output_height = image_height
    output_width = image_width
    
    pad_height = (kernel_height - 1)//2
    pad_width = (kernel_width - 1)//2

    image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=255)
    
    result = np.zeros((output_height, output_width))
    
#     kernel = np.fliplr(np.flipud(kernel))

    for i in range(output_height):
        for j in range(output_width):
            result[i, j] = np.min(image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return result


def find_max_min_2d_array(arr):

    max_element = arr[0][0]
    min_element = arr[0][0]

    for row in arr:
        for element in row:
            if element > max_element:
                max_element = element
            elif element < min_element:
                min_element = element

    return max_element, min_element


def find_connected_components(image):

    rows, cols = len(image), len(image[0])
    visited = [[False] * cols for _ in range(rows)]
    clusters = []
    cluster_label = 1

    for i in range(rows):
        for j in range(cols):
            if not visited[i][j] and image[i][j] == 0:
                current_cluster = []
                stack = [(i, j)]

                while stack:
                    row, col = stack.pop()
                    if 0 <= row < rows and 0 <= col < cols and not visited[row][col] and image[row][col] == 0:
                        visited[row][col] = True
                        current_cluster.append((row, col))
                        
                        stack.append((row - 1, col))  # Up
                        stack.append((row + 1, col))  # Down
                        stack.append((row, col - 1))  # Left
                        stack.append((row, col + 1))  # Right

                clusters.append((cluster_label, current_cluster))
                cluster_label += 1

    return clusters


def filter_and_set_threshold(image, clusters, threshold):

    clusters_to_remove = []

    for cluster_label, cluster_pixels in clusters:
        if len(cluster_pixels) < threshold:
            for row, col in cluster_pixels:
                image[row][col] = 255
            clusters_to_remove.append(cluster_label)

    new_clusters = [(label, pixels) for label, pixels in clusters if label not in clusters_to_remove]

    return image, new_clusters


def find_mean_median_cluster_size(clusters):

    cluster_sizes = [len(cluster_pixels) for _, cluster_pixels in clusters]
    cluster_sizes.sort()
    cluster_sizes = np.array(cluster_sizes)
    mean = np.mean(cluster_sizes)
    median = np.median(cluster_sizes)

    return mean, median
                

def find_cluster_points(clusters):

    centroids = []
    leftmost_pixels = []
    rightmost_pixels = []

    for _, cluster_pixels in clusters:
        
        if len(cluster_pixels) > 0:
            
            sorted_pixels = sorted(cluster_pixels, key=lambda pixel: pixel[1])
            
            leftmost_pixel = sorted_pixels[0]
            leftmost_pixels.append(leftmost_pixel)
            
            avg_row = int(sum(row for row, _ in cluster_pixels) / len(cluster_pixels))
            avg_col = int(sum(col for _, col in cluster_pixels) / len(cluster_pixels))
            centroids.append((avg_row, avg_col))
            
            rightmost_pixel = sorted_pixels[-1]
            rightmost_pixels.append(rightmost_pixel)

    return leftmost_pixels, centroids, rightmost_pixels


def print_cluster_parameters(clusters):

    print(f"No. of Clusters: {len(clusters)}")

    for label, pixels in clusters:
        cluster_size = len(pixels)
        print(f"Cluster {label}: Size {cluster_size}")


def find_parameters(image_path):
    
    bgr_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    blurred_image = convolution(gray_image, gaussian_kernel)
#     blurred_image = convolution(blurred_image, gaussian_kernel)
    
    max_pixel_value, min_pixel_value = find_max_min_2d_array(blurred_image)
    average_pixel_value = (float(max_pixel_value) + float(min_pixel_value))/2
    
    blurred_image[blurred_image >= average_pixel_value] = 255
    blurred_image[blurred_image < average_pixel_value] = 0    
    binary_image = blurred_image
    
    eroded_dilated_image = erode(binary_image, erode_kernel)
    eroded_dilated_image = dilate(eroded_dilated_image, dilate_kernel)
    eroded_dilated_image = dilate(eroded_dilated_image, dilate_kernel)
    eroded_dilated_image = dilate(eroded_dilated_image, dilate_kernel)
    eroded_dilated_image = erode(eroded_dilated_image, erode_kernel)
    eroded_dilated_image = dilate(eroded_dilated_image, np.ones((3,3)))
    
    clusters = find_connected_components(eroded_dilated_image)
#     print_cluster_parameters (clusters)
    
    mean_cluster_size, median_cluster_size = find_mean_median_cluster_size(clusters)
#     print('Mean =', mean_cluster_size)
#     print('Median =', median_cluster_size)

    cluster_image, new_clusters = filter_and_set_threshold(eroded_dilated_image, clusters, mean_cluster_size/2)
#     print_cluster_parameters (new_clusters)

    num_sutures = len(new_clusters)
#     print(num_sutures)
    
    leftmost_pixels, centroids, rightmost_pixels = find_cluster_points(new_clusters)
#     print(leftmost_pixels)
#     print(centroids)
#     print(rightmost_pixels)
    
    inter_suture_distance = []
    for i in range(len(new_clusters)-1):
        cent1_row, cent1_column = centroids[i]
        cent2_row, cent2_column = centroids[i+1]
        distance = (((cent1_row - cent2_row)**2 + (cent1_column - cent2_column)**2)**0.5)/len(cluster_image)
        inter_suture_distance.append(distance)
    
    inter_suture_distance = np.array(inter_suture_distance)
    # print(inter_suture_distance)
    mean_distance = np.mean(inter_suture_distance)
    variance_distance = np.var(inter_suture_distance)
    
    suture_angle = []
    for i in range(len(new_clusters)):
        left = leftmost_pixels[i]
        right = rightmost_pixels[i]        
        dx = right[1] - left[1]
#         print(dx)
        if dx == 0:
            dx = 1
        dy = left[0] - right[0]
#         print(dy)
        dy_dx = dy/dx
        angle = np.degrees(np.arctan(dy_dx))
        suture_angle.append(angle)
        
    suture_angle = np.array(suture_angle)
    # print(suture_angle)
    mean_angle = np.mean(suture_angle)
    variance_angle = np.var(suture_angle)
    
    cluster_points_image = np.zeros((cluster_image.shape[0], cluster_image.shape[1], 3), dtype=np.uint8)
    cluster_points_image[cluster_image == 0] = [0, 0, 0]
    cluster_points_image[cluster_image == 255] = [255, 255, 255]

    cluster_centroid_image = np.zeros((cluster_image.shape[0], cluster_image.shape[1], 3), dtype=np.uint8)
    cluster_centroid_image[cluster_image == 0] = [0, 0, 0]
    cluster_centroid_image[cluster_image == 255] = [255, 255, 255]

    cluster_extremes_image = np.zeros((cluster_image.shape[0], cluster_image.shape[1], 3), dtype=np.uint8)
    cluster_extremes_image[cluster_image == 0] = [0, 0, 0]
    cluster_extremes_image[cluster_image == 255] = [255, 255, 255]
    
    rgb_suture_distance = rgb_image.copy()
    rgb_suture_angle = rgb_image.copy() 
    
    for i in range(len(new_clusters)):
        pixel1 = (leftmost_pixels[i][1], leftmost_pixels[i][0])
        pixel2 = (centroids[i][1], centroids[i][0])
        pixel3 = (rightmost_pixels[i][1], rightmost_pixels[i][0])
#         cv2.line(cluster_points_image, pixel1, pixel2, (255, 255, 0), 2)
#         cv2.line(cluster_points_image, pixel2, pixel3, (255, 255, 0), 2)
#         cv2.line(rgb_suture_distance, pixel1, pixel2, (255, 255, 0), 2)
#         cv2.line(rgb_suture_distance, pixel2, pixel3, (255, 255, 0), 2)
        cv2.line(rgb_suture_angle, pixel1, pixel3, (255, 255, 0), 2)
    
    for i in range(len(new_clusters)-1):
        pixel1 = (centroids[i][1], centroids[i][0])
        pixel2 = (centroids[i+1][1], centroids[i+1][0])
#         cv2.line(cluster_points_image, pixel1, pixel2, (0, 0, 255), 1)
        cv2.line(rgb_suture_distance, pixel1, pixel2, (0, 0, 255), 1)
    
    for leftmost_pixel in leftmost_pixels:
        row, col = leftmost_pixel
        cv2.circle(cluster_points_image, (col, row), 3, (255, 0, 0), -1)
        cv2.circle(cluster_extremes_image, (col, row), 3, (255, 0, 0), -1)
#         cv2.circle(rgb_suture_distance, (col, row), 3, (255, 0, 0), -1)
        cv2.circle(rgb_suture_angle, (col, row), 3, (255, 0, 0), -1)
        
    for centroid in centroids:
        row, col = centroid
        cv2.circle(cluster_points_image, (col, row), 3, (255, 0, 0), -1)
        cv2.circle(cluster_centroid_image, (col, row), 3, (255, 0, 0), -1)
        cv2.circle(rgb_suture_distance, (col, row), 3, (255, 0, 0), -1)
#         cv2.circle(rgb_suture_angle, (col, row), 3, (255, 0, 0), -1)

    for rightmost_pixel in rightmost_pixels:
        row, col = rightmost_pixel
        cv2.circle(cluster_points_image, (col, row), 3, (255, 0, 0), -1)
        cv2.circle(cluster_extremes_image, (col, row), 3, (255, 0, 0), -1)
#         cv2.circle(rgb_suture_distance, (col, row), 3, (255, 0, 0), -1)
        cv2.circle(rgb_suture_angle, (col, row), 3, (255, 0, 0), -1)
     
    if not os.path.exists('part1_images'):
        os.makedirs('part1_images')
        
    base_filename, file_extension = os.path.splitext(os.path.basename(image_path))
    
    output_filename = f"part1_images/{base_filename}_gray.png"
    cv2.imwrite(output_filename, gray_image)
    
    output_filename = f"part1_images/{base_filename}_binary.png"
    cv2.imwrite(output_filename, binary_image)
    
    output_filename = f"part1_images/{base_filename}_eroded_dilated.png"
    cv2.imwrite(output_filename, eroded_dilated_image)
    
    output_filename = f"part1_images/{base_filename}_cluster.png"
    cv2.imwrite(output_filename, cluster_image)
    
    output_filename = f"part1_images/{base_filename}_cluster_points.png"
    cluster_points_image = cv2.cvtColor(cluster_points_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_filename, cluster_points_image)

    output_filename = f"part1_images/{base_filename}_cluster_centroid.png"
    cluster_centroid_image = cv2.cvtColor(cluster_centroid_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_filename, cluster_centroid_image)

    output_filename = f"part1_images/{base_filename}_cluster_extremes.png"
    cluster_extremes_image = cv2.cvtColor(cluster_extremes_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_filename, cluster_extremes_image)
    
    output_filename = f"part1_images/{base_filename}_rgb_suture_distance.png"
    rgb_suture_distance = cv2.cvtColor(rgb_suture_distance, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_filename, rgb_suture_distance)
    
    output_filename = f"part1_images/{base_filename}_rgb_suture_angle.png"
    rgb_suture_angle = cv2.cvtColor(rgb_suture_angle, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_filename, rgb_suture_angle)
    
    return num_sutures, mean_distance, variance_distance, mean_angle, variance_angle


if part_id == 1:

    output_dict = {
        'image_name': [],
        'number of sutures': [],
        'mean inter suture spacing': [],
        'variance of inter suture spacing': [],
        'mean suture angle wrt x-axis': [],
        'variance of suture angle wrt x-axis': []
    }

    folder_path = input_dir
    file_list = os.listdir(folder_path)

    for file_name in file_list:
        print(file_name)
        output_dict['image_name'].append(file_name)
        image_path = os.path.join(folder_path, file_name)
        
        num_sutures, mean_distance, variance_distance, mean_angle, variance_angle = find_parameters(image_path)
        # print(num_sutures, round(mean_distance,6), round(variance_distance,6), round(mean_angle,6), round(variance_angle,6))
        
        output_dict['number of sutures'].append(num_sutures)
        output_dict['mean inter suture spacing'].append(mean_distance)
        output_dict['variance of inter suture spacing'].append(variance_distance)
        output_dict['mean suture angle wrt x-axis'].append(mean_angle)
        output_dict['variance of suture angle wrt x-axis'].append(variance_angle) 
        
    output_df = pd.DataFrame(output_dict)
    output_df.to_csv(output_dir, index=False, float_format='%.6f')


elif part_id == 2:

    output_dict = {
        'img1_path': [],
        'img2_path': [],
        'output_distance': [],
        'output_angle': [],
    }

    file_path = input_dir
    input_df = pd.read_csv(file_path)
    input_df_array = input_df.values
    entry_no = 1

    for row in input_df_array:

        print("Entry No. :", entry_no)
        entry_no += 1
        
        img1_path = row[0]
        print(img1_path)
        output_dict['img1_path'].append(img1_path)
        _, _, variance_distance1, _, variance_angle1 = find_parameters(img1_path)
        
        img2_path = row[1]
        print(img2_path)
        output_dict['img2_path'].append(img2_path)
        _, _, variance_distance2, _, variance_angle2 = find_parameters(img2_path)
        
        if variance_distance1 <= variance_distance2:
            output_dict['output_distance'].append(1)
        else:
            output_dict['output_distance'].append(2)
            
        if variance_angle1 <= variance_angle2:
            output_dict['output_angle'].append(1)
        else:
            output_dict['output_angle'].append(2)

    output_df = pd.DataFrame(output_dict)
    output_df.to_csv(output_dir, index=False)