import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import cv2
import os
import math
import random
import sys


if len(sys.argv) != 4:
    print("Usage: python3 main.py <part_id> <input_dir> <output_dir>")
    sys.exit(1)

part_id = int(sys.argv[1])
input_dir = sys.argv[2]
output_dir = sys.argv[3]


def gaussian_kernel(g_size, g_sigma):
    kernel = np.zeros((g_size, g_size))
    g_center = g_size // 2
    for i in range(g_size):
        for j in range(g_size):
            kernel[i, j] = np.exp(-((i - g_center) ** 2 + (j - g_center) ** 2) / (2 * g_sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel


def convolution(image, kernel, padding=True):
    if padding:
        pad_height = kernel.shape[0] // 2
        pad_width = kernel.shape[1] // 2
        image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=255)
#         image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    result_height = image.shape[0]-kernel.shape[0]+1
    result_width = image.shape[1]-kernel.shape[1]+1
    all_windows = np.lib.stride_tricks.sliding_window_view(image, kernel.shape)
    num_windows = result_height*result_width
    windows_matrix = all_windows.reshape(num_windows, -1)
    flattened_kernel = kernel.flatten()
    result = np.dot(windows_matrix, flattened_kernel)
    result = result.reshape(result_height, result_width)
    return result


def get_delta_sigmas(num_scales,sigma):
    delta_sigmas=[sigma]
    for scale_idx in range(num_scales+2):
        delta_sigmas.append(math.sqrt( ( ( 2**( (scale_idx+1.0)/num_scales ) ) * sigma )**2 - (  ( 2**( scale_idx/num_scales ) )*sigma )**2 ) )
    delta_sigmas=np.array(delta_sigmas)
    return delta_sigmas


def upsampling(image, g_size):
    h, w = image.shape
    upsampled_image = np.zeros((h*2, w*2))
    upsampled_image[::2, ::2] = image
    sigma = max(math.sqrt(1.6**2 - 1), math.sqrt(0.01))
    upsampled_image = convolution(upsampled_image, gaussian_kernel(g_size, sigma))
    return upsampled_image


def is_extremum(arr1,arr2,arr3):
    c=arr2[1,1]
    if(np.all(arr1>=c) and np.all(arr3>=c) and arr2[0,0]>=c and arr2[0,1]>=c and arr2[0,2]>=c and arr2[1,0]>=c and arr2[1,2]>=c and arr2[2,0]>=c and arr2[2,1]>=c and arr2[2,2]>=c) or (np.all(arr1<=c) and np.all(arr3<=c) and arr2[0,0]<=c and arr2[0,1]<=c and arr2[0,2]<=c and arr2[1,0]<=c and arr2[1,2]<=c and arr2[2,0]<=c and arr2[2,1]<=c and arr2[2,2]<=c):
        return True
    else:
        return False
    

def is_not_edgepoint(arr):
    trace=(arr[1,2]+arr[1,0]-2*arr[1,1])+(arr[0,1]+arr[2,1]-2*arr[1,1])
    det=(arr[1,2]+arr[1,0]-2*arr[1,1])*(arr[0,1]+arr[2,1]-2*arr[1,1])-(0.25*(arr[0,2]+arr[2,0]-arr[0,0]-arr[2,2]))**2
    r=10
    if((r*(trace**2))<(det*((r+1)**2))):
        return True
    else:
        return False
    

def find_octave_keypts(DoG_images, octave_idx, unique_kpts):

    num_DoG_images = len(DoG_images)
    h, w = DoG_images.shape[1:]
    octave_keypts={}

    for scale_idx in range(1,num_DoG_images-1):
        for j in range(1,h-1):
            for k in range(1,w-1):
                if(DoG_images[scale_idx,j,k]>0.0025):
                    if(is_not_edgepoint(DoG_images[scale_idx,j-1:j+2,k-1:k+2])):
                        if(is_extremum(DoG_images[scale_idx-1,j-1:j+2,k-1:k+2],DoG_images[scale_idx,j-1:j+2,k-1:k+2],DoG_images[scale_idx+1,j-1:j+2,k-1:k+2])):
                            # print(scale_idx)
                            keypt=(k,j)
                            # print(keypt)
                            original_keypt=(math.floor(k*(2**(octave_idx-1))), math.floor(j*(2**(octave_idx-1))))

                            if scale_idx not in octave_keypts:
                                octave_keypts[scale_idx]=[]

                            octave_keypts[scale_idx].append(keypt)
                            unique_kpts.add(original_keypt)
                            
    return octave_keypts


def get_principle_angles(orient_bins):

    original_orient_bins=orient_bins.copy()
    orient_bins /= np.max(orient_bins)
    max_locs=np.where(orient_bins>=0.8)[0]
    principle_angles=[]

    for i in max_locs:
        p1 = i-1
        p2 = i
        p3 = i+1
        t1=(10*p1)+5
        t2=(10*p2)+5
        t3=(10*p3)+5
        x=[t1,t2,t3]

        if(i>0 and i<35):
            y=[original_orient_bins[p1],original_orient_bins[p2],original_orient_bins[p3]]
            if(y[1]>y[0] and y[1]>y[2]):
                A = np.vstack([np.array(x)**2, x, np.ones(len(x))]).T
                coefficients = np.linalg.solve(A, y)
                final_t=-(coefficients[1])/(2*coefficients[0])
                principle_angles.append(final_t)

        elif(i==0):
            y=[original_orient_bins[35],original_orient_bins[0],original_orient_bins[1]]
            if(y[1]>y[0] and y[1]>y[2]):
                A = np.vstack([np.array(x)**2, x, np.ones(len(x))]).T
                coefficients = np.linalg.solve(A, y)
                final_t=-(coefficients[1])/(2*coefficients[0])
                if(final_t<0):
                    final_t = final_t+360.0
                principle_angles.append(final_t)

        elif(i==35):
            y=[original_orient_bins[34],original_orient_bins[35],original_orient_bins[0]]
            if(y[1]>y[0] and y[1]>y[2]):
                A = np.vstack([np.array(x)**2, x, np.ones(len(x))]).T
                coefficients = np.linalg.solve(A, y)
                final_t=-(coefficients[1])/(2*coefficients[0])
                if(final_t>360):
                    final_t = final_t-360.0
                principle_angles.append(final_t)
        # print(coefficients,final_t)
                
    return principle_angles


def gaussian_kernel2(g_size, g_sigma):
    kernel = np.zeros((g_size, g_size))
    g_center = 7
    for i in range(g_size):
        for j in range(g_size):
            kernel[i, j] = np.exp(-((i - g_center) ** 2 + (j - g_center) ** 2) / (2 * g_sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel


def find_gradients_magnitude_theta(gaussian_images, scale_idx, x, y):
    dx_kernel=np.array([[-1,0,1]])
    dy_kernel=np.array([[-1,0,1]]).T
    # print(gaussian_images[scale_idx,y-7:y+9,x-8:x+10])
    dx=convolution(gaussian_images[scale_idx,y-7:y+9,x-8:x+10],dx_kernel,False)
    # print(dx[:3,:3])
    dy=convolution(gaussian_images[scale_idx,y-8:y+10,x-7:x+9],dy_kernel,False)
    # print(dy[:3,:3])
    grad_magnitude=np.sqrt((dx**2)+(dy**2))
    # print(grad_magnitude[:3,:3])
    theta=np.degrees(np.arctan(np.divide(dy,dx)))
    theta[theta<0]+=360
    return grad_magnitude, theta


def find_keypt_orientation_descriptor(angle, desc_gauss_window, grad_magnitude, theta):
    descriptor=np.array([])
    for m in range(0,16,4):
        for n in range(0,16,4):
            sub_gauss_window=desc_gauss_window[m:m+4,n:n+4].copy()
            sub_grad_magnitude=grad_magnitude[m:m+4,n:n+4].copy()
            sgw_sub_grad_magnitude = sub_grad_magnitude*sub_gauss_window
            sub_angle=theta[m:m+4,n:n+4].copy()                    
            sub_angle-=angle
            sub_angle[sub_angle<0]+=360

            sub_bins=np.zeros(8)
            for p in range(4):
                for q in range(4):
                    sub_bin_idx = int(sub_angle[p,q]//45)
                    sub_bins[sub_bin_idx]+=sgw_sub_grad_magnitude[p,q]

            descriptor=np.concatenate((descriptor,sub_bins))
    return descriptor


def get_octave_keypts_desc(octave_keypts, gaussian_images, octave_idx, delta_sigmas):
    
    octave_keypts_desc_dict={}
    desc_gauss_window = gaussian_kernel2(16,8.0)
    # print(desc_gauss_window[:3,:3])

    for scale_idx, scale_keypts in octave_keypts.items():
        for keypt in scale_keypts:
            h,w=gaussian_images[0].shape
            x,y = keypt

            if (x>=8 and x<=w-10):
                if (y>=8 and y<=h-10):
                    original_keypt=(math.floor(x*(2**(octave_idx-1))), math.floor(y*(2**(octave_idx-1))))
                    
                    if original_keypt not in octave_keypts_desc_dict:
                        octave_keypts_desc_dict[original_keypt]=[]

                    sigma = np.sqrt(np.sum((delta_sigmas[0:scale_idx+1])**2))
                    # print(sigma*1.5)
                    orient_gauss_window=gaussian_kernel2(16,1.5*sigma)
                    # print(orient_gauss_window[:3,:3])

                    grad_magnitude, theta = find_gradients_magnitude_theta(gaussian_images, scale_idx, x, y)
                    ogw_grad_magnitude = grad_magnitude*orient_gauss_window

                    orient_bins=np.zeros(36)
                    for m in range(16):
                        for n in range(16):
                            orient_bin_idx = int(theta[m,n]//10)
                            orient_bins[orient_bin_idx]+=ogw_grad_magnitude[m,n]
                    principle_angles=get_principle_angles(orient_bins)
                    # print(principle_angles)

                    for angle in principle_angles:
                        descriptor=find_keypt_orientation_descriptor(angle, desc_gauss_window, grad_magnitude, theta)
                        # print(descriptor)
                        octave_keypts_desc_dict[original_keypt].append(descriptor)

    return octave_keypts_desc_dict


def merge_dicts(dict1,dict2):
    merged_dict = {}
    for key in set(dict1.keys()).union(dict2.keys()):
        if key in dict1 and key in dict2:
            merged_dict[key] = dict1[key]+dict2[key]
        elif key in dict1:
            merged_dict[key] = dict1[key]
        elif key in dict2:
            merged_dict[key] = dict2[key]
    return merged_dict


def find_keypts_and_desc(gray_image, num_octaves=5, num_scales=3, sigma=1.6):

    delta_sigmas = get_delta_sigmas(num_scales,sigma)
    # print(delta_sigmas)

    gray_image = gray_image/255.0
    upsampled_image = upsampling(gray_image,5)
    # print("upsampled_image")
    # print(upsampled_image[:3,:3])
    image_c = upsampled_image.copy()
    
    keypts_desc_dict = {}
    unique_kpts = set()
    g_size = 11

    for octave_idx in range(num_octaves):

        gaussian_images = [image_c]
        # print(image_c[:3, :3])
        for scale_idx in range(1,num_scales+3):
            # print(f"Creating Gaussian Pyramid, Scale = 2^({octave_no}+{scale}/{num_scales})*sigma")
            kernel = gaussian_kernel(g_size, delta_sigmas[scale_idx])
            image_c = convolution(image_c, kernel)
            gaussian_images.append(image_c)
#             gaussian_images.append(cv2.GaussianBlur(gaussian_images[-1], (0, 0), sigmaX=delta_sigmas[scale_idx], sigmaY=delta_sigmas[scale_idx]))
        gaussian_images = np.array(gaussian_images)
      
        DoG_images = []
        for scale_idx in range(num_scales+2):
            # print(f"Creating DoG Pyramid, Scale = 2^({octave_no}+{scale}/{num_scales})*sigma")
            DoG_image = gaussian_images[scale_idx+1]-gaussian_images[scale_idx]
            DoG_images.append(DoG_image)
        DoG_images = np.array(DoG_images)
 
        octave_keypts = find_octave_keypts(DoG_images, octave_idx, unique_kpts)
        
        octave_keypts_desc_dict = get_octave_keypts_desc(octave_keypts, gaussian_images, octave_idx, delta_sigmas)

        keypts_desc_dict = merge_dicts(keypts_desc_dict,octave_keypts_desc_dict)

        image_c = gaussian_images[-3][::2, ::2]

    return (unique_kpts, keypts_desc_dict)


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))


def match_keypoints(descriptors1, descriptors2, keypoints1, keypoints2, w1, w2, threshold=0.6):
    matches = []
    for i in range(len(descriptors1)):
        distances = []
        if keypoints1[i][0] > w1//2:            
            for j in range(len(descriptors2)):
                if keypoints2[j][0] < w2//2:
                    distances.append(euclidean_distance(descriptors1[i],descriptors2[j]))
                else:
                    distances.append(float('inf'))
        # print(distances)
        if len(distances) > 0:
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            second_best_distance = np.partition(distances, 1)[1]
            if best_distance <= threshold * second_best_distance:
                matches.append((i, best_match_index))
    return matches


def random_color():
    return (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))


def find_inliers_info(prediction_arr, destination_arr, confidence):
    
    threshold = 5
    num_keypts = prediction_arr.shape[0]
    difference = prediction_arr-destination_arr
    delta_X = difference[:,0]
    delta_Y = difference[:,1]
    distance = np.sqrt(delta_X**2 + delta_Y**2)
    # print(distance)
    num_inliers = np.count_nonzero(distance < threshold)
    fraction_of_inliers = num_inliers/num_keypts
    inlier_idx = np.where(distance < threshold)[0]
    
    return inlier_idx,fraction_of_inliers 


def find_H(source_list, destination_list):

    A=[]
    num_keypts = len(source_list)

    for j in range(num_keypts):
        a = source_list[j][0]
        b = source_list[j][1]
        c = destination_list[j][0]
        d = destination_list[j][1]
        row1 = np.array([a,b,1,0,0,0,-c*a,-c*b,-c])
        row2 = np.array([0,0,0,a,b,1,-d*a,-d*b,-d])
        A.append(row1)
        A.append(row2)

    A = np.array(A)

    A_trans_A = np.dot(A.T,A)
    eigen_val_arr, eigen_vec_arr = np.linalg.eig(A_trans_A)
    smallest_eigen_vec = eigen_vec_arr[:, np.argmin(eigen_val_arr)]
    homography = smallest_eigen_vec.reshape((3,3))
    
    return homography


def findHomography(keypoints1, keypoints2):
    
    num_keypts = len(keypoints1)
    num_iterations = 100000
    confidence = 0
    total_source_list=[]
    total_destination_list=[]

    for i in range(num_iterations):

        random_keypts1 = []
        random_keypts2 = []        
        other_than_random_keypts_idx = []
        other_than_random_keypts1=[]
        other_than_random_keypts2=[]

        random_keypts_idx = random.sample(range(num_keypts), 4)

        for keypt_idx in random_keypts_idx:
            random_keypts1.append(keypoints1[keypt_idx])
            random_keypts2.append(keypoints2[keypt_idx])
                           
        s = set()        
        for j in range(4):
            s.add((tuple(random_keypts1[j]),tuple(random_keypts2[j])))
        if len(s) < 4:
            continue

        homography = find_H(random_keypts1,random_keypts2)
        # print(homography)

        for keypt_idx in range(num_keypts):
            if keypt_idx not in random_keypts_idx:
                other_than_random_keypts_idx.append(keypt_idx)
                other_than_random_keypts1.append(keypoints1[keypt_idx])
                other_than_random_keypts2.append(keypoints2[keypt_idx])

        array_other_than_random_keypts1 = np.array(other_than_random_keypts1)
        array_other_than_random_keypts2 = np.array(other_than_random_keypts2)
        source_array = np.hstack((array_other_than_random_keypts1, np.ones((num_keypts-4, 1))))
        # print(source_matrix)      

        prediction_array=np.dot(source_array,homography.T)
        # print(prediction_matrix)

        prediction_array[:,0]/=prediction_array[:,2]
        prediction_array[:,1]/=prediction_array[:,2]

        inliers_idx, fraction_of_inliers = find_inliers_info(prediction_array[:,:2], array_other_than_random_keypts2, confidence)
        
        # print(fraction_of_inliers)
        
        if fraction_of_inliers > confidence:

            confidence = fraction_of_inliers

            total_source_list=[]
            total_destination_list=[]

            for keypt_idx in random_keypts_idx:
                total_source_list.append(keypoints1[keypt_idx])
                total_destination_list.append(keypoints2[keypt_idx])

            mask=np.zeros(num_keypts)

            for idx in inliers_idx:
                total_source_list.append(other_than_random_keypts1[idx])
                total_destination_list.append(other_than_random_keypts2[idx])
                mask[other_than_random_keypts_idx[idx]] = 1

    print("confidence = ", confidence)           

    homography = find_H(total_source_list, total_destination_list)
    return homography, mask


def warpPerspective(original_image, adjusted_homography_matrix, calc_warp_w1, calc_warp_h1):

    H_inv = np.linalg.inv(adjusted_homography_matrix)
    original_h, original_w = original_image.shape[:2]
    warp_w = calc_warp_w1
    warp_h = calc_warp_h1
    warped_image = np.ones((warp_h, warp_w, 3), dtype=np.uint8) * 0
    # point = 0

    for y in range(warp_h):
        for x in range(warp_w):
            # print(point)
            # point = point+1
            warped_point = np.array([x, y, 1])
            original_point = np.dot(H_inv, warped_point.T)
            original_point /= original_point[2]
            if original_point[0]>=0 and original_point[0]<=original_w-1 and original_point[1]>=0 and original_point[1]<=original_h-1:
                original_point = (np.round(original_point)).astype(int)
                # print(original_point)
                warped_image[y, x] = original_image[original_point[1], original_point[0]]
            
    return warped_image


# def warpPerspective(original_image, adjusted_homography_matrix, calc_warp_w1, calc_warp_h1):

#     H_inv = np.linalg.inv(adjusted_homography_matrix)
#     original_h, original_w = original_image.shape[:2]
#     warp_w = calc_warp_w1
#     warp_h = calc_warp_h1
#     warped_image = np.ones((warp_h, warp_w, 3), dtype=np.uint8) * 0
#     # point = 0

#     warped_points = []

#     for y in range(warp_h):
#         for x in range(warp_w):
#             # print(point)
#             # point = point+1
#             warped_points.append([x,y,1])

#     warped_points = np.array(warped_points)
#     # print(warped_points.shape)
#     # print(warped_points[:5])

#     original_points = np.dot(warped_points, H_inv.T)
#     # print(original_points.shape)
#     # print(original_points[:5])

#     original_points[:,0] /= original_points[:,2]
#     original_points[:,1] /= original_points[:,2]
#     # print(original_points[:5])

#     original_points = (np.round(original_points)).astype(int)
#     # print(original_points.shape)

#     warped_points = warped_points[(original_points[:,0] >= 0) & (original_points[:,0] <= original_w-1) & (original_points[:,1] >= 0) & (original_points[:,1] <= original_h-1)]
#     # print(warped_points.shape)

#     original_points = original_points[(original_points[:,0] >= 0) & (original_points[:,0] <= original_w-1) & (original_points[:,1] >= 0) & (original_points[:,1] <= original_h-1)]
#     # print(original_points.shape)

#     for i in range (len(original_points)):
#         warped_image[warped_points[i][1], warped_points[i][0]] = original_image[original_points[i][1], original_points[i][0]]

#     del warped_points
#     del original_points

#     return warped_image


def cross_product(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])


def is_point_inside_rectangle(rectangle_x, rectangle_y, point):
    products = [cross_product((rectangle_x[i], rectangle_y[i]), (rectangle_x[i + 1], rectangle_y[i + 1]), point) for i in range(3)]
    products.append(cross_product((rectangle_x[3], rectangle_y[3]), (rectangle_x[0], rectangle_y[0]), point))
    return all(prod >= 0 for prod in products) or all(prod <= 0 for prod in products)


def find_weights(img,ip):
        x=ip[:,0]
        y=ip[:,1]
        x = list(x)
        y = list(y)
        x.append(x[0])
        y.append(y[0])
        cons=[]
        for i in range(4):
            temp=[]
            temp.append((y[i+1]-y[i]))
            temp.append(-1*(x[i+1]-x[i]))
            temp.append(((y[i]*(x[i+1]-x[i]))-((y[i+1]-y[i])*x[i])))
            cons.append(temp)
        h,w=img.shape[:2]
        weights=np.zeros((h,w))
        for i in range(h):
              for j in range(w):
                    if (is_point_inside_rectangle(x,y,(j,i))):
                        m=float('inf')
                        for l in cons:
                            m=min(m,(abs(j*l[0]+i*l[1]+l[2]))/(math.sqrt(l[0]**2+l[1]**2)))
                        weights[i,j]=m
                    elif not np.array_equal(img[i, j, :], [0, 0, 0]):
                        weights[i,j]=100
        return weights


def multiply_image_channels(image, matrix):

    channels = cv2.split(image)
    multiplied_channels = [channel * matrix for channel in channels]
    result_image = cv2.merge(multiplied_channels)
    return result_image


def divide_image_channels(image, matrix):

    channels = cv2.split(image)
    divided_channels = [np.divide(channel, matrix, out=np.zeros_like(channel), where=(matrix != 0)) for channel in channels]
    result_image = cv2.merge(divided_channels)
    return result_image


def stitch_left(image1, image2, keypoints1, keypoints2, corners_new, output_dir, tracker):
    
    # retval, mask = cv2.findHomography(srcPoints, dstPoints, method=None, ransacReprojThreshold=None, maxIters=None, confidence=None, refineIters=None)
#     homography, mask = cv2.findHomography(keypoints1, keypoints2, cv2.RANSAC)
    homography, mask = findHomography(keypoints1, keypoints2)
    # print("homography")
    # print(homography)
    # print(mask)

    keypoints1 = keypoints1[mask.ravel() == 1]
    keypoints2 = keypoints2[mask.ravel() == 1]

    h1, w1 = image1.shape[:2]
    # print("original image1 size")
    # print(image1.shape[:2])
    h2, w2 = image2.shape[:2]
    # print("original image2 size")
    # print(image2.shape[:2])

    corners1 = np.array([[0, 0, 1], [0, h1 - 1, 1], [w1 - 1, h1 - 1, 1], [w1 - 1, 0, 1]])
    # print("original image1 corners")
    # print(corners1)
    
    transformed_corners1 = np.dot(homography, corners1.T).T
    # print(transformed_corners1)
    normalized_corners1 = transformed_corners1[:, :2] / transformed_corners1[:, 2][:, np.newaxis]
    # print("original transformed image1 corners")
    # print(normalized_corners1)

    min_x = np.min(normalized_corners1[:, 0])
    min_y = np.min(normalized_corners1[:, 1])
    max_x = np.max(normalized_corners1[:, 0])
    max_y = np.max(normalized_corners1[:, 1])
    # print(min_x, min_y, max_x, max_y)
    # print(math.floor(min_x), math.floor(min_y), math.ceil(max_x), math.ceil(max_y))

    calc_warp_w1 = math.ceil(max_x) - math.floor(min_x)
    calc_warp_h1 = math.ceil(max_y) - math.floor(min_y)
    # print("calculated warped image1 size")
    # print(calc_warp_h1, calc_warp_w1)

    trans_x = -math.floor(min_x)
    trans_y = -math.floor(min_y)

    translation_matrix = np.array([[1, 0, trans_x], [0, 1, trans_y], [0, 0, 1]])
    # print("translation_matrix")
    # print(translation_matrix)

    adjusted_homography_matrix = np.dot(translation_matrix, homography)
    # print("adjusted_homography_matrix")
    # print(adjusted_homography_matrix)

    transformed_corners3 = np.dot(adjusted_homography_matrix, corners1.T).T
    normalized_corners3 = transformed_corners3[:, :2] / transformed_corners3[:, 2][:, np.newaxis]
    # print("new transformed image1 corners")
    # print(normalized_corners3)

    warp_image1 = warpPerspective(image1, adjusted_homography_matrix, calc_warp_w1, calc_warp_h1)
    # warp_image1 = cv2.warpPerspective(image1, adjusted_homography_matrix, (calc_warp_w1, calc_warp_h1), borderValue=(0, 0, 0))
    # print("actual warped image1 size")
    # print(warp_image1.shape[:2])
    warp_h1, warp_w1, _ = warp_image1.shape

    # plt.subplot(1,2,1)
    # plt.imshow(warp_image1)
    # plt.axis('off')
    # plt.imsave('warp_image1.jpg', warp_image1, format='jpg')

    # plt.subplot(1,2,2)
    # plt.imshow(image2)
    # plt.axis('off')
    # plt.imsave('image2.jpg', image2, format='jpg')

    # plt.show()

    keypoints1 = np.hstack((keypoints1, np.ones((keypoints1.shape[0], 1))))
    # print(keypoints1)
    warp_keypoints1 = np.dot(adjusted_homography_matrix, keypoints1.T).T
    # print(transformed_keypoints1)
    warp_keypoints1 = warp_keypoints1[:, :2] / warp_keypoints1[:, 2][:, np.newaxis]
    # print(warp_keypoints1)

    warp_image1_c = warp_image1.copy()
    image2_c = image2.copy()      

    for i in range (len(keypoints1)):
        x1 = warp_keypoints1[i][0]
        y1 = warp_keypoints1[i][1]
        x2 = keypoints2[i][0]
        y2 = keypoints2[i][1]
        color = random_color()
        cv2.circle(warp_image1_c, (int(x1), int(y1)), 5, color, -1)
        cv2.circle(image2_c, (int(x2), int(y2)), 5, color, -1)

    # plt.subplot(1, 2, 1)
    # plt.imshow(warp_image1_c)
    # # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(image2_c)
    # # plt.axis('off')
    
    # plt.savefig(os.path.join(output_dir, f"warped_matching_sift_{tracker+1}_{tracker+2}.jpg"), format='jpg')
        
    # plt.show()

    transformed_corners_new = np.dot(adjusted_homography_matrix, corners_new.T).T
    # print(transformed_corners1)
    normalized_corners_new = transformed_corners_new[:, :2] / transformed_corners_new[:, 2][:, np.newaxis]

    warp_image1_weights = find_weights(warp_image1,normalized_corners_new)
    # plt.subplot(1,2,1)
    # plt.imshow(warp_image1_weights,cmap='gray')
    # plt.axis('off')
    # plt.show()

    warp_image1 = multiply_image_channels(warp_image1, warp_image1_weights)

    # plt.imshow(warp_image1)
    # plt.show()

    corners2 = np.array([[0, 0], [0, h2 - 1], [w2 - 1, h2 - 1], [w2 - 1, 0]])
    # print("original image2 corners")
    # print(corners2)

    image2_weights = find_weights(image2,corners2)
    # plt.subplot(1,2,2)
    # plt.imshow(image2_weights,cmap='gray')
    # plt.savefig(os.path.join(output_dir, f"weights_{tracker+1}_{tracker+2}.jpg"), format='jpg')
    # plt.axis('off')
    # plt.show()

    image2 = multiply_image_channels(image2, image2_weights)

    new_w = w2 + trans_x
    # print(w2,trans_x)
    # print(new_w)
    new_h = max(math.ceil(max_y),h2)-min(math.floor(min_y),0)
    # print(math.ceil(max_y),h2,math.floor(min_y),0)
    # print(new_h)
    extended_image = np.ones((new_h, new_w, 3), dtype=np.uint8) * 0.0
    extended_weight = np.ones((new_h, new_w), dtype=np.uint8) * 0.0

    if min_y<=0:
    
        extended_image[:warp_h1, :warp_w1] += warp_image1
        extended_image[trans_y:trans_y+h2, trans_x:trans_x+w2] += image2

        extended_weight[:warp_h1, :warp_w1] += warp_image1_weights
        extended_weight[trans_y:trans_y+h2, trans_x:trans_x+w2] += image2_weights

        # print(extended_image[:3,:3,0])

    else:
    
        extended_image[-trans_y:-trans_y+warp_h1, :warp_w1] += warp_image1
        extended_image[:h2, trans_x:trans_x+w2] += image2

        extended_weight[-trans_y:-trans_y+warp_h1, :warp_w1] += warp_image1_weights
        extended_weight[:h2, trans_x:trans_x+w2] += image2_weights

        trans_y = 0

    extended_image = divide_image_channels(extended_image, extended_weight).astype(np.uint8)

    return (extended_image, trans_x, trans_y)


def stitch_right(image1, image2, keypoints1, keypoints2, corners1_new, corners2_new, output_dir, tracker):
    
    # retval, mask = cv2.findHomography(srcPoints, dstPoints, method=None, ransacReprojThreshold=None, maxIters=None, confidence=None, refineIters=None)
#     homography, mask = cv2.findHomography(keypoints2, keypoints1, cv2.RANSAC)
    homography, mask = findHomography(keypoints2, keypoints1)
    # print("homography")
    # print(homography)
    # print(mask)

    keypoints1 = keypoints1[mask.ravel() == 1]
    keypoints2 = keypoints2[mask.ravel() == 1]

    h1, w1 = image1.shape[:2]
    # print("original image1 size")
    # print(image1.shape[:2])
    h2, w2 = image2.shape[:2]
    # print("original image2 size")
    # print(image2.shape[:2])

    corners2 = np.array([[0, 0, 1], [0, h2 - 1, 1], [w2 - 1, h2 - 1, 1], [w2 - 1, 0, 1]])
    # print("original image2 corners")
    # print(corners2)

    transformed_corners2 = np.dot(homography, corners2.T).T
    # print(transformed_corners2)
    normalized_corners2 = transformed_corners2[:, :2] / transformed_corners2[:, 2][:, np.newaxis]
    # print("original transformed image2 corners")
    # print(normalized_corners2)

    min_x = np.min(normalized_corners2[:, 0])
    min_y = np.min(normalized_corners2[:, 1])
    max_x = np.max(normalized_corners2[:, 0])
    max_y = np.max(normalized_corners2[:, 1])
    # print(min_x, min_y, max_x, max_y)
    # print(math.floor(min_x), math.floor(min_y), math.ceil(max_x), math.ceil(max_y))

    calc_warp_w2 = math.ceil(max_x) - math.floor(min_x)
    calc_warp_h2 = math.ceil(max_y) - math.floor(min_y)
    # print("calculated warped image2 size")
    # print(calc_warp_h2, calc_warp_w2)

    trans_x = -math.floor(min_x)
    trans_y = -math.floor(min_y)

    translation_matrix = np.array([[1, 0, trans_x], [0, 1, trans_y], [0, 0, 1]])
    # print("translation_matrix")
    # print(translation_matrix)

    adjusted_homography_matrix = np.dot(translation_matrix, homography)
    # print("adjusted_homography_matrix")
    # print(adjusted_homography_matrix)

    transformed_corners3 = np.dot(adjusted_homography_matrix, corners2.T).T
    normalized_corners3 = transformed_corners3[:, :2] / transformed_corners3[:, 2][:, np.newaxis]
    # print("new transformed image2 corners")
    # print(normalized_corners3)

    warp_image2 = warpPerspective(image2, adjusted_homography_matrix, calc_warp_w2, calc_warp_h2)
    # warp_image2 = cv2.warpPerspective(image2, adjusted_homography_matrix, (calc_warp_w2, calc_warp_h2), borderValue=(0, 0, 0))
    # print("actual warped image2 size")
    # print(warp_image2.shape[:2])
    warp_h2, warp_w2, _ = warp_image2.shape

    # plt.subplot(1,2,1)
    # plt.imshow(image1)
    # plt.axis('off')
    # plt.imsave('image1.jpg', image1, format='jpg')

    # plt.subplot(1,2,2)
    # plt.imshow(warp_image2)
    # plt.axis('off')
    # plt.imsave('warp_image2.jpg', warp_image2, format='jpg')

    # plt.show()

    keypoints2 = np.hstack((keypoints2, np.ones((keypoints2.shape[0], 1))))
    # print(keypoints2)
    warp_keypoints2 = np.dot(adjusted_homography_matrix, keypoints2.T).T
    # print(transformed_keypoints2)
    warp_keypoints2 = warp_keypoints2[:, :2] / warp_keypoints2[:, 2][:, np.newaxis]
    # print(warp_keypoints2)

    image1_c = image1.copy()   
    warp_image2_c = warp_image2.copy()

    for i in range (len(keypoints1)):
        x1 = keypoints1[i][0]
        y1 = keypoints1[i][1]
        x2 = warp_keypoints2[i][0]
        y2 = warp_keypoints2[i][1]
        color = random_color()
        cv2.circle(image1_c, (int(x1), int(y1)), 10, color, -1)
        cv2.circle(warp_image2_c, (int(x2), int(y2)), 10, color, -1)

    # plt.subplot(1, 2, 1)
    # plt.imshow(image1_c)
    # # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(warp_image2_c)
    # # plt.axis('off')
    
    # plt.savefig(os.path.join(output_dir, f"warped_matching_sift_{tracker+1}_{tracker+2}.jpg"), format='jpg')
        
    # plt.show()

    # print("original image2 corners")
    # print(corners2)

    image1_weights = find_weights(image1,corners1_new)
    # plt.subplot(1,2,1)
    # plt.imshow(image1_weights,cmap='gray')
    # plt.axis('off')
    # plt.show()

    image1 = multiply_image_channels(image1, image1_weights)
    
    transformed_corners2_new = np.dot(adjusted_homography_matrix, corners2_new.T).T
    normalized_corners2_new = transformed_corners2_new[:, :2] / transformed_corners2_new[:, 2][:, np.newaxis]

    warp_image2_weights = find_weights(warp_image2,normalized_corners2_new)
    # plt.subplot(1,2,2)
    # plt.imshow(warp_image2_weights,cmap='gray')
    # # plt.axis('off')
    # plt.savefig(os.path.join(output_dir, f"weights_{tracker+1}_{tracker+2}.jpg"), format='jpg')
        
    # plt.show()

    warp_image2 = multiply_image_channels(warp_image2, warp_image2_weights)

    # plt.imshow(warp_image1)
    # plt.show()  

    new_w = warp_w2 - trans_x
    # print(warp_w2,-trans_x)
    # print(new_w)
    new_h = max(math.ceil(max_y),h1)-min(math.floor(min_y),0)
    # print(math.ceil(max_y),h1,math.floor(min_y),0)
    # print(new_h)
    extended_image = np.ones((new_h, new_w, 3), dtype=np.uint8) * 0.0
    extended_weight = np.ones((new_h, new_w), dtype=np.uint8) * 0.0

    if min_y<=0:
    
        extended_image[:warp_h2, -trans_x:-trans_x+warp_w2] += warp_image2
        extended_image[trans_y:trans_y+h1, :w1] += image1
        extended_weight[:warp_h2, -trans_x:-trans_x+warp_w2] += warp_image2_weights
        extended_weight[trans_y:trans_y+h1, :w1] += image1_weights

    else:

        extended_image[-trans_y:-trans_y+warp_h2, -trans_x:-trans_x+warp_w2] += warp_image2
        extended_image[:h1, :w1] += image1
        extended_weight[-trans_y:-trans_y+warp_h2, -trans_x:-trans_x+warp_w2] += warp_image2_weights
        extended_weight[:h1, :w1] += image1_weights

        trans_y = 0

    extended_image = divide_image_channels(extended_image, extended_weight).astype(np.uint8)

    return (extended_image,trans_y)


def capture_screenshots(video_path, output_folder, num_screenshots):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("total frames = ", total_frames)

    frames_per_interval = math.floor((total_frames-1) / (num_screenshots-1))
    
    print("frames per interval = ", frames_per_interval)

    frame_idx = 0
    screenshot_no = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frames_per_interval == 0:
            screenshot_path = os.path.join(output_folder, f'{screenshot_no}.jpg')
            cv2.imwrite(screenshot_path, frame)
            screenshot_no += 1

        frame_idx += 1

    cap.release()


if part_id == 1:
    folder_path = input_dir

elif part_id == 2:
    video_path = input_dir    
    folder_path =  os.path.join(output_dir, "Video_Screenshots")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)       
    num_screenshots = 5
    capture_screenshots(video_path, folder_path, num_screenshots)
    
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


images = []
all_images_keypts_desc_dict_list=[]
keypoints_list = []
descriptors_list = []

file_list = os.listdir(folder_path)
file_list = np.sort(file_list)

for file_name in file_list:
    
    print(file_name)
    image_path = os.path.join(folder_path, file_name)
    base_filename, file_extension = os.path.splitext(os.path.basename(image_path))
    bgr_image = cv2.imread(image_path)
    height, width = bgr_image.shape[:2]
    bgr_image = cv2.resize(bgr_image, (width, height))
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    images.append(rgb_image)    
    
    # plt.subplot(1, 2, 1)
    # plt.imshow(rgb_image)
    # plt.axis('off')

    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    unique_kpts, keypts_desc_dict = find_keypts_and_desc(gray_image)
    all_images_keypts_desc_dict_list.append(keypts_desc_dict)

    sift_rgb_image = rgb_image.copy()

    for kp in unique_kpts:
        x, y = kp
        color = random_color()
        cv2.circle(sift_rgb_image, (int(x),int(y)), 10, color, -1)

    # plt.subplot(1, 2, 2)
    # plt.imshow(sift_rgb_image)
    # # plt.axis('off')
    # plt.imsave(os.path.join(output_dir, f"sift_{base_filename}.jpg"), sift_rgb_image, format='jpg')
    # plt.show()   

for image_keypts_desc_dict in all_images_keypts_desc_dict_list:
    image_keypoints=[]
    image_descriptors=[]

    for keypt, desc_list in image_keypts_desc_dict.items():
        for desc in desc_list:
            image_keypoints.append(keypt)
            image_descriptors.append(desc)
            
    keypoints_list.append(image_keypoints)
    descriptors_list.append(image_descriptors)

    print(len(image_keypoints),len(image_descriptors))


num_images = len(images)

all_matches = []

for i in range(num_images-1):
    h1, w1 = images[i].shape[:2]
    h2, w2 = images[i+1].shape[:2]
    matches = match_keypoints(descriptors_list[i], descriptors_list[i+1], keypoints_list[i], keypoints_list[i+1], w1, w2)
    all_matches.append(matches)

for i in range(num_images-1):
    image1 = images[i].copy()
    image2 = images[i+1].copy()

    print(i, i+1, len(all_matches[i]))      

    for match in all_matches[i]:
        kp1_idx, kp2_idx = match
        kp1 = keypoints_list[i][kp1_idx]
        kp2 = keypoints_list[i+1][kp2_idx]
        x1, y1 = kp1
        x2, y2 = kp2
        color = random_color()
        cv2.circle(image1, (int(x1), int(y1)), 10, color, -1)
        cv2.circle(image2, (int(x2), int(y2)), 10, color, -1)

    # plt.subplot(1, 2, 1)
    # plt.imshow(image1)
    # # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(image2)
    # # plt.axis('off')
        
    # plt.savefig(os.path.join(output_dir, f"matching_sift_{i+1}_{i+2}.jpg"), format='jpg')
    # plt.show()


tracker = max(0,num_images//2-2)
img1_idx = tracker
image1_left = images[img1_idx].copy()
trans_x_left = 0
trans_y_left = 0

while tracker <= num_images//2-1:

    keypoints1 = []
    keypoints2 = []

    img2_idx = tracker+1
    image2_left = images[img2_idx].copy()
    
    for match in all_matches[tracker]:
        kp1_idx, kp2_idx = match
        kp1_pt = keypoints_list[img1_idx][kp1_idx]
        kp2_pt = keypoints_list[img2_idx][kp2_idx]
        keypoints1.append(kp1_pt)
        keypoints2.append(kp2_pt)

    keypoints1 = np.array(keypoints1)
    keypoints2 = np.array(keypoints2)

    keypoints1[:, 0] += trans_x_left
    keypoints1[:, 1] += trans_y_left


    h2, w2 = image2.shape[:2]
    corners_new = np.array([[trans_x_left, trans_y_left, 1], [trans_x_left, h2-1+trans_y_left, 1], [w2-1+trans_x_left, h2-1+trans_y_left, 1], [w2-1+trans_x_left, trans_y_left, 1]])


    # image1_test = image1_left.copy()

    # for i in range (len(keypoints1)):
    #     x1 = keypoints1[i][0]
    #     y1 = keypoints1[i][1]
    #     color = random_color()
    #     cv2.circle(image1_test, (int(x1), int(y1)), 5, color, -1)

    # plt.imshow(image1_test)
    # # plt.axis('off')
    # plt.show()
    

    image1_left, trans_x_left, trans_y_left = stitch_left(image1_left, image2_left, keypoints1, keypoints2, corners_new, output_dir, tracker)
    # plt.imshow(image1_left)
    # # plt.axis('off')
    # plt.imsave(os.path.join(output_dir, f"stitched_{tracker+1}_{tracker+2}.jpg"), image1_left, format='jpg')
    # plt.show()

    tracker += 1
    img1_idx = tracker

# plt.imsave('Panaroma_left.jpg', image1_left, format='jpg')

# plt.imsave(os.path.join(output_dir, 'Panaroma_left.jpg'), image1_left, format='jpg')


tracker = min(num_images//2+1,num_images-2)
img2_idx = tracker+1
image2_right = images[img2_idx].copy()
trans_y_right = 0

while tracker >= num_images//2+1:

    keypoints1 = []
    keypoints2 = []

    img1_idx = tracker
    image1_right = images[img1_idx].copy()
    
    for match in all_matches[tracker]:
        kp1_idx, kp2_idx = match
        kp1_pt = keypoints_list[img1_idx][kp1_idx]
        kp2_pt = keypoints_list[img2_idx][kp2_idx]
        keypoints1.append(kp1_pt)
        keypoints2.append(kp2_pt)

    keypoints1 = np.array(keypoints1)
    keypoints2 = np.array(keypoints2)

    keypoints2[:, 1] += trans_y_right

    h1, w1 = image1.shape[:2]
    corners1_new = np.array([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]])
    corners2_new = np.array([[0, trans_y_right, 1], [0, h1-1+trans_y_right, 1], [w1-1, h1-1+trans_y_right, 1], [w1-1, trans_y_right, 1]])

    image2_right, trans_y_right = stitch_right(image1_right, image2_right, keypoints1, keypoints2, corners1_new, corners2_new, output_dir, tracker)
    # plt.imshow(image2_right)
    # # plt.axis('off')
    # plt.imsave(os.path.join(output_dir, f"stitched_{tracker+1}_{tracker+2}.jpg"), image2_right, format='jpg')
    # plt.show()

    tracker -= 1
    img2_idx = tracker+1

# plt.imsave('Panaroma_right.jpg', image2_right, format='jpg')

# plt.imsave(os.path.join(output_dir, 'Panaroma_right.jpg'), image2_right, format='jpg')


if num_images==2:
    Panaroma = image1_left

else:
    tracker = num_images//2
    img1_idx = tracker
    img2_idx = tracker+1

    keypoints1 = []
    keypoints2 = []

    image1_right = image1_left
        
    for match in all_matches[tracker]:
        kp1_idx, kp2_idx = match
        kp1_pt = keypoints_list[img1_idx][kp1_idx]
        kp2_pt = keypoints_list[img2_idx][kp2_idx]
        keypoints1.append(kp1_pt)
        keypoints2.append(kp2_pt)

    keypoints1 = np.array(keypoints1)
    keypoints2 = np.array(keypoints2)

    keypoints1[:, 0] += trans_x_left
    keypoints1[:, 1] += trans_y_left
    keypoints2[:, 1] += trans_y_right

    h1, w1 = images[tracker].shape[:2]
    corners1_new = np.array([[trans_x_left, trans_y_left], [trans_x_left, h1-1+trans_y_left], [w1-1+trans_x_left, h1-1+trans_y_left], [w1-1+trans_x_left, trans_y_left]])

    h2, w2 = images[tracker+1].shape[:2]
    corners2_new = np.array([[0, trans_y_right, 1], [0, h2-1+trans_y_right, 1], [w2-1, h2-1+trans_y_right, 1], [w2-1, trans_y_right, 1]])

    Panaroma, trans_y_right = stitch_right(image1_right, image2_right, keypoints1, keypoints2, corners1_new, corners2_new, output_dir, tracker)
    
# plt.imshow(Panaroma)
# plt.axis('off')
# plt.imsave(os.path.join(output_dir, 'Panaroma.jpg'), Panaroma, format='jpg')
# plt.show()

cv2.imwrite(os.path.join(output_dir, 'Panaroma.jpg'), cv2.cvtColor(Panaroma, cv2.COLOR_RGB2BGR))