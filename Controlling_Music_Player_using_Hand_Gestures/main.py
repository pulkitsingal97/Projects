import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import mediapipe as mp
from sklearn import svm, metrics
import ast
import sys
from joblib import dump, load
import pygame

# python3 main.py 1 _ "[['Final/closed1/train', 'Final/closed2/train', 'Final/closed3/train'], ['Final/open1/train', 'Final/open2/train']]" train_op
# python3 main.py 2 train_op/position_svm.joblib "[['Final/closed1/valid', 'Final/closed2/valid', 'Final/closed3/valid'], ['Final/open1/valid', 'Final/open2/valid']]" valid_op
# python3 main.py 3 train_op/position_svm.joblib "['Final/closed1/valid', 'Final/closed2/valid', 'Final/closed3/valid', 'Final/open1/valid', 'Final/open2/valid']" test_op
# python3 main.py 4 Welcome.mp3 "[]" app_op

if len(sys.argv) != 5:
    print("Usage: python3 main.py <part_id> <model/mp3file_dir> <data_dir_list> <output_dir>")
    sys.exit(1)

part_id = int(sys.argv[1])
model_dir = sys.argv[2]
data_dir = ast.literal_eval(sys.argv[3])
output_dir = sys.argv[4]

def increase_bbox(bbox, scale_factor, height_image, width_image):
    x, y, w, h = bbox
    delta_w = int((scale_factor - 1) * w / 2)
    delta_h = int((scale_factor - 1) * h / 2)
    x1_new = max(0, x - delta_w)
    y1_new = max(0, y - delta_h)
    x2_new = min(width_image, x + w + delta_w)
    y2_new = min(height_image, y + h + delta_h)
    w_new = x2_new - x1_new
    h_new = y2_new - y1_new
    return x1_new, y1_new, w_new, h_new


def add_text_in_bbox(image, bbox, text, color, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
    
    x1, y1, x2, y2 = bbox
    
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, text_height = text_size

    text_x = center_x - text_width // 2
    text_y = center_y + text_height // 2

    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)


def convolution(image, kernel, padding=True):
    if padding:
        pad_height = kernel.shape[0] // 2
        pad_width = kernel.shape[1] // 2
        image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')
    result_height = image.shape[0]-kernel.shape[0]+1
    result_width = image.shape[1]-kernel.shape[1]+1
    all_windows = np.lib.stride_tricks.sliding_window_view(image, kernel.shape)
    num_windows = result_height*result_width
    windows_matrix = all_windows.reshape(num_windows, -1)
    flattened_kernel = kernel.flatten()
    result = np.dot(windows_matrix, flattened_kernel)
    result = result.reshape(result_height, result_width)
    return result
    

def find_gradients_magnitude_theta(gray_image):

    dx_kernel=np.array([[-1,0,1]])
    dy_kernel=np.array([[-1,0,1]]).T

    dx=convolution(gray_image,dx_kernel,True)
    dy=convolution(gray_image,dy_kernel,True)

    grad_magnitude=np.sqrt((dx**2)+(dy**2))

    theta=np.degrees(np.arctan2(dy,dx))
    theta[theta < 0] += 180
    theta[theta == 180] = 0

    return grad_magnitude, theta
    

def get_hog(gray_cropped_image, orientations, pixels_per_cell, cells_per_block):

    grad_magnitude, grad_theta = find_gradients_magnitude_theta(gray_cropped_image)
    patch_height, patch_width = grad_magnitude.shape
    bin_theta = 180.0 / orientations

    cy, cx = pixels_per_cell
    num_cells_y = patch_height // cy
    num_cells_x = patch_width // cx

    histogram = np.zeros((num_cells_y, num_cells_x, orientations))

    for i in range(0, patch_height-cy+1, cy):
        for j in range(0, patch_width-cx+1, cx):
            cell_grad_magnitude = grad_magnitude[i:i+cy, j:j+cx].copy()
            cell_grad_theta = grad_theta[i:i+cy, j:j+cx].copy()

            cell_descriptor=np.zeros(orientations)
            for p in range(cy):
                for q in range(cx):
                    # print('p', p, 'q', q)

                    theta = cell_grad_theta[p,q]

                    idx = int((theta-bin_theta/2)//bin_theta)
                    start = bin_theta*(idx+0.5)
                    end = bin_theta*(idx+1.5)
                    diff_start = theta-start
                    diff_end = end-theta

                    if idx == orientations-1:
                        idx = -1

                    grad = cell_grad_magnitude[p,q]
                    cell_descriptor[idx] += grad*diff_end/bin_theta
                    cell_descriptor[idx+1] += grad*diff_start/bin_theta

            histogram[i//cy, j//cx] = cell_descriptor

    hog_image = np.zeros_like(gray_cropped_image)
    for i in range(histogram.shape[0]):
        for j in range(histogram.shape[1]):
            cell_histogram = histogram[i, j].copy()
            max_val = np.max(cell_histogram)
            if max_val != 0 :
                factor = min(0.01, (1/max_val)*(cx/2))
            else:
                factor = 1
            cell_histogram *= factor
            angle = 90 + bin_theta/2  # edges are perpendicular to gradient directions
            for hist_value in cell_histogram:
                angle_radian = np.deg2rad(angle)
                y = i * cy + cy // 2 + int(hist_value * np.sin(angle_radian))
                x = j * cx + cx // 2 + int(hist_value * np.cos(angle_radian))
                if not ( (y == i * cy + cy // 2) and (x == j * cx + cx // 2) ):
                    cv2.line(hog_image, (j * cx + cx // 2, i * cy + cy // 2), (x, y), 255, 1)
                angle += bin_theta

    descriptor = np.array([])

    for i in range(0, num_cells_y-cells_per_block[0]+1, cells_per_block[0]//2):
        for j in range(0, num_cells_x-cells_per_block[1]+1, cells_per_block[1]//2):

            block_descriptor=np.array([])
            for p in range(i, i+cells_per_block[0]):
                for q in range(j, j+cells_per_block[1]):
                    block_descriptor = np.concatenate((block_descriptor, histogram[p,q]))
            block_descriptor /= np.linalg.norm(block_descriptor)

            descriptor=np.concatenate((descriptor, block_descriptor))

    return descriptor, hog_image


def evaluate (labels, predictions, scores, output_dir):

    print(metrics.classification_report(labels, predictions, target_names=["Closed", "Open"], digits=3))

    cm = metrics.confusion_matrix(labels, predictions, labels=["Closed", "Open"])
    print("Confusion Matrix =")
    print(cm)
    cm_disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Closed", "Open"])
    cm_disp.plot()
    # plt.show()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.jpg'))

    print()
    cc, co, oc, oo = cm.ravel()
    tpr_closed = cc / (cc + co)
    tpr_open = oo / (oo + oc)
    fpr_closed = oc / (oc + oo)
    fpr_open = co / (co + cc)
    print("True Positive Rate (Closed):", round(tpr_closed,3))
    print("True Positive Rate (Open):", round(tpr_open,3))
    print("False Positive Rate (Closed):", round(fpr_closed,3))
    print("False Positive Rate (Open):", round(fpr_open,3))

    print()
    scores_closed = -scores
    fpr_closed, tpr_closed, thresholds_closed = metrics.roc_curve(labels, scores_closed, pos_label="Closed")
    roc_auc_closed = metrics.auc(fpr_closed, tpr_closed)
    print("roc-auc (Closed) =", roc_auc_closed)
    roc_auc_disp_closed = metrics.RocCurveDisplay(fpr=fpr_closed, tpr=tpr_closed, roc_auc=roc_auc_closed)
    roc_auc_disp_closed.plot()
    # plt.show()
    plt.savefig(os.path.join(output_dir, 'roc_auc_closed.jpg'))

    fpr_open, tpr_open, thresholds_open = metrics.roc_curve(labels, scores, pos_label="Open")
    roc_auc_open = metrics.auc(fpr_open, tpr_open)
    print("roc-auc (Open) =", roc_auc_open)
    roc_auc_disp_open = metrics.RocCurveDisplay(fpr=fpr_open, tpr=tpr_open, roc_auc=roc_auc_open)
    roc_auc_disp_open.plot()
    # plt.show()
    plt.savefig(os.path.join(output_dir, 'roc_auc_open.jpg'))


def play_music(is_paused):
    if not is_paused:
        pygame.mixer.music.play(-1)
    else:
        pygame.mixer.music.unpause()
    print("Open Hand detected, Playing Music!")


def pause_music():
    pygame.mixer.music.pause()
    print("Closed Hand detected, Pausing Music!")


def capture_images_for_training(folder_path, num_images):
    cap = cv2.VideoCapture(0)
    count = 1
    enter_pressed = False
    
    while count < num_images:
        
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow('Capture', frame)

        if not enter_pressed:
            if cv2.waitKey(1) == 13:
                enter_pressed = True

        if enter_pressed:
            image_path = os.path.join(folder_path, f'image_{count}.jpg')
            cv2.imwrite(image_path, frame)
            count += 1

            image_path = os.path.join(folder_path, f'image_{count}.jpg')
            cv2.imwrite(image_path, frame)

    cap.release()
    cv2.destroyAllWindows()
    

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

patch_height = 128
patch_width = 64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Training

if part_id == 1:

    closed_train_dir_list = data_dir[0]
    open_train_dir_list = data_dir[1]

    # closed_train_dir_list = ['Final/closed1/train', 'Final/closed2/train', 'Final/closed3/train']
    # open_train_dir_list = ['Final/open1/train', 'Final/open2/train']

    train_examples_dir = os.path.join(output_dir, "train_examples")
    if not os.path.exists(train_examples_dir):
        os.makedirs(train_examples_dir)

    train_features = []
    train_labels = []


    folder_no = 0

    for closed_train_dir in closed_train_dir_list:

        folder_no += 1
        folder_path = closed_train_dir
        file_list = os.listdir(folder_path)

        for file_name in file_list:
            
            image_path = os.path.join(folder_path, file_name)
            base_filename, file_extension = os.path.splitext(os.path.basename(image_path))
            bgr_image = cv2.imread(image_path)
            bgr_image2 = bgr_image.copy()
            height_image, width_image = bgr_image.shape[:2]
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_image)

            if results.multi_hand_landmarks:

                hand_no = 0

                for hand_landmarks in results.multi_hand_landmarks:

                    hand_no += 1                
                    landmark_points = []

                    for landmark in hand_landmarks.landmark:

                        x = int(landmark.x * bgr_image.shape[1])
                        y = int(landmark.y * bgr_image.shape[0])
                        landmark_points.append([x, y])
                    
                    landmark_points = np.array(landmark_points)  
                    x, y, w, h = cv2.boundingRect(landmark_points) 
                    scale_factor = 1.3
                    x, y, w, h = increase_bbox((x, y, w, h), scale_factor, height_image, width_image)

                    cropped_positive_image = bgr_image[y:y+h, x:x+w]

                    resized_cropped_positive_image = cv2.resize(cropped_positive_image, (patch_width, patch_height))

                    gray_cropped_positive_image = cv2.cvtColor(resized_cropped_positive_image, cv2.COLOR_BGR2GRAY)

                    positive_hog_features, positive_hog_image = get_hog(gray_cropped_positive_image, 9, (8, 8), (2, 2))

                    train_features.append(positive_hog_features)
                    train_labels.append("Closed")

                    cv2.imwrite(os.path.join(train_examples_dir, f"closed{folder_no}_{base_filename}_hand{hand_no}.jpg"), resized_cropped_positive_image)
                    cv2.imwrite(os.path.join(train_examples_dir, f"closed{folder_no}_{base_filename}_hand{hand_no}_hog.jpg"), positive_hog_image)

                    if hand_no == 1:
                        color = (0, 100, 0)
                    else:
                        color = (255, 0, 0)
                        
                    cv2.rectangle(bgr_image2, (x, y), (x + w, y + h), color, 2)
                    add_text_in_bbox(bgr_image2, (x, y, x+w, y+h), "Closed", color)

                    print(closed_train_dir, " ", file_name, "  Hand No.", hand_no, "  Class: Closed")

                cv2.imwrite(os.path.join(train_examples_dir, f"closed{folder_no}_{base_filename}_train.jpg"), bgr_image2)

            else:

                add_text_in_bbox(bgr_image2, (0, 0, width_image, height_image), "No Hands Detected", (0, 100, 0))
                cv2.imwrite(os.path.join(train_examples_dir, f"closed{folder_no}_{base_filename}.jpg"), bgr_image2)
                
                print(closed_train_dir, " ", file_name, "  No Hands Detected")


    folder_no = 0

    for open_train_dir in open_train_dir_list:

        folder_no += 1
        folder_path = open_train_dir
        file_list = os.listdir(folder_path)

        for file_name in file_list:
            
            image_path = os.path.join(folder_path, file_name)
            base_filename, file_extension = os.path.splitext(os.path.basename(image_path))
            bgr_image = cv2.imread(image_path)
            bgr_image2 = bgr_image.copy()
            height_image, width_image = bgr_image.shape[:2]
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_image)

            if results.multi_hand_landmarks:

                hand_no = 0

                for hand_landmarks in results.multi_hand_landmarks:

                    hand_no += 1                
                    landmark_points = []

                    for landmark in hand_landmarks.landmark:

                        x = int(landmark.x * bgr_image.shape[1])
                        y = int(landmark.y * bgr_image.shape[0])
                        landmark_points.append([x, y])
                    
                    landmark_points = np.array(landmark_points)  
                    x, y, w, h = cv2.boundingRect(landmark_points) 
                    scale_factor = 1.3
                    x, y, w, h = increase_bbox((x, y, w, h), scale_factor, height_image, width_image)

                    cropped_positive_image = bgr_image[y:y+h, x:x+w]

                    resized_cropped_positive_image = cv2.resize(cropped_positive_image, (patch_width, patch_height))

                    gray_cropped_positive_image = cv2.cvtColor(resized_cropped_positive_image, cv2.COLOR_BGR2GRAY)

                    positive_hog_features, positive_hog_image = get_hog(gray_cropped_positive_image, 9, (8, 8), (2, 2))

                    train_features.append(positive_hog_features)
                    train_labels.append("Open")

                    cv2.imwrite(os.path.join(train_examples_dir, f"open{folder_no}_{base_filename}_hand{hand_no}.jpg"), resized_cropped_positive_image)
                    cv2.imwrite(os.path.join(train_examples_dir, f"open{folder_no}_{base_filename}_hand{hand_no}_hog.jpg"), positive_hog_image)

                    if hand_no == 1:
                        color = (0, 100, 0)
                    else:
                        color = (255, 0, 0)
                        
                    cv2.rectangle(bgr_image2, (x, y), (x + w, y + h), color, 2)
                    add_text_in_bbox(bgr_image2, (x, y, x+w, y+h), "Open", color)

                    print(open_train_dir, " ", file_name, "  Hand No.", hand_no, "  Class: Open")

                cv2.imwrite(os.path.join(train_examples_dir, f"open{folder_no}_{base_filename}_train.jpg"), bgr_image2)

            else:

                add_text_in_bbox(bgr_image2, (0, 0, width_image, height_image), "No Hands Detected", (0, 100, 0))
                cv2.imwrite(os.path.join(train_examples_dir, f"open{folder_no}_{base_filename}.jpg"), bgr_image2)
                
                print(open_train_dir, " ", file_name, "  No Hands Detected")


    position_svm = svm.SVC(kernel='linear')
    position_svm.fit(train_features, train_labels)
    dump(position_svm, os.path.join(output_dir, 'position_svm.joblib'))

    train_labels =np.array(train_labels)
    # print(train_labels)

    train_predictions = position_svm.predict(train_features)
    # print(train_predictions)

    train_scores = position_svm.decision_function(train_features)
    # print(train_scores)

    print("Training Dataset Evaluation:")
    evaluate (train_labels, train_predictions, train_scores, output_dir)


# Validation

elif part_id == 2:

    closed_valid_dir_list = data_dir[0]
    open_valid_dir_list = data_dir[1]

    # closed_valid_dir_list = ['Final/closed1/valid', 'Final/closed2/valid', 'Final/closed3/valid']
    # open_valid_dir_list = ['Final/open1/valid', 'Final/open2/valid']

    position_svm = load(model_dir)

    valid_examples_dir = os.path.join(output_dir, "valid_examples")
    if not os.path.exists(valid_examples_dir):
        os.makedirs(valid_examples_dir)

    valid_dict = {
        'Directory': [],
        'Image Name': [],
        'Hand No.': [],
        'Prediction (Open/Closed)': []
    }

    valid_features = []
    valid_labels = []
    valid_predictions = []


    folder_no = 0

    for closed_valid_dir in closed_valid_dir_list:

        folder_no += 1
        folder_path = closed_valid_dir
        file_list = os.listdir(folder_path)

        for file_name in file_list:
            
            image_path = os.path.join(folder_path, file_name)
            base_filename, file_extension = os.path.splitext(os.path.basename(image_path))
            bgr_image = cv2.imread(image_path)
            bgr_image2 = bgr_image.copy()
            height_image, width_image = bgr_image.shape[:2]
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_image)

            if results.multi_hand_landmarks:

                hand_no = 0

                for hand_landmarks in results.multi_hand_landmarks:

                    hand_no += 1
                    
                    landmark_points = []

                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * bgr_image.shape[1])
                        y = int(landmark.y * bgr_image.shape[0])
                        landmark_points.append([x, y])
                    
                    landmark_points = np.array(landmark_points)  
                    x, y, w, h = cv2.boundingRect(landmark_points) 
                    scale_factor = 1.3
                    x, y, w, h = increase_bbox((x, y, w, h), scale_factor, height_image, width_image)

                    cropped_positive_image = bgr_image[y:y+h, x:x+w]

                    resized_cropped_positive_image = cv2.resize(cropped_positive_image, (patch_width, patch_height))

                    gray_cropped_positive_image = cv2.cvtColor(resized_cropped_positive_image, cv2.COLOR_BGR2GRAY)

                    positive_hog_features, positive_hog_image = get_hog(gray_cropped_positive_image, 9, (8, 8), (2, 2))

                    valid_features.append(positive_hog_features)
                    valid_labels.append("Closed")

                    prediction = position_svm.predict(positive_hog_features.reshape(1, -1))[0]
                    valid_predictions.append(prediction)

                    cv2.imwrite(os.path.join(valid_examples_dir, f"closed{folder_no}_{base_filename}_hand{hand_no}.jpg"), resized_cropped_positive_image)
                    cv2.imwrite(os.path.join(valid_examples_dir, f"closed{folder_no}_{base_filename}_hand{hand_no}_hog.jpg"), positive_hog_image)

                    if hand_no == 1:
                        color = (0, 100, 0)
                    else:
                        color = (255, 0, 0)
                        
                    cv2.rectangle(bgr_image2, (x, y), (x + w, y + h), color, 2)
                    add_text_in_bbox(bgr_image2, (x, y, x+w, y+h), prediction, color)

                    valid_dict['Directory'].append(closed_valid_dir)
                    valid_dict['Image Name'].append(f"{base_filename}.jpg")
                    valid_dict['Hand No.'].append(hand_no)
                    valid_dict['Prediction (Open/Closed)'].append(prediction)

                    print(closed_valid_dir, " ", file_name, "  Hand No.", hand_no, "  Prediction:", prediction)

                cv2.imwrite(os.path.join(valid_examples_dir, f"closed{folder_no}_{base_filename}_prediction.jpg"), bgr_image2)

            else:

                valid_dict['Directory'].append(closed_valid_dir)
                valid_dict['Image Name'].append(f"{base_filename}.jpg")
                valid_dict['Hand No.'].append("No Hands Detected")
                valid_dict['Prediction (Open/Closed)'].append("No Hands Detected")

                add_text_in_bbox(bgr_image2, (0, 0, width_image, height_image), "No Hands Detected", (0, 100, 0))
                cv2.imwrite(os.path.join(valid_examples_dir, f"closed{folder_no}_{base_filename}_prediction.jpg"), bgr_image2)
                
                print(closed_valid_dir, " ", file_name, "  No Hands Detected")


    folder_no = 0

    for open_valid_dir in open_valid_dir_list:

        folder_no += 1
        folder_path = open_valid_dir
        file_list = os.listdir(folder_path)

        for file_name in file_list:
            
            image_path = os.path.join(folder_path, file_name)
            base_filename, file_extension = os.path.splitext(os.path.basename(image_path))
            bgr_image = cv2.imread(image_path)
            bgr_image2 = bgr_image.copy()
            height_image, width_image = bgr_image.shape[:2]
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_image)

            if results.multi_hand_landmarks:

                hand_no = 0

                for hand_landmarks in results.multi_hand_landmarks:

                    hand_no += 1
                    
                    landmark_points = []

                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * bgr_image.shape[1])
                        y = int(landmark.y * bgr_image.shape[0])
                        landmark_points.append([x, y])
                    
                    landmark_points = np.array(landmark_points)  
                    x, y, w, h = cv2.boundingRect(landmark_points) 
                    scale_factor = 1.3
                    x, y, w, h = increase_bbox((x, y, w, h), scale_factor, height_image, width_image)

                    cropped_positive_image = bgr_image[y:y+h, x:x+w]

                    resized_cropped_positive_image = cv2.resize(cropped_positive_image, (patch_width, patch_height))

                    gray_cropped_positive_image = cv2.cvtColor(resized_cropped_positive_image, cv2.COLOR_BGR2GRAY)

                    positive_hog_features, positive_hog_image = get_hog(gray_cropped_positive_image, 9, (8, 8), (2, 2))

                    valid_features.append(positive_hog_features)
                    valid_labels.append("Open")

                    prediction = position_svm.predict(positive_hog_features.reshape(1, -1))[0]
                    valid_predictions.append(prediction)

                    cv2.imwrite(os.path.join(valid_examples_dir, f"open{folder_no}_{base_filename}_hand{hand_no}.jpg"), resized_cropped_positive_image)
                    cv2.imwrite(os.path.join(valid_examples_dir, f"open{folder_no}_{base_filename}_hand{hand_no}_hog.jpg"), positive_hog_image)

                    if hand_no == 1:
                        color = (0, 100, 0)
                    else:
                        color = (255, 0, 0)
                        
                    cv2.rectangle(bgr_image2, (x, y), (x + w, y + h), color, 2)
                    add_text_in_bbox(bgr_image2, (x, y, x+w, y+h), prediction, color)

                    valid_dict['Directory'].append(open_valid_dir)
                    valid_dict['Image Name'].append(f"{base_filename}.jpg")
                    valid_dict['Hand No.'].append(hand_no)
                    valid_dict['Prediction (Open/Closed)'].append(prediction)

                    print(open_valid_dir, " ", file_name, "  Hand No.", hand_no, "  Prediction:", prediction)

                cv2.imwrite(os.path.join(valid_examples_dir, f"open{folder_no}_{base_filename}_prediction.jpg"), bgr_image2)

            else:

                valid_dict['Directory'].append(open_valid_dir)
                valid_dict['Image Name'].append(f"{base_filename}.jpg")
                valid_dict['Hand No.'].append("No Hands Detected")
                valid_dict['Prediction (Open/Closed)'].append("No Hands Detected")

                add_text_in_bbox(bgr_image2, (0, 0, width_image, height_image), "No Hands Detected", (0, 100, 0))
                cv2.imwrite(os.path.join(valid_examples_dir, f"open{folder_no}_{base_filename}_prediction.jpg"), bgr_image2)
                
                print(open_valid_dir, " ", file_name, "  No Hands Detected")


    valid_df = pd.DataFrame(valid_dict)
    valid_df.to_excel(os.path.join(output_dir, 'Valid_Predictions.xlsx'), index=False)

    valid_labels =np.array(valid_labels)
    # print(valid_labels)

    valid_predictions =np.array(valid_predictions)
    # print(valid_predictions)

    valid_scores = position_svm.decision_function(valid_features)
    # print(valid_scores)

    print("Validation Dataset Evaluation:")
    evaluate (valid_labels, valid_predictions, valid_scores, output_dir)


# Testing

elif part_id == 3:

    test_dir_list = data_dir

    # test_dir_list = ['Final/closed1/valid']

    position_svm = load(model_dir)

    test_examples_dir = os.path.join(output_dir, "test_examples")
    if not os.path.exists(test_examples_dir):
        os.makedirs(test_examples_dir)

    test_dict = {
        'Directory': [],
        'Image Name': [],
        'Hand No.': [],
        'Prediction (Open/Closed)': []
    }

    test_features = []
    test_predictions = []


    folder_no = 0

    for test_dir in test_dir_list:

        folder_no += 1
        folder_path = test_dir
        file_list = os.listdir(folder_path)

        for file_name in file_list:
            
            image_path = os.path.join(folder_path, file_name)
            base_filename, file_extension = os.path.splitext(os.path.basename(image_path))
            bgr_image = cv2.imread(image_path)
            bgr_image2 = bgr_image.copy()
            height_image, width_image = bgr_image.shape[:2]
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_image)

            if results.multi_hand_landmarks:

                hand_no = 0

                for hand_landmarks in results.multi_hand_landmarks:

                    hand_no += 1
                    
                    landmark_points = []

                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * bgr_image.shape[1])
                        y = int(landmark.y * bgr_image.shape[0])
                        landmark_points.append([x, y])
                    
                    landmark_points = np.array(landmark_points)  
                    x, y, w, h = cv2.boundingRect(landmark_points) 
                    scale_factor = 1.3
                    x, y, w, h = increase_bbox((x, y, w, h), scale_factor, height_image, width_image)

                    cropped_positive_image = bgr_image[y:y+h, x:x+w]

                    resized_cropped_positive_image = cv2.resize(cropped_positive_image, (patch_width, patch_height))

                    gray_cropped_positive_image = cv2.cvtColor(resized_cropped_positive_image, cv2.COLOR_BGR2GRAY)

                    positive_hog_features, positive_hog_image = get_hog(gray_cropped_positive_image, 9, (8, 8), (2, 2))

                    test_features.append(positive_hog_features)

                    prediction = position_svm.predict(positive_hog_features.reshape(1, -1))[0]
                    test_predictions.append(prediction)

                    cv2.imwrite(os.path.join(test_examples_dir, f"{folder_no}_{base_filename}_hand{hand_no}.jpg"), resized_cropped_positive_image)
                    cv2.imwrite(os.path.join(test_examples_dir, f"{folder_no}_{base_filename}_hand{hand_no}_hog.jpg"), positive_hog_image)

                    if hand_no == 1:
                        color = (0, 100, 0)
                    else:
                        color = (255, 0, 0)
                        
                    cv2.rectangle(bgr_image2, (x, y), (x + w, y + h), color, 2)
                    add_text_in_bbox(bgr_image2, (x, y, x+w, y+h), prediction, color)

                    test_dict['Directory'].append(test_dir)
                    test_dict['Image Name'].append(f"{base_filename}.jpg")
                    test_dict['Hand No.'].append(hand_no)
                    test_dict['Prediction (Open/Closed)'].append(prediction)

                    print(test_dir, " ", file_name, "  Hand No.", hand_no, "  Prediction:", prediction)

                cv2.imwrite(os.path.join(test_examples_dir, f"{folder_no}_{base_filename}_prediction.jpg"), bgr_image2)

            else:

                test_dict['Directory'].append(test_dir)
                test_dict['Image Name'].append(f"{base_filename}.jpg")
                test_dict['Hand No.'].append("No Hands Detected")
                test_dict['Prediction (Open/Closed)'].append("No Hands Detected")

                add_text_in_bbox(bgr_image2, (0, 0, width_image, height_image), "No Hands Detected", (0, 100, 0))
                cv2.imwrite(os.path.join(test_examples_dir, f"{folder_no}_{base_filename}_prediction.jpg"), bgr_image2)
                
                print(test_dir, " ", file_name, "  No Hands Detected")


    test_df = pd.DataFrame(test_dict)
    test_df.to_excel(os.path.join(output_dir, 'Test_Predictions.xlsx'), index=False)


# Application
    
elif part_id == 4:

    num_images = 100

    closed_train_dir = os.path.join(output_dir, "captured_images/closed")
    if not os.path.exists(closed_train_dir):
        os.makedirs(closed_train_dir)
    print("Capturing Closed Hand Images for Training")
    capture_images_for_training(closed_train_dir, num_images)

    open_train_dir = os.path.join(output_dir, "captured_images/open")
    if not os.path.exists(open_train_dir):
        os.makedirs(open_train_dir)
    print("Capturing Open Hand Images for Training")
    capture_images_for_training(open_train_dir, num_images)

    train_examples_dir = os.path.join(output_dir, "train_examples")
    if not os.path.exists(train_examples_dir):
        os.makedirs(train_examples_dir)

    train_features = []
    train_labels = []

    folder_path = closed_train_dir
    file_list = os.listdir(folder_path)

    for file_name in file_list:
        
        image_path = os.path.join(folder_path, file_name)
        base_filename, file_extension = os.path.splitext(os.path.basename(image_path))
        bgr_image = cv2.imread(image_path)
        bgr_image2 = bgr_image.copy()
        height_image, width_image = bgr_image.shape[:2]
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:

            hand_no = 0

            for hand_landmarks in results.multi_hand_landmarks:

                hand_no += 1                
                landmark_points = []

                for landmark in hand_landmarks.landmark:

                    x = int(landmark.x * bgr_image.shape[1])
                    y = int(landmark.y * bgr_image.shape[0])
                    landmark_points.append([x, y])
                
                landmark_points = np.array(landmark_points)  
                x, y, w, h = cv2.boundingRect(landmark_points) 
                scale_factor = 1.3
                x, y, w, h = increase_bbox((x, y, w, h), scale_factor, height_image, width_image)

                cropped_positive_image = bgr_image[y:y+h, x:x+w]

                resized_cropped_positive_image = cv2.resize(cropped_positive_image, (patch_width, patch_height))

                gray_cropped_positive_image = cv2.cvtColor(resized_cropped_positive_image, cv2.COLOR_BGR2GRAY)

                positive_hog_features, positive_hog_image = get_hog(gray_cropped_positive_image, 9, (8, 8), (2, 2))

                train_features.append(positive_hog_features)
                train_labels.append("Closed")

                cv2.imwrite(os.path.join(train_examples_dir, f"closed{base_filename}_hand{hand_no}.jpg"), resized_cropped_positive_image)
                cv2.imwrite(os.path.join(train_examples_dir, f"closed{base_filename}_hand{hand_no}_hog.jpg"), positive_hog_image)

                if hand_no == 1:
                    color = (0, 100, 0)
                else:
                    color = (255, 0, 0)
                    
                cv2.rectangle(bgr_image2, (x, y), (x + w, y + h), color, 2)
                add_text_in_bbox(bgr_image2, (x, y, x+w, y+h), "Closed", color)

                print(closed_train_dir, " ", file_name, "  Hand No.", hand_no, "  Class: Closed")

            cv2.imwrite(os.path.join(train_examples_dir, f"closed{base_filename}_train.jpg"), bgr_image2)

        else:

            add_text_in_bbox(bgr_image2, (0, 0, width_image, height_image), "No Hands Detected", (0, 100, 0))
            cv2.imwrite(os.path.join(train_examples_dir, f"closed{base_filename}.jpg"), bgr_image2)
            
            print(closed_train_dir, " ", file_name, "  No Hands Detected")

    folder_path = open_train_dir
    file_list = os.listdir(folder_path)

    for file_name in file_list:
        
        image_path = os.path.join(folder_path, file_name)
        base_filename, file_extension = os.path.splitext(os.path.basename(image_path))
        bgr_image = cv2.imread(image_path)
        bgr_image2 = bgr_image.copy()
        height_image, width_image = bgr_image.shape[:2]
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:

            hand_no = 0

            for hand_landmarks in results.multi_hand_landmarks:

                hand_no += 1                
                landmark_points = []

                for landmark in hand_landmarks.landmark:

                    x = int(landmark.x * bgr_image.shape[1])
                    y = int(landmark.y * bgr_image.shape[0])
                    landmark_points.append([x, y])
                
                landmark_points = np.array(landmark_points)  
                x, y, w, h = cv2.boundingRect(landmark_points) 
                scale_factor = 1.3
                x, y, w, h = increase_bbox((x, y, w, h), scale_factor, height_image, width_image)

                cropped_positive_image = bgr_image[y:y+h, x:x+w]

                resized_cropped_positive_image = cv2.resize(cropped_positive_image, (patch_width, patch_height))

                gray_cropped_positive_image = cv2.cvtColor(resized_cropped_positive_image, cv2.COLOR_BGR2GRAY)

                positive_hog_features, positive_hog_image = get_hog(gray_cropped_positive_image, 9, (8, 8), (2, 2))

                train_features.append(positive_hog_features)
                train_labels.append("Open")

                cv2.imwrite(os.path.join(train_examples_dir, f"open{base_filename}_hand{hand_no}.jpg"), resized_cropped_positive_image)
                cv2.imwrite(os.path.join(train_examples_dir, f"open{base_filename}_hand{hand_no}_hog.jpg"), positive_hog_image)

                if hand_no == 1:
                    color = (0, 100, 0)
                else:
                    color = (255, 0, 0)
                    
                cv2.rectangle(bgr_image2, (x, y), (x + w, y + h), color, 2)
                add_text_in_bbox(bgr_image2, (x, y, x+w, y+h), "Open", color)

                print(open_train_dir, " ", file_name, "  Hand No.", hand_no, "  Class: Open")

            cv2.imwrite(os.path.join(train_examples_dir, f"open{base_filename}_train.jpg"), bgr_image2)

        else:

            add_text_in_bbox(bgr_image2, (0, 0, width_image, height_image), "No Hands Detected", (0, 100, 0))
            cv2.imwrite(os.path.join(train_examples_dir, f"open{base_filename}.jpg"), bgr_image2)
            
            print(open_train_dir, " ", file_name, "  No Hands Detected")


    position_svm = svm.SVC(kernel='linear')
    position_svm.fit(train_features, train_labels)
    dump(position_svm, os.path.join(output_dir, 'position_svm.joblib'))

    train_labels =np.array(train_labels)

    train_predictions = position_svm.predict(train_features)

    train_scores = position_svm.decision_function(train_features)

    print("Training Dataset Evaluation:")
    evaluate (train_labels, train_predictions, train_scores, output_dir)


    pygame.init()
    pygame.mixer.music.load(model_dir)

    position_svm = load(os.path.join(output_dir, 'position_svm.joblib'))

    cap = cv2.VideoCapture(0)

    is_paused = False

    while True:

        print("Open Hand = Play Music, Closed Hand = Pause Music")
        
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow('Capture', frame)

        bgr_image = frame
        bgr_image2 = bgr_image.copy()
        height_image, width_image = bgr_image.shape[:2]
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:

            hand_no = 0

            for hand_landmarks in results.multi_hand_landmarks:

                hand_no += 1
                
                landmark_points = []

                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * bgr_image.shape[1])
                    y = int(landmark.y * bgr_image.shape[0])
                    landmark_points.append([x, y])
                
                landmark_points = np.array(landmark_points)  
                x, y, w, h = cv2.boundingRect(landmark_points) 
                scale_factor = 1.3
                x, y, w, h = increase_bbox((x, y, w, h), scale_factor, height_image, width_image)

                cropped_positive_image = bgr_image[y:y+h, x:x+w]

                resized_cropped_positive_image = cv2.resize(cropped_positive_image, (patch_width, patch_height))

                gray_cropped_positive_image = cv2.cvtColor(resized_cropped_positive_image, cv2.COLOR_BGR2GRAY)

                positive_hog_features, positive_hog_image = get_hog(gray_cropped_positive_image, 9, (8, 8), (2, 2))

                prediction = position_svm.predict(positive_hog_features.reshape(1, -1))[0]

                if prediction == "Open":
                    play_music(is_paused)
                    is_paused = True
                elif prediction == "Closed":
                    pause_music()

        key = cv2.waitKey(1)
        if key == 13:
            pygame.mixer.music.pause()
            break

    cap.release()
    cv2.destroyAllWindows()