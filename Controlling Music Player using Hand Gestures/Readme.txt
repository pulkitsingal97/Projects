Terminal Command: python3 main.py <part_id> <model/mp3file_dir> <data_dir_list> <output_dir>

Task 1: Training
Example usage: python3 main.py 1 _ "[['Final/closed1/train', 'Final/closed2/train', 'Final/closed3/train'], ['Final/open1/train', 'Final/open2/train']]" train_op

Task 2: Validation
Example usage: python3 main.py 2 train_op/position_svm.joblib "[['Final/closed1/valid', 'Final/closed2/valid', 'Final/closed3/valid'], ['Final/open1/valid', 'Final/open2/valid']]" valid_op

Task 3: Testing
Example usage: python3 main.py 3 train_op/position_svm.joblib "['Final/closed1/valid', 'Final/closed2/valid', 'Final/closed3/valid', 'Final/open1/valid', 'Final/open2/valid']" test_op

Task 4: Application
Example usage: python3 main.py 4 Welcome.mp3 "[]" app_op


Structure of main.py:
Importing of the necessary Libraries
Function Definitions (convolution, image gradients, finding and plotting HOGs, SVM model evaluation, image capturing, music play/pause)
Code for Training and Saving of SVM Model
Code for Validation of SVM Model
Code for Testing of SVM Model
Code for Real Life Application of SVM Model (Controlling music playing in Laptop using hand gestures)