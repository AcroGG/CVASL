import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.python.solutions.drawing_utils import draw_landmarks_on_image
# Can only handle one hand at a time adding two creates a value error for the model
# it is expecting one hand of 42 features 2 hands creates 84 features which the model is not expecting
# adding two hands on image entry also creates this error but wont let you run the model due to the array to be trained
# has an inhomogeneous shape after 1 demensions the detected shape is (400,) + inhomogeneous part
# which creates value error. 

# Must update create data to be able to handle two hands at once.
# Or get the model to be able to ingnore the inhomogeneous shaped array in the data pickle file. 
base = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
hands = mp.solutions.hands

drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

hands = hands.Hands(static_image_mode=False, max_num_hands = 2, min_detection_confidence=0.3)

Data_Dir = './archive'

data = []
labels = []

for dir_ in os.listdir(Data_Dir):
    for img_path in os.listdir(os.path.join(Data_Dir, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(Data_Dir, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
