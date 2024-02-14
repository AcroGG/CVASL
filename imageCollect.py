import os
import cv2


#1. Data Image Collection
#2. Data Processing
#     a. processing and choosing only our hands. Adding data landmarks
#     b. Saving this data pickling? dill? json? numpy.save?
#3. Training our model
#    a. using the saved byte data and numpy arrays train using tensor flow? scikit?
#    b. save model into another byte format to save from retraining over and over saving resources and time. 
#4. Sign Language classifier
#    a. actually showing our model works live 
#    b. import the saved model data to match with our labels 
#   c. draw hand landmarks/outline
#4. Data Addition
#    a. create a script that can append data into the data archive as a new and last folder instead of having to do all the hand signals again handling new phrases or letters.

Data_Dir = './archive'
if not os.path.exists(Data_Dir):
    os.makedirs(Data_Dir)

num_classes = 4 # update to include the entire library or small phrases. 
data_size = 100 

cap = cv2.VideoCapture(0)

for j in range(num_classes):
    if not os.path.exists(os.path.join(Data_Dir, str(j))):
        os.makedirs(os.path.join(Data_Dir, str(j)))
    
    print('collecting data for class{}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'When Ready press "Q"', (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break
    
    count = 0
    while count < data_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(Data_Dir, str(j), '{}.jpg'.format(count)), frame)
        count+=1
    
cap.release()
cv2.destroyAllWindows()

