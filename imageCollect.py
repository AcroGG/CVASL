import os
import cv2

Data_Dir = './archive'
if not os.path.exists(Data_Dir):
    os.makedirs(Data_Dir)

num_classes = 3
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

