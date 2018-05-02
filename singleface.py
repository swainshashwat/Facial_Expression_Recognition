import cv2
import sys
from em_model import EMR
import numpy as np

EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

# initialize the cascade
cascade_classifier = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')  

def format_image(image):
    """
    Function to format frame
    """
    if len(image.shape) > 2 and image.shape[2] == 3:
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    faces = cascade_classifier.detectMultiScale(image,scaleFactor = 1.3 ,minNeighbors = 5)

    if not len(faces) > 0:
        return None

    
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    face = max_area_face

    
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

    try:
        image = cv2.resize(image, (48,48), interpolation = cv2.INTER_CUBIC) / 255.
    except Exception:
        print("----->Problem during resize")
        return None

    return image


network = EMR()
network.build_network()

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
feelings_faces = []


for index, emotion in enumerate(EMOTIONS):
    feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))

while True:
    ret, frame = cap.read()
    facecasc = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, 1.3, 5)

    
    result = network.predict(format_image(frame))
    if result is not None:
        if result[0][6] < 0.6:
            result[0][6] = result[0][6] - 0.12
            result[0][:3] += 0.01
            result[0][4:5] += 0.04
        
        for index, emotion in enumerate(EMOTIONS):
            cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1);
            cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4), (255, 0, 0), -1)

        
        maxindex = np.argmax(result[0])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,EMOTIONS[maxindex],(10,360), font, 2,(255,255,255),2,cv2.LINE_AA) 
        face_image = feelings_faces[maxindex]
        print(face_image[:,:,3])

        for c in range(0, 3):
            
            # frame[200:320,10:130,c] = frame[200:320, 10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)
            frame[200:320, 10:130, c] = face_image[:,:,c] * (face_image[:, :,3 ] / 255.0) +  frame[200:320, 10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)

    if not len(faces) > 0:
        
        a = 1
    else:
        
        max_area_face = faces[0]
        for face in faces:
            if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
                max_area_face = face
        face = max_area_face
        (x,y,w,h) = max_area_face
        frame = cv2.rectangle(frame,(x,y-50),(x+w,y+h+10),(255,0,0),2)

    cv2.imshow('Video', cv2.resize(frame,None,fx=2,fy=2,interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()