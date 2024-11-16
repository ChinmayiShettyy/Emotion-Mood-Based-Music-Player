# OpenCV program to detect face in real time 
# import libraries of python OpenCV 
# where its functionality resides 
import cv2 
import numpy as np
# load the required trained XML classifiers 
# https://github.com/Itseez/opencv/blob/master/ 
# data/haarcascades/haarcascade_frontalface_default.xml 
# Trained XML classifiers describes some features of some 
# object we want to detect a cascade function is trained 
# from a lot of positive(faces) and negative(non-faces) 
# images.
def display(emotion):
    condition= True
    print("inside function",emotion)
    
        
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
    color=np.asarray((255, 255, 0))
    color = color.astype(int)
    color = color.tolist()
    # https://github.com/Itseez/opencv/blob/master 
    # /data/haarcascades/haarcascade_eye.xml 
    # Trained XML file for detecting eyes 
    #eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
    def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                    font_scale=2, thickness=2):
        x, y = coordinates[:2]
        cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)
        """cv2.putText(image_array, emotion_probability, (x + x_offset, y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)"""	
    # capture frames from a camera 
    cap = cv2.VideoCapture(0) 

    # loop runs if capturing has been initialized. 
    while condition: 

            # reads frames from a camera 
            ret, img = cap.read() 

            # convert to gray scale of each frames 
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

            # Detects faces of different sizes in the input image 
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            

            for (x,y,w,h) in faces: 
                    # To draw a rectangle in a face 
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
                    roi_gray = gray[y:y+h, x:x+w] 
                    roi_color = img[y:y+h, x:x+w]
                    draw_text((x,y), img, emotion,
                     color , 0, -45, 1, 1)

                    # Detects eyes of different sizes in the input image 
                    #eyes = eye_cascade.detectMultiScale(roi_gray) 

                    #To draw a rectangle in eyes 
                    '''for (ex,ey,ew,eh) in eyes: 
                            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2) '''

            # Display an image in a window 
            cv2.imshow('img',img)
            
            

            # Wait for Esc key to stop 
            #k = cv2.waitKey(2000) 
            if cv2.waitKey(1): 
                    break
            # Close the window
            cap.release()
            # De-allocate any associated memory usage
            cv2.destroyAllWindows()
            condition=False
    return
if __name__ == "__main__":
    display()
