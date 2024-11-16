

import cv2
import argparse
import time
import os
import Update_Model
import glob
import random
import eel
import light
import winsound
import pyttsx3

frequency=2500
duration=1000
eel.init('WD')
emotions=["angry", "happy", "sad", "neutral"]
fishface = cv2.face.FisherFaceRecognizer_create()
font = cv2.FONT_HERSHEY_SIMPLEX
'''try:
    fishface.load("model.xml")
except:
    print("No trained model found... --update will create one.")'''

parser=argparse.ArgumentParser(description="Options for emotions based music player(Updating the model)")
parser.add_argument("--update", help="Call for taking new images and retraining the model.", action="store_true")
args=parser.parse_args()    
facedict={}
video_capture=cv2.VideoCapture(0)
facecascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def crop(clahe_image, face):
    for (x, y, w, h) in face:
        faceslice=clahe_image[y:y+h, x:x+w]
        faceslice=cv2.resize(faceslice, (350, 350))
        facedict["face%s" %(len(facedict)+1)]=faceslice
    return faceslice

def showImage(frame):
    cv2.imshow('window_frame', frame)

def grab_face():
    #ret, frame=video_capture.read()#1
    #gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#
    #rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#
    #bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)#
    #cv2.imshow('window_frame', bgr_image)#
    ret, frame=light.nolight()
    
    #cv2.imshow("Video", frame)#2
    cv2.imwrite('test.jpg', frame)
    cv2.imwrite("images/main%s.jpg" %count, frame)
    gray=cv2.imread('test.jpg',0)
    #gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#3
    #noise amplification
    clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image=clahe.apply(gray)
    return clahe_image

def detect_face():
    clahe_image=grab_face()
    #acalefactor == reduce image size. minneigh = how many face neighbours
    face=facecascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(face)>=1:
        faceslice=crop(clahe_image, face)        
        #return faceslice
    else:
        engine = pyttsx3.init()
        engine.say('face not detected')
        engine.runAndWait()
        print("No/Multiple faces detected!!, passing over the frame")

def save_face(emotion):
    engine = pyttsx3.init()
    
    engine.say("Look "+emotion+" untill the timer expires and keep the same emotion for some time.")
    engine.runAndWait()
    print("\n\nLook "+emotion+" untill the timer expires and keep the same emotion for some time.")
    winsound.Beep(frequency, duration)
    #print('\a')
    
    
    for i in range(0, 5):
        print(5-i)
        time.sleep(2)
    
    while len(facedict.keys())<16:
        detect_face()

    for i in facedict.keys():
        path, dirs, files = next(os.walk("dataset/%s" %emotion))
        file_count = len(files)+1
        cv2.imwrite("dataset/%s/%s.jpg" %(emotion, (file_count)), facedict[i])
    facedict.clear()

def update_model(emotions):
    engine = pyttsx3.init()
    engine.say("show your emotions one by one to train")
    engine.runAndWait()
    print("Update mode for model is ready")
    checkForFolders(emotions)
    
    for i in range(0, len(emotions)):
        save_face(emotions[i])
    print("Collected the images, looking nice! Now updating the model...")
    Update_Model.update(emotions)
    engine = pyttsx3.init()
    engine.say("Model trained")
    engine.runAndWait()
    print("Model train successful!!")

def checkForFolders(emotions):
    for emotion in emotions:
        if os.path.exists("dataset/%s" %emotion):
            pass
        else:
            os.makedirs("dataset/%s" %emotion)

def identify_emotions():
    prediction=[]
    confidence=[]

    for i in facedict.keys():
        pred, conf=fishface.predict(facedict[i])
        cv2.imwrite("images/%s.jpg" %i, facedict[i])
        prediction.append(pred)
        confidence.append(conf)
        
        
    output=emotions[max(set(prediction), key=prediction.count)]
    engine = pyttsx3.init()
    engine.say("You seem to be"+output)
    engine.runAndWait()
    
    print("You seem to be %s" % output)
    #print("confidence %s" %conf)
    facedict.clear()
    return output;
    #songlist=[]
    #songlist=sorted(glob.glob("songs/%s/*" %output))
    #random.shuffle(songlist)
    #os.startfile(songlist[0])
count=0
@eel.expose
def getEmotion():
    engine = pyttsx3.init()
    engine.say("capturing your emotion")
    engine.runAndWait()
   
    count=0
    #t=threading.Thread(target=detect_face)
    #t.start()
    while True:
        count=count+1
        
        detect_face()
        if args.update:
            update_model(emotions)
            break
        elif count==10:
            fishface.read("model2.xml")
            
            
            return identify_emotions()
            break

eel.start('main.html')
    
