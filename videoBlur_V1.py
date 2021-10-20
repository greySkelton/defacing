#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

#thid function can be called to display images. Useful tool
def plotImages(img):
    plt.imshow(img, cmap="gray")
    plt.axis('off')
    plt.style.use('seaborn')
    plt.show()


# In[2]:


#this function inputs an image, detects a face, and outputs the image with a blurred face
def averageBlur(image):

    # convert BGR image into RGB image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_data = face_detect.detectMultiScale(image, 1.3, 5)

    # Draw rectangle around the faces which is our region of interest (ROI)
    for (x, y, w, h) in face_data:
        cv2.rectangle(image, (x, y-30), (x + w, y + h+30), (0, 255, 0), 2)
        roi = image[y:y+h, x:x+w]
        # applying a blur over this new rectangle area
        roi = cv2.blur(roi, (81, 81), 30) #the 2nd parameter is a tuple indicating blur strength. Must be odd
        # impose this blurred image on original image to get final image
        image[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
    return image


# In[5]:


#This code should break a video into frame and blur each frame. 
#Main problem is that it turns mp4 videos into a series of bluish pictures
def videoToFrames(videoPathName, videoName, path):
    vidcap = cv2.VideoCapture(videoPathName)
    success,image = vidcap.read()
    
    dimensions = image.shape
    height = dimensions[0]
    width = dimensions[1]
    
    count = 0
    frames = 20 #number here is amount of Frames to pull
    while count < frames: 
        success,image = vidcap.read() #image is a single frame
        image = averageBlur(image)
        cv2.imwrite(path+videoName+"%d.jpg" % count, image)     # save frame as JPEG file
        count += 1
    framesToVideo(videoName,path,frames) 


# In[17]:


#This code compile a series of images into a video.
#Problem is the timing is way off. Currently makes video at 1fps
def framesToVideo(videoName,path,frames):
    count = 0
    imagex = cv2.imread(path+videoName+"%d.jpg" % count) #imagex is the dir path + <videoname><count>.jpg
    height,width,layers=imagex.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(path+'blurred.avi', fourcc, 1, (width, height))
    
    while count < frames:
        imagex = cv2.imread(path+videoName+"%d.jpg" % count) #imagex is the dir path + <videoname><count>.jpg
        video.write(imagex)
        os.remove(path+videoName+"%d.jpg" % count)
        count += 1
        

    cv2.destroyAllWindows()
    video.release()


# In[18]:


from tkinter import Tk    
from tkinter.filedialog import askopenfilename

def blurVideo():
    Tk().withdraw() 
    videoPathName = askopenfilename() #videoPathName is full path for video with file name and extension
    videoName = videoPathName.split("/")[-1][0:-4] #videoName is the video name without file extension
    x = len(videoName)+4
    path = videoPathName[0:len(videoPathName)- x] #path is the dir path that the video is in
    videoToFrames(videoPathName, videoName, path)
    
blurVideo()   



