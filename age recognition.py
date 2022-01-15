# import face_recognition
# import imutils
import pickle
# import time
import cv2
import os
 
#find path of xml file containing haarcascade file 
# cascPathface = os.path.dirname("/usr/local/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml")
cascPathface = os.path.dirname(
 cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
 
 # load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)
# load the known faces and embeddings saved in last file
data = pickle.loads(open('face_enc', "rb").read())
 
print("Streaming started")
video_capture = cv2.VideoCapture(0)
# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
 
    # convert the input frame from BGR to RGB 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # the facial embeddings for face in input
    encodings = face_recognition.face_encodings(rgb)
    names = []
    # loop over the facial embeddings incase
    # we have multiple embeddings for multiple fcaes
    for encoding in encodings:
       #Compare encodings with encodings in data["encodings"]
       #Matches contain array with boolean values and True for the embeddings it matches closely
       #and False for rest
        matches = face_recognition.compare_faces(data["encodings"],
         encoding)
        #set name =inknown if no encoding matches
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            #Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                #Check the names at respective indexes we stored in matchedIdxs
                name = data["names"][i]
                #increase count for the name we got
                counts[name] = counts.get(name, 0) + 1
            #set name which has highest count
            name = max(counts, key=counts.get)
 
 
        # update the list of names
        names.append(name)
        # loop over the recognized faces
        for ((x, y, w, h), name) in zip(faces, names):
            # rescale the face coordinates
            # draw the predicted face name on the image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()




















# import cv2



# def faceBox(faceNet,frame):
#     frameHeight=frame.shape[0]
#     frameWidth=frame.shape[1]
#     blob=cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)
#     faceNet.setInput(blob)
#     detection=faceNet.forward()
#     bboxs=[]
#     for i in range(detection.shape[2]):
#         confidence=detection[0,0,i,2]
#         if confidence>0.7:
#             x1=int(detection[0,0,i,3]*frameWidth)
#             y1=int(detection[0,0,i,4]*frameHeight)
#             x2=int(detection[0,0,i,5]*frameWidth)
#             y2=int(detection[0,0,i,6]*frameHeight)
#             bboxs.append([x1,y1,x2,y2])
#             cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 1)
#     return frame, bboxs


# faceProto = "opencv_face_detector.pbtxt"
# faceModel = "opencv_face_detector_uint8.pb"

# ageProto = "age_deploy.prototxt"
# ageModel = "age_net.caffemodel"

# genderProto = "gender_deploy.prototxt"
# genderModel = "gender_net.caffemodel"



# faceNet=cv2.dnn.readNet(faceModel, faceProto)
# ageNet=cv2.dnn.readNet(ageModel,ageProto)
# genderNet=cv2.dnn.readNet(genderModel,genderProto)

# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# genderList = ['Male', 'Female']


# video=cv2.VideoCapture('4.mp4')

# padding=20

# while True:
#     ret,frame=video.read()
#     frame,bboxs=faceBox(faceNet,frame)
#     for bbox in bboxs:
#         # face=frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
#         face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
#         blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
#         genderNet.setInput(blob)
#         genderPred=genderNet.forward()
#         gender=genderList[genderPred[0].argmax()]


#         ageNet.setInput(blob)
#         agePred=ageNet.forward()
#         age=ageList[agePred[0].argmax()]


#         label="{},{}".format(gender,age)
#         cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0),-1) 
#         cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)
#     cv2.imshow("Age-Gender",frame)
#     k=cv2.waitKey(1)
#     if k==ord('q'):
#         break
# video.release()
# cv2.destroyAllWindows()














































# # import cv2

# # # Load the cascade
# # face_cascade = cv2.CascadeClassifier(r'/usr/local/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml')

# # # To capture video from webcam.
# # cap = cv2.VideoCapture(0)
# # while True:
# #     # Read the frame
# #     _, img = cap.read()

# #     # Convert to grayscale
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# #     # Detect the faces
# #     faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# #     # Draw the rectangle around each face
# #     for (x, y, w, h) in faces:
# #         cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 4), 10)

# #     # Display
# #     cv2.imshow('img', img)

# #     # Stop if escape key is pressed
# #     k = cv2.waitKey(30) & 0xff
# #     if k == 27:
# #         break

# # # Release the VideoCapture object
# # cap.release()









































# # # import cv2
# # # import sys
# # # import numpy as np
# # # import face_recognition

# # # cascPath = sys.argv[1]
# # # faceCascade = cv2.CascadeClassifier(cascPath)

# # # video_capture = cv2.VideoCapture(0)

# # # while True:
# # #     # Capture frame-by-frame
# # #     ret, frame = video_capture.read()

# # #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# # #     faces = faceCascade.detectMultiScale(
# # #         gray,
# # #         scaleFactor=1.1,
# # #         minNeighbors=5,
# # #         minSize=(30, 30),
# # #         flags=cv2.cv.CV_HAAR_SCALE_IMAGE
# # #     )

# # #     # Draw a rectangle around the faces
# # #     for (x, y, w, h) in faces:
# # #         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

# # #     # Display the resulting frame
# # #     cv2.imshow('Video', frame)

# # #     if cv2.waitKey(1) & 0xFF == ord('q'):
# # #         break

# # # # When everything is done, release the capture
# # # video_capture.release()
# # # cv2.destroyAllWindows()




















































# # # # import numpy as np
# # # # import cv2 as cv

# # # # cap = cv.VideoCapture(0)
# # # # if not cap.isOpened():
# # # #     print("Cannot open camera")
# # # #     exit()
# # # # while True:
# # # #     # Capture frame-by-frame
# # # #     ret, frame = cap.read()
# # # #     # if frame is read correctly ret is True
# # # #     if not ret:
# # # #         print("Can't receive frame (stream end?). Exiting ...")
# # # #         break
# # # #     # Our operations on the frame come here
# # # #     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
# # # #     # Display the resulting frame
# # # #     cv.imshow('frame', gray)
# # # #     if cv.waitKey(1) == ord('q'):
# # # #         break
# # # # # When everything done, release the capture
# # # # cap.release()
# # # # cv.destroyAllWindows()





















































# # # # # plot photo with detected faces using opencv cascade classifier
# # # # from cv2 import imread
# # # # from cv2 import imshow
# # # # from cv2 import waitKey
# # # # from cv2 import destroyAllWindows
# # # # from cv2 import CascadeClassifier
# # # # from cv2 import rectangle
# # # # import cv2
# # # # # load the photograph
# # # # from PIL import Image

# # # # video_capture = cv2.VideoCapture(1)
  
# # # # # open method used to open different extension image file
# # # # # pixels = Image.open(r'/Users/paarth/OneDrive - qerdp.co.uk/Coding/test1.jpg') 
  
# # # # # This method will show image in any image viewer 

# # # # # pixels = cv2.imread(r'/Users/paarth/OneDrive - qerdp.co.uk/Coding/test2.jpg')
# # # # # gray = cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY)


# # # # # # load the pre-trained model
# # # # # classifier = CascadeClassifier(r'/usr/local/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml')
# # # # # # (r'/Users/paarth/OneDrive - qerdp.co.uk/Coding/haarcascade_frontalface_default.xml')
# # # # # # perform face detection
# # # # # bboxes = classifier.detectMultiScale(pixels, 1.05, 8)
# # # # # # print bounding box for each detected face
# # # # # for box in bboxes:
# # # # #     # print(box)
# # # # # 	# extract
# # # # # 	x, y, width, height = box
# # # # # 	x2, y2 = x + width, y + height
# # # # # 	# draw a rectangle over the pixels
# # # # # 	rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)
# # # # # # show the image
# # # # # imshow('face detection', pixels)
# # # # # # keep the window open until we press a key
# # # # # waitKey(0)
# # # # # # close the window
# # # # # destroyAllWindows()