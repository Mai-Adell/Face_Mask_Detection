from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grap the dimensions of the frame and then construct a blob from it
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224,224),
                                (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()


    faces = []
    locs = []
    predictions = []

    # loop over the detections
    for i in range(0,detections.shape[2]):
        # extrcat the confidence (probability) associated with the detection
        confidence = detections[0,0,i,2]

        # filter out weak detections by removing detections with confidence less than the minimum confidence
        if confidence > 0.5:
            # compute x & y coordinates of the bounding box of the object
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype('int')

            # ensure the bounding box is falling within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel, ordering , resize it to 224*224 and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face = cv2.resize(face,(224,224))
            face = img_to_array(face)
            face = preprocess_input(face)

            #add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX,startY,endX,endY))

    # only make a prediction if at least one face is detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all* faces at the same time rather than one by one predictions
        # in the above for loop
        faces = np.array(faces, dtype="float32")
        predictions = maskNet.predict(faces, batch_size=32)  # maskNet is the model we have trained to detect masks


    # return a 2-tuples of the face locations and thier corresponding predictions
    return (locs, predictions)

# read the models that detect the face from disk
prototxtPath =  r'C:\Users\future\Downloads\Face-Mask-Detection-master\face_detector\deploy.prototxt'
weightsPath = r'C:\Users\future\Downloads\Face-Mask-Detection-master\face_detector\res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath,weightsPath)  # faceNet reads the 2 models used for face detection so now faceNet is the model responsible for detecting the face

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
vs = VideoStream(src=0).start() # intger given to src represents the number of the camera of your laptop you want to use (0 for first acmera , 1 for second camera ..etc)

print("statring video streaming.....")
#loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it to have the max width pf 400 pixel
    frame = vs.read()
    frame = imutils.resize(frame,width=400)

    # detect faces in the frame and determine if they are wearing masks or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding locations
    for (box, pred) in zip(locs, preds):
        #unback bounding box and predictions
        (startX,startY,endX,endY) = box
        (mask,withoutMask) = pred

        # determine the class label and color we'll use to draw the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0,225,0) if label == "Mask" else (0,0,225) # color here uses BGR technique (BGR: Blue, Green ,Red)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label,max(mask,withoutMask)*100)

        # display the label and bounding box rectangle on the output frame
        cv2.putText(frame,label,(startX,startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)  #FONT_HERSHEY_SIMPLEX is the name of the font used
        cv2.rectangle(frame,(startX,startY),(endX,endY),color, 2)

    # show the output frame
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
