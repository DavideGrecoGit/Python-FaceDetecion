import cv2; 

# Load some pre-trained data on face frontals from opencv
trainded_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
trainded_smile_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')


imgSingleFacePath = "testSingleFace.jpg" 
imgMultipleFacesPath = "testMultipleFaces.jpg" 


def faceDetector(grayScaleImg, scaleFactor):
    return trainded_face_data.detectMultiScale(grayScaleImg, scaleFactor, minNeighbors=3)
    
def smileDetector(grayScaleImg, scaleFactor):
    return trainded_smile_data.detectMultiScale(grayScaleImg, scaleFactor, minNeighbors=25)

 
def faceAndSmileDetector(img, faceScaleFactor=1.1, smileScaleFactor=1.1):

    grayScaleFace = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Get faces coordinates
    face_coordinates = faceDetector(grayScaleFace,faceScaleFactor)

    for (x,y,w,h) in face_coordinates:
        # Draw the rectangle
        cv2.rectangle(img, (x,y), (x+w,y+h), (0, 255, 0), 4)

        face = img[y:y+h, x:x+w]

        grayScaleFace = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        smile_coordinates = smileDetector(grayScaleFace,faceScaleFactor)

        for (x_smile,y_smile,w_smile,h_smile) in smile_coordinates:
            # Draw the rectangle
            cv2.rectangle(face, (x_smile,y_smile), (x_smile+w_smile,y_smile+h_smile), (50, 50, 200), 4)

def detectFromImage(imgPath):
    # Load an image to detect faces in
    img = cv2.imread(imgPath)

    faceAndSmileDetector(img)

    cv2.imshow("Face from Image detector", img)
    cv2.waitKey()

def detectFromWebcam():

    webcam = cv2.VideoCapture(0)

    key=-1

    while key==-1:
        # Read the current frame
        successful_frame_read, frame = webcam.read()

        if(not successful_frame_read):
            key=-1;

        faceAndSmileDetector(frame, 1.2, 1.2)

        cv2.imshow("Face from Webcam detector", frame)
        key = cv2.waitKey(1)

    webcam.release()


detectFromImage(imgSingleFacePath)
detectFromImage(imgMultipleFacesPath)
detectFromWebcam()

