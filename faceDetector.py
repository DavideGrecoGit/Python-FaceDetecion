import cv2; 

# Load some pre-trained data on face frontals from opencv
trainded_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

imgSingleFacePath = "testSingleFace.jpg" 
imgMultipleFacesPath = "testMultipleFaces.jpg" 


def coordinatesFromImg(img):
    # Convert to grayscale
    grayScaleImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Detect faces 
    face_coordinates = trainded_face_data.detectMultiScale(grayScaleImg)
 
    return face_coordinates

def faceFromImage(imgPath):
    # Load an image to detect faces in
    img = cv2.imread(imgPath)

    #Get faces coordinates
    face_coordinates = coordinatesFromImg(img)

    for (x,y,w,h) in face_coordinates:
        # Draw the rectangle
        cv2.rectangle(img, (x,y), (x+w,y+h), (0, 255, 0), 4)

    cv2.imshow("Face from Image detector", img)
    cv2.waitKey()

def faceFromWebcam():

    webcam = cv2.VideoCapture(0)

    key=-1

    while key==-1:
        # Read the current frame
        successful_frame_read, frame = webcam.read()

        #Get faces coordinates
        face_coordinates = coordinatesFromImg(frame)

        for (x,y,w,h) in face_coordinates:
            # Draw the rectangle
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 4)

        cv2.imshow("Face from Webcam detector", frame)
        key = cv2.waitKey(1)

    webcam.release()

faceFromImage(imgSingleFacePath)
faceFromImage(imgMultipleFacesPath)
faceFromWebcam()



