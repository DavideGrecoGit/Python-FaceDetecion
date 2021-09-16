import cv2; 

# Load some pre-trained data on face frontals from opencv
trainded_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

imgSingleFacePath = "Python/Python-FaceDetection/testSingleFace.jpg" 
imgMultipleFacesPath = "Python/Python-FaceDetection/testMultipleFaces.jpg" 


def faceFromImage(imgPath):
    # Load an image to detect faces in
    img = cv2.imread(imgPath)

    # Change to grayscale
    grayScaleImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Detect faces 
    face_coordinates = trainded_face_data.detectMultiScale(grayScaleImg)

    for (x,y,w,h) in face_coordinates:
        # Draw the rectangle
        cv2.rectangle(img, (x,y), (x+w,y+h), (0, 255, 0), 4)

    cv2.imshow("Face detection", img)
    cv2.waitKey()


faceFromImage(imgSingleFacePath)
faceFromImage(imgMultipleFacesPath)
