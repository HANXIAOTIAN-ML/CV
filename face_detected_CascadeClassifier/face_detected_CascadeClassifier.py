import cv2

faceCascade = cv2.CascadeClassifier("D:/Anaconda/envs/tensorbase/Lib/site-packages/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml")
#cap = cv2.VideoCapture(0)
#ret, image = cap.read()
image=cv2.imread('test.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(30,30),)
if faces is not None:
   for (x, y, width, height) in faces:
       cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

cv2.imwrite("test_detected.jpg",image)
cv2.imshow("Face", image)
cv2.waitKey(0)
