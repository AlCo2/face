import pathlib
import cv2


cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "C:\\Users\\Coder\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml"

clf = cv2.CascadeClassifier(str(cascade_path))
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(gray, 1.3, 5)
    for(x,y,width,height) in faces:
        cv2.rectangle(frame, (x,y), (x+width, y+height), (255,255,0), 1)
        roi_gray = gray[y:y+width, x:x+width]
        roi_color = frame[y:y+height, x:x+width]
    
    cv2.imshow("faces", frame)
    if(cv2.waitKey(1)==ord("q")):
        break

camera.release()
cv2.destroyAllWindows()
