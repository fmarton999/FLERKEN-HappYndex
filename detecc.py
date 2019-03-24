from scipy.spatial import distance as dist
from imutils.video import VideoStream, FPS
from imutils import face_utils
import imutils
import numpy as np
import time
import dlib
import cv2
from time import gmtime
import sqlite3
import time
import sys

argument = sys.argv[1]

if(argument.isdigit()):
	SOURCE = int(argument)
else:
	SOURCE = argument
print SOURCE
#SOURCE = 0# "http://192.168.43.1:1024/video"

conn = sqlite3.connect('./databases/photos.db')
c = conn.cursor()


def smile(mouth):
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    avg = (A+B+C)/3
    D = dist.euclidean(mouth[0], mouth[6])
    mar=avg/D
    return mar
    
    
    
def createTable():
	c.execute("""CREATE TABLE IF NOT EXISTS photos (date INTEGER, happyndex REAL, imagename TEXT)""")
	return
	
def deleteTable():
	global c
	c.execute ("""DROP TABLE IF EXISTS photos""")
	
def insertPhoto(imagename, happyndex, time):
	global c
	c.execute ("INSERT INTO photos VALUES ('" + str(int(time)) + "','"  + str(happyndex) + "','" + imagename + "')")
	conn.commit()
	

createTable()
COUNTER = 0
TOTAL = 0


shape_predictor= "./shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)


(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print("[INFO] starting video stream thread...")
try:
	vs = VideoStream(src=SOURCE).start()
except:
	exit(1)
fileStream = False
time.sleep(1.0)

fps= FPS().start()
createTable()
lastImageTime=-222222222220
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	
	happinesCorend=0
	num =0
	for rect in rects:
		num+=1
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		mouth= shape[mStart:mEnd]
		mar= smile(mouth)
		mouthHull = cv2.convexHull(mouth)
		#print(shape)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
		#print mar
		happinesCorend+=mar
		happinesCore = happinesCorend / num
		if happinesCore <= .29 :
			COUNTER += 1
		print lastImageTime
		if happinesCore <= .29 and  COUNTER >= 25 and lastImageTime+30<int(time.time()):
				TOTAL += 1
				frame = vs.read()
				time.sleep(.3)
				frame2= frame.copy()
				img_name = ("/home/manc/Desktop/spreddit/images/"+ str(happinesCore) + ' ' +str(int(time.time()))+".png")
				cv2.imwrite(img_name, frame)
				print("{} written!".format(img_name))
				COUNTER = 0
				lastImageTime = time.time()
				print int(lastImageTime)
				insertPhoto(img_name,happYndex,lastImageTime)
	
		#cv2.putText(frame, "MAR: {}".format(mar), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	#kiszamoljuk a boldogsag indexet
	if(len(rects)==0):
		print "NO FACE DETECTED"
	else:
		happYndex = happinesCore *(.95**(len(rects)-1))
		print "HAPPYNDEX:"+str(happYndex)
	cv2.imshow("Frame" + str(SOURCE), frame)
	fps.update()
	
	key2 = cv2.waitKey(1) & 0xFF
	if key2 == ord('q'):
		break
	
fps.stop()


cv2.destroyAllWindows()








"""
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

cam = cv2.VideoCapture(0)

while True:
	prv, img = cam.read()
	
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	detect_faces = face_cascade.detectMultiScale(gray)
	
	for (fx, fy, fw, fh) in detect_faces:
		print("Face detected")
		
		cv2.rectangle(img, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 2)
		face_gray = gray[fy:fy + fh,fx:fx + fw]
		face_color = img[fy:fy + fh,fx:fx + fw]
		
		detect_eyes = eye_cascade.detectMultiScale(face_gray)
		for(ex, ey, ew, eh) in detect_eyes:
			cv2.rectangle(face_color, (ex,ey), (ex+ew,ey+eh), (255, 0, 0), 2)
			
		detect_smile = smile_cascade.detectMultiScale(face_gray)
		for(ex, ey, ew, eh) in detect_smile:
			cv2.rectangle(face_color, (ex,ey), (ex+ew,ey+eh), (0, 255, 0), 2)
		
		
		cv2.imshow("faces",img)
		
		
		wk = cv2.waitKey(30) & 0xff
		if(wk == 27):
			break
	
	

cam.release()
cv2.destroyAllWindows()
"""
