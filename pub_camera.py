import time
import numpy as np
import cv2
import json

from datetime import datetime
from src.MQTT.DoSomething import DoSomething
from picamera.array import PiRGBArray
from picamera import PiCamera
from pytz import timezone


def start_recoring(publisher):

	with PiCamera() as camera:

		camera.resolution = (640, 480)
		camera.framerate = 32
		camera.rotation = 180

		rawCapture = PiRGBArray(camera, size=(640, 480))

		time.sleep(0.1)
		hog = cv2.HOGDescriptor()
		hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

		for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

			image = frame.array

			

			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			boxes, weights = hog.detectMultiScale(gray, winStride=(10,10))

			boxes = _non_max_suppression_fast(boxes, 1.0)
			boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

			if len(boxes) == 0:
				# write/overwrite for the potential video
				cv2.imwrite('assets/storage/last_image.png', image)

			for ((xA, yA, xB, yB),weight) in zip(boxes,weights):

				if weight == np.max(weights):
					# display the detected boxes in the colour picture
					cv2.rectangle(image, (xA, yA), (xB, yB),(0, 255, 0), 2)
					amsterdam = timezone('Europe/Amsterdam')
					timestamp = datetime.now(amsterdam).strftime("%m-%d-%Y_%H:%M:%S")
					img_path = f'assets/storage/photo/{timestamp}.png'

					cv2.imwrite(img_path, image)
					cv2.imwrite('assets/storage/last_image.png', image)

					body = {
							'timestamp': timestamp,
							'class': 'Human', 
							'path': img_path 
					}

					publisher.myMqttClient.myPublish("/devices/C0001", json.dumps(body))

					rawCapture.truncate(0) 
					rawCapture.seek(0)

					return 

			rawCapture.truncate(0) 
			rawCapture.seek(0)


# source : https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def _non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")



if __name__ == "__main__":
	
	publisher = DoSomething("Publisher - Human Detection")
	publisher.run()
	
	while True:
		start_recoring(publisher)
		time.sleep(5)

