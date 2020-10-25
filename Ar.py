import cv2
import numpy as np

class Ar:
	'''Augmented Reality using openCv called computer vision
	capturePath = 0 primaryCamera, capturePath=1 secondayCamera, capturePath="dir//fileName.mp4 '''

	def __init__(self, **input):
		self.input = input
		if "frameHeight" in input.keys() and "frameWidth" in input.keys():
			self.frameHeight = input["frameHeight"]
			self.frameWidth = input["frameWidth"]
		else:  # other wise  default
			self.frameHeight = 360
			self.frameWidth = 360

		self.cap = cv2.VideoCapture(input["capturePath"])
		self.myVid = cv2.VideoCapture(input["putVideo"])  # putting video on target image
		self.imgTarget = cv2.imread(input["targetImagePath"])  # target place where video will be put


if __name__=='__main__':
	AR1 = Ar(capturePath=r"DataSet//MainVideo.mp4",putVideo=r"DataSet//putVideo.mp4", targetImagePath=r"DataSet//mount.jpg")