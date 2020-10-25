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

		self.imgTarget = cv2.resize(self.imgTarget, (420,420))
		# test our taking input is properly import in program
		sucesss, imgVideo = self.myVid.read()

		# fetch the height and width of target image and putVideo so that overlay of putVideo on target properly
		'''Resize of video which we want to put on target image because 
			its consistent overlay on traget if both frames have same size'''
		hT, wT, cT = self.imgTarget.shape
		imgVideo = cv2.resize(imgVideo, (wT,hT))

		# for showing video use loop to capture frames
		while True:
			sucesss, imgCap = self.cap.read()
			imgCap = cv2.resize(imgCap, (wT, hT))
			# show our taking imges of video
			cv2.imshow("Main Frame", imgCap)
			cv2.imshow("ImgTarget", self.imgTarget)
			cv2.imshow("ImgVideo", imgVideo)
			cv2.waitKey(0)


if __name__=='__main__':
	AR1 = Ar(capturePath=r"DataSet//MainVideo.mp4",putVideo=r"DataSet//putVideo.mp4", targetImagePath=r"DataSet//mount.jpg")