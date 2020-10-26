import cv2
import numpy as np

class Ar:

	'''Augmented Reality using openCv called computer vision
	capturePath = 0 primaryCamera, capturePath=1 secondayCamera, capturePath="dir//fileName.mp4 '''

	def __init__(self, **capture):
		self.input = capture
		if "frameHeight" in capture.keys() and "frameWidth" in capture.keys():
			self.frameHeight = capture["frameHeight"]
			self.frameWidth = capture["frameWidth"]
		else:  # other wise  default
			self.frameHeight = 360
			self.frameWidth = 360

		self.cap = cv2.VideoCapture(capture["capturePath"])
		self.myVid = cv2.VideoCapture(capture["putVideo"])  # putting video on target image
		self.imgTarget = cv2.imread(capture["targetImagePath"])  # target place where video will be put

	def computeAr(self):
		'''to show and computation of Augmented Reality  '''

		# test our taking input is properly import in program
		sucesss, imgVideo = self.myVid.read()

		# fetch the height and width of target image and putVideo so that overlay of putVideo on target properly
		'''Resize of video which we want to put on target image because 
			its consistent overlay on traget if both frames have same size'''
		hT, wT, cT = self.imgTarget.shape
		imgVideo = cv2.resize(imgVideo, (wT,hT))

		# main detector for finding features of target image in mainFrame video
		orb = cv2.ORB_create(nfeatures=1000) # image matching  detector
		kp1, des1 = orb.detectAndCompute(self.imgTarget, None)
		self.imgTarget = cv2.drawKeypoints(self.imgTarget, kp1, None)

		# for showing video use loop to capture frames
		while True:
			sucesss, imgCap = self.cap.read()
			imgAugment = imgCap.copy()
			# declaration of final outPut image which will be augumented
			imgCap = cv2.resize(imgCap, (wT+200, hT+200))
			imgAugment = imgCap.copy()
			# testing this algorithm on our MainVideo frame for working properly or not
			kp2, des2 = orb.detectAndCompute(imgCap, None)
			#imgCap = cv2.drawKeypoints(imgCap, kp2, None)

			# create brute fore matcher algorithm for matching keyPoints using descriptor
			bf = cv2.BFMatcher()
			matches = bf.knnMatch(des1, des2, k=2)
			good = []
			for m, n in matches:

				if m.distance < 0.75 * n.distance:
					good.append(m)
			print(len(good))  # print out how many keyPoints are matches

			# draw good matches on images
			imgFeatures = cv2.drawMatches(self.imgTarget, kp1, imgCap, kp2, good,None, flags=2 )
			cv2.imshow("ImgFeatures", imgFeatures)

			#  if features are greater than 20 so we can call that we find out target images on Main Frame
			if len(good) > 20:
				print("This is our good feature matches", good)
				# m.query is the target image
				srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # looping and find out the good matches
				dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)  # looping and find out the good matches

				matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
				print(matrix)

				# FIND OUT bounding box using perspective transform and Poly lines to draw lines
				pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
				dst = cv2.perspectiveTransform(pts, matrix)

				# draw poly lines on main frame image
				img2WithPoly = cv2.polylines(imgCap, [np.int32(dst)], True, (255,0,255), 2)

				# find out the warp perspective of an  image
				imgWrap = cv2.warpPerspective(imgVideo, matrix, (imgCap.shape[1], imgCap.shape[0]))

				#Masking of mainFrame image for putting proper video that help
				maskNew = np.zeros([imgCap.shape[0], imgCap.shape[1]], dtype=np.uint8)
				# now we have to color area where we find image as white actually this will mask
				cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))

				# now we inverse image for get color region of mainFrame image
				maskInv = cv2.bitwise_not(maskNew)
				imgAugment = cv2.bitwise_and(src1=imgAugment,src2= imgAugment, mask=maskInv)

			# show our taking imges of video
			cv2.imshow("Main Frame", imgCap)
			cv2.imshow("ImgTarget", self.imgTarget)
			cv2.imshow("ImgVideo", imgVideo)
			cv2.imshow("Poly Line image", img2WithPoly)
			cv2.imshow("Image Wraper", imgWrap)
			cv2.imshow("Image Masking", maskNew)
			cv2.imshow("Image Masking inverse", maskInv)
			cv2.imshow("image Augument", imgAugment)
			cv2.waitKey(0)


if __name__ == '__main__':
	AR1 = Ar(capturePath=r"DataSet//MainVideo.mp4",putVideo=r"DataSet//putVideo1.mp4", targetImagePath=r"DataSet//targetImg.png")
	AR1.computeAr()