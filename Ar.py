import cv2
import numpy as np

class Ar:

	'''Augmented Reality using openCv to overlay the video on the object'''

	def __init__(self, **capture):
		'''capturePath = 0 primaryCamera, capturePath=1 secondayCamera, capturePath="dir//fileName.mp4 '''
		self.input = capture
		if "frameHeight" in capture.keys() and "frameWidth" in capture.keys():
			self.frameHeight = capture["frameHeight"]
			self.frameWidth = capture["frameWidth"]
		else:  # other wise  default
			self.frameHeight = 780
			self.frameWidth = 780

		self.cap = cv2.VideoCapture(capture["capturePath"])
		self.myVid = cv2.VideoCapture(capture["putVideo"])  # putting video on target image
		self.imgTarget = cv2.imread(capture["targetImagePath"])  # target place where video will be put
		self.imgTarget = cv2.resize(self.imgTarget, (375, 381))
		self.detection = False

	## TO STACK ALL THE IMAGES IN ONE WINDOW
	def StackImages(self,imgArray, scale, lables=[]):
		"""
		The function `StackImages` takes in an array of images, scales them, and stacks them horizontally
		or vertically depending on the shape of the array, and returns the stacked image.
		
		:param imgArray: The `imgArray` parameter is a 2-dimensional list or array containing the images
		that you want to stack. Each element in the list represents a row of images, and each element
		within the row represents an individual image
		:param scale: The scale parameter is used to resize the images in the imgArray. It determines the
		scaling factor by which the images will be resized
		:param lables: The `lables` parameter is a list of labels for each image in the `imgArray`. These
		labels will be displayed on top of each image in the stacked output
		:return: the stacked image, which is stored in the variable "ver".
		"""
		rows = len(imgArray)
		cols = len(imgArray[0])
		rowsAvailable = isinstance(imgArray[0], list)
		width = imgArray[0][0].shape[1]
		height = imgArray[0][0].shape[0]

		if rowsAvailable:
			for x in range(0, rows):
				for y in range(0, cols):
					imgArray[x][y] = cv2.resize(imgArray[x][y], (width, height), None, scale, scale)
					if len(imgArray[x][y].shape) == 2:
						imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
			imageBlank = np.zeros((height, width, 3), np.uint8)
			hor = [imageBlank] * rows
			hor_con = [imageBlank] * rows
			for x in range(0, rows):
				hor[x] = np.hstack(imgArray[x])
				hor_con[x] = np.concatenate(imgArray[x])
			ver = np.vstack(hor)
			ver_con = np.concatenate(hor)
		else:
			for x in range(0, rows):
				imgArray[x] = cv2.resize(imgArray[x], (width, height), None, scale, scale)
				if len(imgArray[x].shape) == 2:
					imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
			print(imgArray.shape)
			hor = np.hstack(imgArray)
			hor_con = np.concatenate(imgArray)
			ver = hor
		if len(lables) != 0:
			eachImgWidth = int(ver.shape[1] / cols)
			eachImgHeight = int(ver.shape[0] / rows)
			print(eachImgHeight)
			for d in range(0, rows):
				for c in range(0, cols):
					cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
					              (c * eachImgWidth + len(lables[d]) * 13 + 27, 30 + eachImgHeight * d),
					              (255, 255, 255), cv2.FILLED)
					cv2.putText(ver, lables[d], (eachImgWidth * c + 10, eachImgHeight * d + 20),
					            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
		return ver

	def computeAr(self):
		'''to show and computation of Augmented Reality  '''

		# test our taking input is properly import in program
		global img2WithPoly
		_, imgVideo = self.myVid.read()

		# frame counter for counting frame rates
		frameCount = 0

		# fetch the height and width of target image and putVideo so that overlay of putVideo on target properly
		'''Resize of video which we want to put on target image because 
			its consistent overlay on traget if both frames have same size'''
		hT, wT, cT = self.imgTarget.shape
		imgVideo = cv2.resize(imgVideo, (wT, hT))

		# main detector for finding features of target image in mainFrame video
		orb = cv2.ORB_create(nfeatures=2000) # image matching  detector
		kp1, des1 = orb.detectAndCompute(self.imgTarget, None)
		self.imgTarget = cv2.drawKeypoints(self.imgTarget, kp1, None)

		# Initialize imgStack outside the while loop
		imgStack = None
		imgWrap = None  # Initialize imgWrap outside the while loop
		# for showing video use loop to capture frames
		while self.cap.isOpened():
			sucesss, imgCap = self.cap.read()
			
			# declaration of final outPut image which will be augmented
			imgCap = cv2.resize(imgCap, (wT+200, hT+200))
			imgAugment = imgCap.copy()
			imgWrap = imgCap.copy()
   
			if self.detection == False:
				# we not detect anything
				#so initialize video to 0
				self.myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
				frameCount = 0

			else:
				# if myVideo which will play on base that will end so that will replay
				if frameCount == self.myVid.get(cv2.CAP_PROP_FRAME_COUNT):  # check the myVideo total frame is eql to frame count so that loop it again
					# so initialize video to 0
					self.myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
					frameCount = 0 # say that this will again to zero or restart

				sucesss, imgVideo = self.myVid.read()
				imgVideo = cv2.resize(imgVideo, (wT, hT))
				# imgTarget = cv2.resize(self.imgTarget, (wT, hT))

			# testing this algorithm on our MainVideo frame for working properly or not
			kp2, des2 = orb.detectAndCompute(imgCap, None)
			imgCap = cv2.drawKeypoints(imgCap, kp2, None)
			good = []
   # Check if descriptors are valid before matching
			if des1 is not None and des2 is not None and des1.dtype == des2.dtype and des1.shape[1] == des2.shape[1]:
				# create brute force matcher algorithm for matching keyPoints using descriptor
				bf = cv2.BFMatcher()
				matches = bf.knnMatch(des1, des2, k=2)
				if matches:
					for m, n in matches:
						if m.distance < 0.75 * n.distance:
							good.append(m)
					print("key points are find out ", len(good))  # print out how many keyPoints are matches
				# rest of the code remains unchanged
				# ...
			else:
				print("Error: Invalid descriptors")
				
			# draw good matches on images
			imgFeatures = cv2.drawMatches(self.imgTarget, kp1, imgCap, kp2, good, None, flags=2)

			# Check if the indices are within the valid range
			for m in good:
				if m.trainIdx >= len(kp2) or m.queryIdx >= len(kp1):
					print("Invalid key point indices detected")
					break

			imgFeatures = cv2.resize(imgFeatures, (wT+200, hT+200))

			#  if features are greater than 20 so we can call that we find out target images on Main Frame
			if len(good) > 20:
				self.detection = True  # say to target image is find out
				print("This is our good feature matches = ", len(good))
				# m.query is the target image
				srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # looping and find out the good matches
				dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)  # looping and find out the good matches

				matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)

				# FIND OUT bounding box using perspective transform and Poly lines to draw lines
				pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
				# Check if the number of channels is correct (3 channels)
				#if pts.shape[1] == 2 and pts.shape[2] == 2:
				# Check if the number of channels is correct (3 channels)
				if pts.shape[2] == 2:
					dst = cv2.perspectiveTransform(pts, matrix)
		
					# draw poly lines on main frame image
					img2WithPoly = cv2.polylines(imgCap, [np.int32(dst)], True, (255,0,255), 2)

					# find out the warp perspective of an  image
					imgWrap = cv2.warpPerspective(imgVideo, matrix, (imgCap.shape[1], imgCap.shape[0]))

					# Masking of mainFrame image for putting proper video that help
					maskNew = np.zeros([imgCap.shape[0], imgCap.shape[1]], dtype=np.uint8)
					# now we have to color area where we find image as white actually this will mask
					cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))

					# now we inverse image for get color region of mainFrame image
					maskInv = cv2.bitwise_not(maskNew)
					imgAugment = cv2.bitwise_and(src1=imgAugment, src2=imgAugment, mask=maskInv)

					# we overlay warpImage[putVideo] on mainFrame
					imgAugment = cv2.bitwise_or(imgWrap, imgAugment)

					# using my manually created image stacking function for showing images in a proper format
					imgList = ([imgVideo, self.imgTarget, imgCap], [imgFeatures,imgWrap, imgAugment])
					imgStack = self.StackImages(imgList, scale=0.5)
			# show our taking images of web cam input as image frames

			# Update imgStack here, whether or not the condition is met
			imgList = ([imgVideo, self.imgTarget, imgCap], [imgFeatures, imgWrap, imgAugment])
			imgStack = self.StackImages(imgList, scale=0.5)
			
   			#	Output 
			cv2.imshow("Augmentation", imgStack)
			
			frameCount += 1

			if cv2.waitKey(1) & 0xFF == ord("q"):  # for quit the session Press 'q'
				break


if __name__ == '__main__':
	AR1 = Ar(capturePath=1, putVideo=r"DataSet\OverlayVideo.mp4", targetImagePath=r"DataSet//tragetImage.jpg")
	AR1.computeAr()