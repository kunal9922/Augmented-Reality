# Augmented-Reality
Augmented Reality using [HomoGrapy](https://en.wikipedia.org/wiki/Homography_(computer_vision)) computer vision. in this we find out the target[base] image and put video or image or 3D object on that base. Is taken from mainVideo taken from camera 

## Process of code creation 

1. import dependencies 
   	* cv2
   	* numpy 
2. Create a class called Ar
   1. Takes input data
      * capturePath = where we want to take main frame from camera or specific path of mp4
      * putVideo = putting video on target image 
      * targetImagePath = target place where video will be put on.
      * frameHeight = optional frame height by default 480.
      * frameWidth = optional frame Width by default 480
3. Resizing putVideo because overlay putVideo on targetImage consist resizing according to TragetImage size.
4. main detector for finding features of target image in mainFrame video.
   * [ORB (Oriented FAST and Rotated BRIEF)](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html) which use for to find key features in target Image and also find in MainFrame[capturePath] and map them together.
5. create brute fore matcher algorithm for matching keyPoints using descriptor
6. Draw key points on target image as well mainFrame for better understanding.
7. if features are greater than 20 so we can call that we find out target images on Main Frame.
8. find Homography.
9. perspective Transform for overlay putVideo on target image
10. Then Draw polyLines on intesting region where target  image is find on mainFrame.
11. Now overlay image on target Image
    * we get a warp perspective Image 
    * then Masking of mainFrame image for putting proper video that help.
    * now we inverse image for get color region of mainFrame image.
    *  we overlay warpImage[putVideo] on mainFrame.
12.  we overlay warpImage[putVideo] on mainFrame.
13. DONE then show proper pipeline of program using a stack method which  was I created.



### Target Image

![](E:\Augmented-Reality\DataSet\targetImg.png)

### Final Pipe Line of Program

