# Augmented-Reality
Augmented Reality according to WIKIPEDIA = Augmented reality is an interactive experience of a real-world environment where the objects that reside in the real world are enhanced by computer-generated perceptual information, sometimes across multiple sensory modalities, including visual, auditory, haptic, somatosensory and olfactory

in this project Augmented reality can be done by OpenCV_python 

 using [HomoGrapy](https://en.wikipedia.org/wiki/Homography_(computer_vision)) computer vision. in this we firstly take the base image where the object will augmented then we will find(detect) the base image through web  camera if we will able to detect base image then simple we augments the Object3D, image, Video, 3dGraph anything.

## Dependencies  needs to install 

pip install opencv-python

pip install numpy



## Process of code creation 

1. import dependencies 
   	* cv2
   	* numpy 
2. Created a class called Ar
   1. Takes input data
      * capturePath = where we want to take main frame from camera or specific path of mp4
      * putVideo = putting video on target image 
      * targetImagePath = target place where video will be put on.
      * frameHeight = optional frame height by default 480.
      * frameWidth = optional frame Width by default 480
3. Resizing putVideo because overlay putVideo on targetImage consist resizing according to TragetImage size.
4. main detector for finding features of target image in mainFrame video.
   * [ORB (Oriented FAST and Rotated BRIEF)](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html) which use for to find key features in target Image and also find in MainFrame[capturePath] and map them together.
5. create brute force matcher algorithm for matching key-Points using descriptor
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



### Base Image

![ <img src="DataSet\targetImg.png" alt="forest" style="width : 500px;" />)

### Final Pipe Line of Program

<img src="DataSet\FinalPipeLine2.gif" />

