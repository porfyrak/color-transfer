# Color transfer using K-mean algorithm 

Is a method that tries to change the color of our target image based on source image. Best value of K in our algorithm is 1 - 3  based on our images.
The implementation is based on the following paper Color transfer based on color classification by Bin Xie Kang Zhang and Ke Liu https://iopscience.iop.org/article/10.1088/1742-6596/1601/3/032028.

## INPUT AND OUTPUT OF OUR IMPLEMENTATION

Input -> Source image, Target image and K value

Output -> Transferred image

## METHOD STEPS

1) Convert RGB images to L*A*B
2) Use K-Mean to segment the images in K regions both for Source and Target image
3) Implement Color Tranfer Algorithm
    For every region AND channel:
      a) Estimate Mean kai Standard Deviation
      b) Remove mean from target image 
      c) Multiply target image with Source_Std / Target_Std
      d) Add to target image the mean value of source image
4) Merge channels
5) Convert image from L*A*B to RGB  
 
 ## Packages 
 
Numpy, OpenCV and Scipty

## Addition 

Color Histogram of Source Image and Target Image. Based on the color distribution we are able to estimate the K value easier. For example, if the different channels has a large number of peaks it is better to use k = 1. 
