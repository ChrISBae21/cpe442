# Compilation of Projects and assignments for CPE442 Real-Time Embedded Systems<br /> #

  ### Lab 2: Image Viewer
  
  > Installed and used the OpenCV Library to create an image viewer. Written in C/C++, takes a filename as a command line argument and displays the image to the user.

  ### Lab 3: Sobel Filter
  
  > This lab uses Lab 2 to take a video frame-by-frame and apply a gray-scale, then a sobel-filter to the image, and displays the video back to the user. The program is deployed on a Raspberry PI and takes a video filename as a command line argument and displays the video with a sobel filter applied over it. 

  ### Lab 4: Sobel pthread optimization

  > This program optimizes the "barebone" sobel program in Lab 3 by introducing psuedo-parallelism in pthreads. The program grabs a frame, splits it into four "subframes" and has a different thread process apply the gray-scale and sobel on each quarter of the image. Barriers and introduced before moving on to each step (gray --> sobel --> display --> newframe --> gray...) to ensure no race conditions occur at the critical section. Deployed on the Raspberry Pi, noticeable improvements are seen.


