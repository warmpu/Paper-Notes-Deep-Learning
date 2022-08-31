# SIMPLE ONLINE AND REALTIME TRACKING
Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos, Ben Upcroft. _7 Jul 2017 (this ver)_

>This paper explores a pragmatic approach to multiple object tracking where the main focus is to associate objects efficiently for online and realtime applications. 

* Official paper: [ArXiv](https://arxiv.org/abs/1602.00763)
* Official code: [Github](https://github.com/abewley/sort)

# Overview
- this work is primarily targeted towards online tracking where only detections from the previous and the current frame are presented to the tracker
- a tracking-bydetection framework for the problem of multiple object tracking (MOT) where objects are detected each frame and represented as bounding boxes.   
  - The MOT problem can be viewed as a data association problem where the aim is to associate detections across frames in a video sequence
  - the trade-off between accuracy and speed appears quite pronounced, since the speed of most accurate trackers is considered too slow for realtime applications

# METHODOLOGY

> The proposed method is described by the key components of detection, propagating object states into future frames, associating current detections with existing objects, and managing the lifespan of tracked objects.

1. Detection

- they utilise the Faster Region CNN (FrRCNN) detection framework
- the detection quality has a significant impact on tracking performance

2. Estimation Model

>  the object model, i.e. the representation and the motion model used to propagate a target’s identity into the next frame

- the inter-frame displacements of each object with a linear constant velocity model which is independent of other objects and camera motion
- The state of each target is modelled as:

$$\boldsymbol{x}=\left[x_{c}, y_{c}, s, a, \dot{x}_{c}, \dot{y}_{c}, \dot{s}\right]^{\top}$$

  - **u** and **v** represent the horizontal and vertical pixel location of the centre of the target
  - the scale **s** and **r** denote the scale (area) and the aspect ratio of the target’s bounding box respectively

-  a detection is associated to a target, the detected bounding box is used to update the target state where the velocity components are solved optimally via a Kalman filter framework

3. Data Association

- each target’s bounding box geometry is estimated by predicting its new location in the current frame
- The assignment cost matrix is then computed as the intersection-over-union (IOU) distance between each detection and all predicted bounding boxes from the existing targets.
- The assignment is solved optimally using the Hungarian algorithm

4. Creation and Deletion of Track Identities

- objects enter and leave the image, unique identities need to be created or destroyed accordingly
  -  any detection with an overlap less than IOUmin to signify the existence of an untracked object.
  -  Tracks are terminated if they are not detected for $T_{Lost}$ frames
-  The tracker is initialised using the geometry of the bounding box with the velocity set to zero
   -  the velocity is unobserved at this point 
   -  the covariance of the velocity component is initialised with large values, reflecting this uncertainty

# extraRef

- [Viblo.asian](https://viblo.asia/p/sort-deep-sort-mot-goc-nhin-ve-object-tracking-phan-1-Az45bPooZxY)