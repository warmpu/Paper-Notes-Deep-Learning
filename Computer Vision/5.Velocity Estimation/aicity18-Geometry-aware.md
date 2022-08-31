# Geometry-aware Traffic Flow Analysis by Detection and Tracking
`UIUC, IBM, SIT` TEAM |  _16 December 2018 (this version, v2)_

> _3-th_ position among all participating teams  in Track 1 of NVIDIA AI City Challenge 2018

> We propose a simple yet effective method combing both learning based detection and geometric calibration based estimation. 
>We use a learning based method to detect and track vehicles, and use a geometry based camera calibration method to calculate the speed of those vehicles

* PAPER: [IEEE](https://ieeexplore.ieee.org/document/8575394)
* CODE: [Github]()


# Related Work
![fig34](../../asset/images/Speed%20Estimation/geometry-aware/fig3-4.jpg)
 
1. Object Detection Object Tracking (excluded for convenience)

2. Geometry Understanding

**the vanishing point of the first frame of a video**

- In case Fig4: the lanes of the road are parallel to each other in a large area of each frame
	- we first find the dash line which separates two lanes in a frame where straight segments of the dash line can be seen
	- use a math formula to express the extended lines of those straight dash line segments
	- We let the vanishing point be the point in the frame which has the lowest mean square distance to all those straight dash line segments


# Proposed Approach

1. Detection and Tracking
(skip for main concept)

2. Geometry-aware Traffic Flow Analysis

**Requirements**
- the camera plane is relatively perpendicular to the lanes on road =>
	- the distance between any point in the image and the vanishing point equals the distance between the vanishing point and the intersection of a perpendicular line going through the point of interest to the vertical line going through the vanishing point
	
	![fig45](../../asset/images/Speed%20Estimation/geometry-aware/fig5-6.jpg)

	- this relationship can be approximately expressed by:

	$$d_{x_1^{\"} x_2^{"}} = d_{x_1^{\'} x_2^{'}} \qquad \text{(Eq.1)}$$

-  the relationship of the distance between the vanishing point and a point on the vertical line across the vanishing point in real world. 
-  Fig.6:
   -  $x_{\inf}$ is infinitely far away from the image plane and on the ground
   -  Map this infinity point to the image plane -> $x_{\inf}^{\'}$
   -   For points on the ground _x1 , x2_ , the mapping from the truth ground to the image plane $x_1^{\'}, x_2^{\'}$

$$d_{x_{1}^{\prime} x_{2}^{\prime}}=d_{F x_{0}} \frac{d_{x_{1} x_{\infty}}}{d_{x_{0} x_{\infty}}} \frac{d_{x_{0} x_{i}}}{d_{x_{0} x_{1}}}-d_{F x_{0}} \frac{d_{x_{2} x_{\infty}}}{d_{x_{0} x_{\infty}}} \frac{d_{x_{0} x_{i}}}{d_{x_{0} x_{2}}}$$

   - Because $\frac{d_{x_{1} x_{\infty}}}{d_{x_{0} x_{\infty}}}=\frac{d_{x_{2} x_{\infty}}}{d_{x_{0} x_{\infty}}}=1$, we can get

     $$d_{x_{1}^{\prime} x_{2}^{\prime}}=\frac{C}{d_{x_{0} x_{1}}}-\frac{C}{d_{x_{0} x_{2}}}$$

     - where _C = dFx0.dx0xi_ is a constant
     
- Then, distance between x1,x2 in real is:

$$d_{x_{1} x_{2}}=\frac{C^{\prime}}{d_{x_{\infty}^{\prime} x_{2}^{\prime}}^{\prime}}-\frac{C^{\prime}}{d_{x_{\infty}^{\prime} x_{1}^{\prime}}^{\prime}}$$

- where **C'** is some constant, and **d'*** is a distance measured in the number of pixels.
  
- After finding the vanishing point of a frame, if we know $C^{\prime}$. then they measure the **distance of several line segments** that are parallel to the lane on road and are visible in
the video can get the real distance 

- can calculate the displacement distance of an object by

$$d=\left|\frac{C}{x_{1} \cos \theta_{1}}-\frac{C}{x_{2} \cos \theta_{2}}\right|$$

  - where **d** is the displacement distance of an object in real world
  - $x_{1} \text{ and } x_{2}$ are the distances between object positions in different frames with the vanishing point respectively
  - $\theta_{1} \text{ and } \theta_{2}$ are the corresponding angles between the vertical line across the vanishing point and the line segment from the vanishing point to the object position.
  - **C** is calculated by:
  
  $$C=\underset{C}{\operatorname{argmin}} \sum_{i=1}^{n}\left(\left|\frac{C}{x_{i 1} \cos \theta_{i 1}}-\frac{C}{x_{i 2} \cos \theta_{i 2}}\right|-d_{i}\right)^{2}$$

  -  $x_{i 1} \text{ and } x_{i 2}$ are marked line segment in the frame
  -  $d_{i}$ is the corresponding distance in real world measured from Google Maps.

- the speed of a vehicle by

$$s=d \frac{f_{r}}{f_{n}}$$

  - $f_{r}$ is the frame rate
  - $f_{n}$ is the difference of frame indexes between two frames




