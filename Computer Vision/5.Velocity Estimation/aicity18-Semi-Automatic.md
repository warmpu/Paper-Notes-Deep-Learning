# A Semi-Automatic 2D solution for Vehicle Speed Estimation from Monocular Videos
 _16 December 2018 (this version, v2)_

> _7-th_ position among all participating teams  in Track 1 of NVIDIA AI City Challenge 2018

> In this work, we present a novel approach for vehicle speed estimation from monocular videos. The pipeline consists of modules for multi-object detection, robust tracking, and speed estimation. The tracking algorithm has the capability for jointly tracking individual vehicles and estimating velocities in the image domain. However, since camera parameters are often unavailable and extensive variations are present in the scenes, transforming measurements in the image domain to real world is challenging. We propose a simple two-stage algorithm to approximate the transformation. Images are first rectified to restore affine properties, then the scaling factor is compensated for each scene. We show the effectiveness of the proposed method with extensive experiments on the traffic speed analysis dataset in the NVIDIA AI City challenge.

* PAPER: [IEEE](https://ieeexplore.ieee.org/document/8575424)
* 
* CODE: [Github](https://github.com/NVIDIAAICITYCHALLENGE/2018AICITY_Maryland)

# Overview
The task is challenging in two aspects:
  -  First, **a robust vehicle detection and tracking algorithm**is required to localize individual vehicles over time under varying car orientations or lighting conditions. 
  -  Second, the transformation from image space to real world is often unavailable and requires expensive measuring equipments such as LIDARs.

# Proposed Approach

The proposed **vehicle speed estimation system**consists of three components: (1) _vehicle detection_ (2) _tracking_, and (3) _speed estimation_

![fig](../../asset/images/Speed%20Estimation/semi-automatic-2d-solution/fig1.jpg)

1. Vehicle Detection

-  apply Mask RCNN, an extension of the well-known Faster R-CNN  framework
-  apply a small fully convolutional network (FCN) to each RoI to predict a pixel-level segmentation mask
-  Mask-RCNN is able to detect the vehicles **at different scales**

![fig2](../../asset/images/Speed%20Estimation/semi-automatic-2d-solution/fig2.jpg)

2. Tracking

- 2 options:  Simple online and real time tracking ([SORT](../1.Tracking/simple-sort.md)) and DeepSORT
-  the main advantage of employing SORT and DEEPSORT for tracking, is that the**pixel velocity can be jointly estimated** with the tracking

3. Speed Estimation

After obtaining pixel velocities from the Kalman filter in the image domain, speed estimation approach has two steps: 

   * **Affine rectification** to restore affine-invariant properties of the scene
   * **Scale recovery** which estimates vehicle speed in the real world.

**Affine Rectification**
- Normally, each frame in the video is the projection of the 3D world on the image plane under a projective transformation, then ratios between segments are not preserved

> In this work, we assume most of the roads can be well-approximated by **a plane**

- the rectification technique which estimate a homography **H** that maps points $\mathbf{x}=[x, y, 1]^{T} --> \mathbf{X}=[X, Y, 1]^{T}$  :

$$
\mathbf{H} \mathbf{x}=\left[\begin{array}{llc}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & 1
\end{array}\right]\left[\begin{array}{l}
x \\
y \\
1
\end{array}\right]=\left[\begin{array}{c}
X \\
Y \\
1
\end{array}\right]=\mathbf{X} \qquad \text{ ( Eq.3 )}
$$

- Non-planar regions would result in degraded speed estimation. once $\mathbf{H}$ is determined, the velocity in the rectified domain can be obtained by differentiating both sides of (3):

$$
\left[\begin{array}{c}
\dot{X} \\
\dot{Y}
\end{array}\right]=\left[\begin{array}{ll}
\frac{h_{11}+C_{2,3} y}{\left(h_{31} x+h_{32} y+1\right)^{2}} & \frac{h_{12}-C_{2,3} x}{\left(h_{31} x+h_{32} y+1\right)^{2}} \\
\frac{h_{21}+C_{1,3} y}{\left(h_{31} x+h_{32} y+1\right)^{2}} & \frac{h_{22}-C_{1,3} x}{\left(h_{31} x+h_{32} y+1\right)^{2}}
\end{array}\right]\left[\begin{array}{l}
\dot{x} \\
\dot{y}
\end{array}\right]
$$

  - where $C_{i, j}$ represents the minor of $\mathbf{H}$ which corresponds to $h_{i, j}$.

- To **estimate H**, One classical approach  is _based on detecting vanishing points_ for two perpendicular directions
  - the two vanishing points as $\mathbf{v}_{1} \text{ and } \mathbf{v}_{2}$ in homogeneous coordinates:
    - $\mathbf{v}_{1}$ corresponds to the direction of the road axis
    - $\mathbf{v}_{2}$ corresponds to the direction perpendicular
    - we have two equation:

    $$\mathbf{H} \mathbf{v}_{1}=\left[\begin{array}{l}
      1 \\
      0 \\
      0
      \end{array}\right], \mathbf{H} \mathbf{v}_{2}=\left[\begin{array}{l}
      0 \\
      1 \\
      0
      \end{array}\right]$$

    - where $\mathbf{H}$ has the form:
  
    $$\left[\begin{array}{ccc}
      a & b & 0 \\
      c & d & 0 \\
      - & \mathbf{v}_{1} \times \mathbf{v}_{2} & -
      \end{array}\right]$$

  - For each location, it manually detected two vanishing points by selecting points in the scene


**Scale recovery**
- after image rectification, it is important to recover the scale in order to translate the speed from pixel space into real world domain
- Using the lane widths and road white strips length for the purpose of scale recovery in the horizontal and vertical directions respectively
- Using Google Maps, It is posible to obtained a rough estimate of the real world distances
  
- For the vertical direction, we propose another approach:
  -  After projection and rectification, the pixels along the vertical direction gets stretched and this effect is more prominent in the pixels near the detected vanishing point => **the scale along the y direction changes non-linearly**
  -  a linear compensator as follows:

  $$\begin{aligned}s(y) &=\frac{s_{2}-s_{1}}{y_{\max }-y_{\min }} y+s_{1}, \\
  s_{1} &=\frac{L_{1}}{l_{1}}, \\
  s_{2} &=\frac{L_{2}}{l_{2}},
  \end{aligned}$$

  - $L_{1} \text{ and } L_{2}$ are lengths of two white strips at two different heights in meters.
  - $l_{1} \text{ and } l_{2}$ are the corresponding lengths in pixels in the transformed image

- Finally, The speed estimate of the vehicle is then given by

$$
\sqrt{\left(s_{x} \dot{X}\right)^{2}+\left(s_{y} \dot{Y}\right)^{2}}
$$

We estimate the speed of all the vehicles in the window whose heights in the image space are from $y_{\min }$ to $y_{\max }$.