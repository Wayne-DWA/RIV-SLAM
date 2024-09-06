# RIV-SLAM: Radar-Inertial-Velocity optimization-based graph SLAM
***RIV-SLAM*** is an open source ROS package for real-time 6DOF SLAM using a 4D Radar and an IMU. It is based on 3D Graph SLAM with Adaptive Probability Distribution GICP scan matching-based odometry estimation and Intensity Scan Context loop detection. It also supports several graph constraints, such as GPS. We have tested this package with ***Oculli Eagle and Sensradar Hugin*** in outdoor structured (buildings, mine), unstructured (trees and grasses, forest) and semi-structured environments.

## 1. Dependency
### 1.1 **Ubuntu** and **ROS**
Ubuntu 64-bit 18.04 or 20.04.
ROS Melodic or Noetic. [ROS Installation](http://wiki.ros.org/ROS/Installation):

### 1.2 ***RIV-SLAM*** requires the following libraries:
- Eigen3
- OpenMP
- PCL
- g2o
### 1.3 The following ROS packages are required:
- geodesy
- nmea_msgs
- pcl_ros
- fast_gicp 
- msgs_radar for different datasets
```
    sudo apt-get install ros-XXX-geodesy ros-XXX-pcl-ros ros-XXX-nmea-msgs ros-XXX-libg2o
```
**NOTICE:** remember to replace "XXX" on above command as your ROS distributions, for example, if your use ROS-noetic, the command should be:
```
    sudo apt-get install ros-noetic-geodesy ros-noetic-pcl-ros ros-noetic-nmea-msgs ros-noetic-libg2o
```

## 2. System architecture
***RIV_SLAM*** consists of four nodelets.

- *preprocessing_nodelet*
- *scan_matching_odometry_nodelet*
- *floor_detection_nodelet*
- *radar_graph_slam_nodelet*

<div align="center">
    <img src="doc/overview.png" width = 70% >
</div>

## 3. optimization-based graph SLAM
Compared to the original software, we have introduced and modified the following modules:

- IMU Preintegration Factor
- Veloctiy Factor
- Ground Factor
- sliding window for optimization
<div align="center">
    <img src="doc/graph.png" width = 70% >
</div>

## 4. Run the package

```
#build the repo
catkin build msgs_radar fast_apdgicp radar_graph_slam
```
Download datasets: [NTU4DRadLM](https://github.com/junzhang2016/NTU4DRadLM) or [MineAndForest](https://github.com/kubelvla/mine-and-forest-radar-dataset) 
```
roslaunch radar_graph_slam radar_graph_slam.launch
```

## 5. Evaluate the results
In our paper, we use [rpg_trajectory_evaluation](https://github.com/uzh-rpg/rpg_trajectory_evaluation.git), the performance indices used are RE (relative error) and ATE (absolute trajectory error).

## 6. Acknowlegement
1. RIV-SLAM is heavily inspired by and based on 4DRadarSLAM [4DRadarSLAM](https://github.com/zhuge2333/4DRadarSLAM) and [koide3/hdl_graph_slam](https://github.com/koide3/hdl_graph_slam) 
2. [wh200720041/iscloam](https://github.com/wh200720041/iscloam) intensity scan context
3. [christopherdoer/reve](https://github.com/christopherdoer/reve) radar ego-velocity estimator
4. [NeBula-Autonomy/LAMP](https://github.com/NeBula-Autonomy/LAMP) odometry check for loop closure validation
5. [slam_in_autonomous_driving](https://github.com/gaoxiang12/slam_in_autonomous_driving) and [Dr. Gao Xiang (高翔)](https://github.com/gaoxiang12). His SLAM tutorial and blogs are the starting point of our SLAM journey.
