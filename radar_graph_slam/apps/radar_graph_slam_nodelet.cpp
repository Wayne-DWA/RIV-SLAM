// SPDX-License-Identifier: BSD-2-Clause

// =================================================================
// Note: Part of the work is not open source due to previous patent


#include <ctime>
#include <mutex>
#include <atomic>
#include <memory>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/registration/icp.h>
#include <pcl/octree/octree_search.h>

#include <ros/ros.h>
#include <geodesy/utm.h>
#include <geodesy/wgs84.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include <std_msgs/Time.h>
#include <std_msgs/String.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>
#include <geographic_msgs/GeoPointStamped.h>
#include <visualization_msgs/MarkerArray.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <radar_graph_slam/SaveMap.h>
#include <radar_graph_slam/LoadGraph.h>
#include <radar_graph_slam/DumpGraph.h>
#include <radar_graph_slam/ros_utils.hpp>
#include <radar_graph_slam/ros_time_hash.hpp>
#include <radar_graph_slam/FloorCoeffs.h>
#include <radar_graph_slam/graph_slam.hpp>
#include <radar_graph_slam/keyframe.hpp>
#include <radar_graph_slam/keyframe_updater.hpp>
#include <radar_graph_slam/loop_detector.hpp>
#include <radar_graph_slam/information_matrix_calculator.hpp>
#include <radar_graph_slam/map_cloud_generator.hpp>
// #include "radar_graph_slam/polynomial_interpolation.hpp"
#include <radar_graph_slam/registrations.hpp>
#include <radar_graph_slam/imu_preintegration.hpp>
#include "radar_graph_slam/nav_state.hpp"

#include "scan_context/Scancontext.h"

#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/edge_se3_plane.hpp>
#include <g2o/edge_se3_priorxy.hpp>
#include <g2o/edge_se3_priorxyz.hpp>
#include <g2o/edge_se3_priorz.hpp>
#include <g2o/edge_se3_priorvec.hpp>
#include <g2o/edge_se3_priorquat.hpp>
#include <g2o/edge_se3_interial.hpp>

#include "utility_radar.h"
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <geometry_msgs/TwistStamped.h>




using namespace std;
using SO3 = Sophus::SO3d;
using SE3 = Sophus::SE3d;
using Vec9d = Eigen::Matrix<double, 9, 1>;
using Mat15d = Eigen::Matrix<double, 15, 15>;

namespace radar_graph_slam {

class RadarGraphSlamNodelet : public nodelet::Nodelet, public ParamServer {
public:
  typedef pcl::PointXYZI PointT;
  typedef PointXYZIRPYT  PointTypePose;
  typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, sensor_msgs::PointCloud2, radar_graph_slam::FloorCoeffs > ApproxSyncPolicy;

  RadarGraphSlamNodelet() {}
  virtual ~RadarGraphSlamNodelet() {}

  virtual void onInit() {
    nh = getNodeHandle();
    mt_nh = getMTNodeHandle();
    private_nh = getPrivateNodeHandle();

    // init parameters
    map_cloud_resolution = private_nh.param<double>("map_cloud_resolution", 0.05);
    trans_odom2map.setIdentity();
    trans_aftmapped.setIdentity();
    trans_aftmapped_incremental.setIdentity();
    last_imu_preinteg_transf.setIdentity();
    Eigen::Isometry3d another_pose = Eigen::Isometry3d::Identity();

    initial_pose.setIdentity();

    max_keyframes_per_update = private_nh.param<int>("max_keyframes_per_update", 10);
    anchor_node = nullptr;
    anchor_edge = nullptr;
    floor_plane_node = nullptr;
    graph_slam.reset(new GraphSLAM(private_nh.param<std::string>("g2o_solver_type", "lm_var")));
    loop_optimizer.reset(new GraphSLAM(private_nh.param<std::string>("g2o_solver_type", "lm_var")));
    keyframe_updater.reset(new KeyframeUpdater(private_nh));
    loop_detector.reset(new LoopDetector(private_nh));
    map_cloud_generator.reset(new MapCloudGenerator());
    inf_calclator.reset(new InformationMatrixCalculator(private_nh));

    floor_edge_stddev = private_nh.param<double>("floor_edge_stddev", 1.0e-6);
    inertial_weight = private_nh.param<double>("inertial_weight", 1.0);

    show_sphere = private_nh.param<bool>("show_sphere", false);

    registration = select_registration_method(private_nh);

    // subscribers
    odom_sub.reset(new message_filters::Subscriber<nav_msgs::Odometry>(mt_nh, odomTopic, 256));
    ground_sub.reset(new message_filters::Subscriber<radar_graph_slam::FloorCoeffs>(mt_nh, "/floor_detection/floor_coeffs", 256));

    if(private_nh.param<bool>("enable_under_floor_removal", false)) 
    {
      cloud_sub.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(mt_nh, "/underfloor_filtered_points", 32)); 
    }
    else cloud_sub.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(mt_nh, "/filtered_points", 32));

    sync.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(32), *odom_sub, *cloud_sub, *ground_sub));
    sync->registerCallback(boost::bind(&RadarGraphSlamNodelet::cloud_callback, this, _1, _2, _3));
    
    imu_sub = nh.subscribe(imuTopic, 1024, &RadarGraphSlamNodelet::imu_callback, this);
    // ego_twist_sub = nh.subscribe("/eagle_data/twist", 1024, &RadarGraphSlamNodelet::twist_callback, this);
    pub_twist_stamped = nh.advertise<geometry_msgs::TwistStamped>("/map_ego_twist_stamped", 5);

    //imu odom
    pubImuOdometry   = nh.advertise<nav_msgs::Odometry>("radar_graph_slam/imuPre/odometry", 2000);
    command_sub = nh.subscribe("/command", 10, &RadarGraphSlamNodelet::command_callback, this);

    //***** publishers ******
    markers_pub = mt_nh.advertise<visualization_msgs::MarkerArray>("/radar_graph_slam/markers", 16);
    // Transform RadarOdom_to_base
    odom2base_pub = mt_nh.advertise<geometry_msgs::TransformStamped>("/radar_graph_slam/odom2base", 16);
    aftmapped_odom_pub = mt_nh.advertise<nav_msgs::Odometry>("/radar_graph_slam/aftmapped_odom", 16);
    aftmapped_odom_incremenral_pub = mt_nh.advertise<nav_msgs::Odometry>("/radar_graph_slam/aftmapped_odom_incremental", 16);
    map_points_pub = mt_nh.advertise<sensor_msgs::PointCloud2>("/radar_graph_slam/map_points", 1, true);
    read_until_pub = mt_nh.advertise<std_msgs::Header>("/radar_graph_slam/read_until", 32);
    
    // odom_frame2frame_pub = mt_nh.advertise<nav_msgs::Odometry>("/radar_graph_slam/odom_frame2frame", 16);

    load_service_server = mt_nh.advertiseService("/radar_graph_slam/load", &RadarGraphSlamNodelet::load_service, this);
    dump_service_server = mt_nh.advertiseService("/radar_graph_slam/dump", &RadarGraphSlamNodelet::dump_service, this);
    save_map_service_server = mt_nh.advertiseService("/radar_graph_slam/save_map", &RadarGraphSlamNodelet::save_map_service, this);

    floor_sub = mt_nh.subscribe("/floor_detection/floor_coeffs", 1024, &RadarGraphSlamNodelet::floor_coeffs_callback, this);
    pubOptimizedPath = nh.advertise<nav_msgs::Path>("/optimized/path", 16);
    pubSMPath = nh.advertise<nav_msgs::Path>("/scanmatch/path", 16);

    double graph_update_interval = private_nh.param<double>("graph_update_interval", 3.0);
    double map_cloud_update_interval = private_nh.param<double>("map_cloud_update_interval", 10.0);
    // optimization_timer = mt_nh.createWallTimer(ros::WallDuration(graph_update_interval), &RadarGraphSlamNodelet::optimization_timer_callback, this);
    map_publish_timer = mt_nh.createWallTimer(ros::WallDuration(map_cloud_update_interval), &RadarGraphSlamNodelet::map_points_publish_timer_callback, this);
  
    b_a_in = Eigen::Vector3d(imuAccBiasN, imuAccBiasN, imuAccBiasN);
    b_g_in = Eigen::Vector3d(imuGyrBiasN, imuGyrBiasN, imuGyrBiasN);

    preinteg_options_.noise_acce_ = imuAccNoise;
    preinteg_options_.noise_gyro_ = imuGyrNoise;
    preinteg_options_.init_bg_ = b_a_in;
    preinteg_options_.init_ba_ = b_g_in;

    double bg_rw2 = 1.0 / (preinteg_options_.noise_gyro_ * preinteg_options_.noise_gyro_);
    preinteg_options_.bg_rw_info_.diagonal() << bg_rw2, bg_rw2, bg_rw2;
    double ba_rw2 = 1.0 / (preinteg_options_.noise_acce_ * preinteg_options_.noise_acce_);
    preinteg_options_.ba_rw_info_.diagonal() << ba_rw2, ba_rw2, ba_rw2;
    cur_preinteg_options_ = preinteg_options_;
    predict_preinteg_options_ = preinteg_options_;

    preinteg_opt = std::make_shared<IMUPreintegrator>(preinteg_options_);
    preinteg_predict = std::make_shared<IMUPreintegrator>(predict_preinteg_options_);
  }


private:
  /**
   * @brief received point clouds are pushed to #keyframe_queue
   * @param odom_msg
   * @param cloud_msg
   * @param ground_msg
   */
  void cloud_callback(const nav_msgs::OdometryConstPtr& odom_msg, const sensor_msgs::PointCloud2::ConstPtr& cloud_msg, const radar_graph_slam::FloorCoeffs::ConstPtr& ground_msg) 
  {
    Eigen::Vector4d ground_coeffs(ground_msg->coeffs[0], ground_msg->coeffs[1], ground_msg->coeffs[2], ground_msg->coeffs[3]);
    Eigen::Vector3d current_velocity (odom_msg->twist.twist.linear.x, odom_msg->twist.twist.linear.y, odom_msg->twist.twist.linear.z);
    // Push ego velocity to queue
    geometry_msgs::TwistWithCovarianceStamped::Ptr twist_(new geometry_msgs::TwistWithCovarianceStamped);
    twist_->header.stamp = cloud_msg->header.stamp;
    twist_->twist.twist = odom_msg->twist.twist;
    twist_->twist.covariance = odom_msg->twist.covariance;
    twist_window.push_back(twist_);
    
    const ros::Time& stamp = cloud_msg->header.stamp;
    Eigen::Isometry3d odom_now = odom2isometry(odom_msg);
    Eigen::Matrix4d matrix_map2base;
    // Publish TF between /map and /base_link
    if(keyframe_queue.size() > 2)
    {
      const KeyFrame::Ptr& keyframe_last = keyframe_queue.back();
      //the relative transformation from the last keyframe to the current position
      Eigen::Isometry3d lastkeyframe_odom_incre =  keyframe_last->odom_scan2scan.inverse() * odom_now;
      geometry_msgs::Pose relativePose = isometry2pose(lastkeyframe_odom_incre);
      Eigen::Isometry3d keyframe_map2base_matrix;
      //last optimized nav state
      keyframe_map2base_matrix.translation() = Eigen::Vector3d(navstates_window.back().p_(0), navstates_window.back().p_(1), navstates_window.back().p_(2));
      keyframe_map2base_matrix.linear() = (navstates_window.back().R_).matrix();
      // map2base = odom^(-1) * base
      matrix_map2base = (keyframe_map2base_matrix * lastkeyframe_odom_incre).matrix();
    }
    geometry_msgs::TransformStamped map2base_trans = matrix2transform(cloud_msg->header.stamp, matrix_map2base, mapFrame, baselinkFrame);
    current_vel_map = matrix_map2base.block<3, 3>(0, 0)* current_velocity;
    // Eigen::Vector3d ego_vel_map = map2base_trans.rotation() * Eigen::Vector3d(current_velocity.x, current_velocity.y, current_velocity.z);
    //quaternion normalization/fix
    if(pow(map2base_trans.transform.rotation.w,2) + pow(map2base_trans.transform.rotation.x,2) + 
        pow(map2base_trans.transform.rotation.y,2) + pow(map2base_trans.transform.rotation.z,2) < pow(0.9,2)) 
      {
        map2base_trans.transform.rotation.w=1; 
        map2base_trans.transform.rotation.x=0; 
        map2base_trans.transform.rotation.y=0; 
        map2base_trans.transform.rotation.z=0;
      }
    map2base_broadcaster.sendTransform(map2base_trans);

    //pub scanmatching path for visualization
    geometry_msgs::PoseStamped pose_stamped_sm;
    pose_stamped_sm.header.stamp = cloud_msg->header.stamp;
    pose_stamped_sm.header.frame_id = mapFrame;
    pose_stamped_sm.pose = odom_msg->pose.pose;
    SM_path.poses.push_back(pose_stamped_sm);
    SM_path.header.stamp = cloud_msg->header.stamp;
    SM_path.header.frame_id = mapFrame;
    pubSMPath.publish(SM_path);
   
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if(baselinkFrame.empty()) {
      baselinkFrame = cloud_msg->header.frame_id;
    }

    // Get time of this key frame for Intergeration, to integerate between two key frames
    auto this_keyframe_stamp = cloud_msg->header.stamp.toSec();
    bool keyframe_updated = keyframe_updater->decide(odom_now, stamp);
    
    double accum_d = keyframe_updater->get_accum_distance();
    Eigen::Isometry3d dummy_pose = Eigen::Isometry3d::Identity();
    // Construct keyframe
    KeyFrame::Ptr keyframe(new KeyFrame(keyframe_index, stamp, odom_now, accum_d, cloud, current_velocity,dummy_pose));
    keyframe->floor_coeffs = ground_coeffs;
    keyframe_index ++;
    keyframes.push_back(keyframe);
    std::lock_guard<std::mutex> lock(keyframe_queue_mutex);

    keyframe_queue.push_back(keyframe);
    keyframe_hash[keyframe->stamp] = keyframe;
    frames_window.push_back(keyframe);
    
    const auto& cur_keyframe = frames_window.back();
    // auto this_frame_stamp = this_keyframe->stamp.toSec();
    trans_odom2map_mutex.lock();
    Eigen::Isometry3d odom2map(trans_odom2map.cast<double>());
    trans_odom2map_mutex.unlock();
    Eigen::Isometry3d cur_odom = odom2map * cur_keyframe->odom_scan2scan;
    odom_window.push_back(cur_odom);
    
    if(keyframes.size() < 2)
    {
      ROS_INFO("keyframes size is less than 2, waiting for more keyframes!");
      return;
    }
    // ROS_INFO("keyframes size is %d", keyframes.size());
    // const KeyFrame::Ptr& current_keyframe = keyframes.back();
    const KeyFrame::Ptr& last_keyframe = keyframes[keyframes.size() - 2];
    auto last_keyframe_stamp = last_keyframe->stamp.toSec();

    // ********** Select number of keyframess to be optimized **********

    if(navstates_window.size() < 1)
    {
      NavStated first_nav_state;
      first_nav_state.v_.setZero();
      first_nav_state.p_.setZero();
      first_nav_state.bg_ = b_g_in;
      first_nav_state.ba_ = b_a_in;
      first_nav_state.timestamp_ = this_keyframe_stamp;
      navstates_window.push_back(first_nav_state);
      bg_window.push_back(b_g_in);
      ba_window.push_back(b_a_in);
    }
    imu_mutex.lock();
    if(imu_opt_queue.empty()) 
    {
      ROS_WARN("there is no imu waiting for optimization!");
      return;
    }
    if(this_keyframe_stamp - last_keyframe_stamp < 0)
    {
      ROS_ERROR("keyframe time stamp is not in order!");
      return;
    }
    if(this_keyframe_stamp - last_keyframe_stamp > 10.0)
    {
      ROS_WARN("large keyframe time interval = %f seconds!", this_keyframe_stamp - last_keyframe_stamp);
      return;
    }
    // std::deque<sensor_msgs::Imu> this_imu_queue;
    while (!imu_opt_queue.empty())
    {
        if (imu_opt_queue.front().header.stamp.toSec() < last_keyframe_stamp)
        {
          lastIMU_opt_stamp = imu_opt_queue.front().header.stamp.toSec();
          imu_opt_queue.pop_front();
          // ROS_INFO("imu_opt_queue size: %d", imu_opt_queue.size());

        }
        else
            break;
    }
    if(lastIMU_opt_stamp - last_keyframe_stamp > 0.1)
      ROS_ERROR("large IMU time interval to last frame  = %d", lastIMU_opt_stamp - last_keyframe_stamp);
    

    cur_preinteg_options_.init_bg_ = bg_window.back();
    cur_preinteg_options_.init_ba_ = ba_window.back();
    std::shared_ptr<IMUPreintegrator> this_preinteg = std::make_shared<IMUPreintegrator>(cur_preinteg_options_);
    while(!imu_opt_queue.empty())
    {
      sensor_msgs::Imu imu_msg = imu_opt_queue.front();
      if(imu_msg.header.stamp.toSec() < this_keyframe_stamp)
      {
          double dt = (lastIMU_opt_stamp < 0) ? (1.0 / 200.0) : (imu_msg.header.stamp.toSec() - lastIMU_opt_stamp);
          // ROS_INFO("imu dt: %f", dt);
          if(dt > 0.01)
          {
            ROS_ERROR("IMU dt is too large dt = %f", dt);
            dt = 0.005;
            ROS_WARN("set IMU dt = %f", dt);
          }   
          lastIMU_opt_stamp = imu_msg.header.stamp.toSec();
          this_preinteg->propagate(dt, 
                                  Eigen::Vector3d(imu_msg.linear_acceleration.x,imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z), 
                                  Eigen::Vector3d(imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z));
          imu_opt_queue.pop_front();
      }
      else
        break;
      
    }
    imu_mutex.unlock();
    
    preinteg_window.push_back(this_preinteg);
    NavStated last_nav_state = navstates_window.back();
    NavStated current_nav_state = preinteg_window.back()->predict(last_nav_state);
    navstates_window.push_back(current_nav_state);

    if(frames_window.size() > 6)
    {
      frames_window.pop_front();
      navstates_window.pop_front();
      preinteg_window.pop_front();
      twist_window.pop_front();
      odom_window.pop_front();
    }

    for(int i = 0; i < frames_window.size(); i++)
    {

      auto cur_nav_state = navstates_window[i];
      g2o::VertexVelocity* cur_vel_vertex = graph_slam -> add_velocity_node(cur_nav_state.v_);
      g2o::VertexGyroBias* cur_bg_vertex = graph_slam -> add_gyro_bais_node(cur_nav_state.bg_);
      g2o::VertexAccBias* cur_ba_vertex = graph_slam -> add_acc_bias_node(cur_nav_state.ba_);
      const auto& curr_keyframe = frames_window[i];
      curr_keyframe->node = graph_slam->add_se3_node(odom_window[i]);

      ba_ver_window.push_back(cur_ba_vertex);
      bg_ver_window.push_back(cur_bg_vertex);
      vel_ver_window.push_back(cur_vel_vertex);
    }

    //add edge between the last keyframe and the current keyframe
    for(int i = 0; i < frames_window.size(); i++)
    { 
      if(i == 0)
      {
        continue;
      }

      const auto& this_keyframe = frames_window[i];
      const auto& prev_keyframe = frames_window[i - 1];

      g2o::EdgeGyroRW* gyro_edge = graph_slam -> add_gyro_rw_edge(bg_ver_window[i-1], bg_ver_window[i], preinteg_options_.bg_rw_info_);
      g2o::EdgeAccRW* acc_edge = graph_slam -> add_acc_rw_edge(ba_ver_window[i-1], ba_ver_window[i], preinteg_options_.ba_rw_info_);  

      Eigen::Isometry3d relative_pose = this_keyframe->odom_scan2scan.inverse() * prev_keyframe->odom_scan2scan;
      Eigen::MatrixXd pose_information = inf_calclator->calc_information_matrix(this_keyframe->cloud, prev_keyframe->cloud, relative_pose);
      auto pose_edge = graph_slam->add_se3_edge(this_keyframe->node, prev_keyframe->node, relative_pose, pose_information);
      graph_slam->add_robust_kernel(pose_edge, private_nh.param<std::string>("odometry_edge_robust_kernel", "NONE"), private_nh.param<double>("odometry_edge_robust_kernel_size", 1.0));
      
      SE3 odomSE3(odom_window[i].rotation(), odom_window[i].translation());
      g2o::EdgePose* scanmatching_pose_edge = graph_slam->add_pose_edge(this_keyframe->node,odomSE3,pose_information);
      graph_slam->add_robust_kernel(scanmatching_pose_edge, "Huber", 0.5);

      g2o::EdgeSE3Interial* imu_interial_edge = graph_slam->add_se3_interial_edge(prev_keyframe->node, vel_ver_window[i-1], bg_ver_window[i-1], ba_ver_window[i-1], this_keyframe->node, vel_ver_window[i], preinteg_window[i-1],inertial_weight);
      graph_slam->add_robust_kernel(imu_interial_edge, "Huber", 0.5);
      Eigen::Vector3d current_ego_v = Eigen::Vector3d(twist_window[i]->twist.twist.linear.x, twist_window[i]->twist.twist.linear.y, twist_window[i]->twist.twist.linear.z);
      current_ego_v = navstates_window[i].R_ * current_ego_v;
      Eigen::Matrix3d current_vel_info_ = Eigen::Matrix3d::Zero();
      if(twist_->twist.covariance[0] != 0 && twist_->twist.covariance[7] != 0 && twist_->twist.covariance[14] != 0)
      {
        current_vel_info_(0,0) = 0.01/twist_window[i]->twist.covariance[0];
        current_vel_info_(1,1) = 0.01/twist_window[i]->twist.covariance[7];
        current_vel_info_(2,2) = 0.01/twist_window[i]->twist.covariance[14];
      }
      else
      {
        // ROS_ERROR("no covariance of twist message!");
        current_vel_info_(0,0) = 10;
        current_vel_info_(1,1) = 10;
        current_vel_info_(2,2) = 10;
      }
      auto ego_vel_edge = graph_slam->add_3d_velocity_edge(vel_ver_window[i], current_ego_v, current_vel_info_);

      Eigen::Vector4d last_ground;
      if(prev_keyframe->floor_coeffs) 
      {
        last_ground = *(prev_keyframe->floor_coeffs);
      }
      else
      {
        last_ground = Eigen::Vector4d(0.0, 0.0, 1.0, 0.5);
        ROS_WARN("no floor coeffs of last keyframe, use default ground plane!");
      }
      auto ground_plane_node = graph_slam->add_plane_node(last_ground);
      // auto ground_plane_node = graph_slam->add_plane_node(Eigen::Vector4d(0.0, 0.0, 1.0, 0.5));
      ground_plane_node->setFixed(true);
      Eigen::Matrix3d ground_information = Eigen::Matrix3d::Identity() * (1.0 / floor_edge_stddev);
      auto ground_edge = graph_slam->add_se3_plane_edge(this_keyframe->node, ground_plane_node, ground_coeffs, ground_information);
      graph_slam->add_robust_kernel(ground_edge, "Huber", 1.0);

    }

    int num_iterations = private_nh.param<int>("g2o_solver_num_iterations", 1024);
    clock_t start_ms = clock();
    graph_slam->optimize(num_iterations);
    clock_t end_ms = clock();
    double time_used = double(end_ms - start_ms) / CLOCKS_PER_SEC;
    opt_time.push_back(time_used);
    ROS_INFO("Pose graph optimization time: %f", time_used);
    bg_window.clear();
    ba_window.clear();

    //get results and update window
    for(int i = 0; i < frames_window.size() ; i++)
    { 
      const auto& this_keyframe = frames_window[i];
      NavStated this_nav_state;
      this_nav_state.R_ = SO3(this_keyframe->node->estimate().linear());
      this_nav_state.p_ = this_keyframe->node->estimate().translation();
      this_nav_state.v_ = vel_ver_window[i]->estimate();
      this_nav_state.bg_ = bg_ver_window[i]->estimate();
      this_nav_state.ba_ = ba_ver_window[i]->estimate();

      this_nav_state.timestamp_ = this_keyframe->stamp.toSec();

      if(failureDetection(this_nav_state.v_,this_nav_state.bg_,this_nav_state.ba_))
      {
        ROS_ERROR("failure detected");
        this_nav_state.ba_ = b_a_in;
        this_nav_state.bg_ = b_g_in;
        Eigen::Vector3d current_ego_v_ = Eigen::Vector3d(twist_window[i]->twist.twist.linear.x, twist_window[i]->twist.twist.linear.y, twist_window[i]->twist.twist.linear.z);
        // ROS_INFO("current ego velocity local: %f, %f, %f", current_ego_v(0), current_ego_v(1), current_ego_v(2));
        current_ego_v_ = navstates_window[i].R_ * current_ego_v_;
        this_nav_state.v_ = current_ego_v_;
        this_nav_state.p_ = odom_window[i].translation();

        //if detected failure, reset the preintegration, we can use the reset function of liosam
        // if (this_nav_state.v_[0] > 2.0) {
        //     this_nav_state.v_[0] = 2.0;
        // }
        // if (this_nav_state.v_[0] < -2.0) {
        //     this_nav_state.v_[0] = -2.0;
        // }
        // if (this_nav_state.v_[1] > 2.0) {
        //     this_nav_state.v_[1] = 2.0;
        // }

        // if (this_nav_state.v_[1] < -2.0) {
        //     this_nav_state.v_[1] = -2.0;
        // }

        // if (this_nav_state.v_[2] > 1.0) {
        //     this_nav_state.v_[2] = 1.0;
        // } else if (this_nav_state.v_[2] < -1.0) {
        //     this_nav_state.v_[2] = -1.0;
        // }
        // this_nav_state.p_ = this_odom.translation();
        // this_nav_state.R_ = SO3(this_odom.linear());
      }
      bg_window.push_back(this_nav_state.bg_);
      ba_window.push_back(this_nav_state.ba_);
      navstates_window[i] = this_nav_state;
    }


    ba_ver_window.clear();
    bg_ver_window.clear();
    vel_ver_window.clear();
    last_nav_state_ = navstates_window.back();

    predict_preinteg_options_.init_bg_ = bg_window.back();
    predict_preinteg_options_.init_ba_ = ba_window.back();
    preinteg_predict = std::make_shared<IMUPreintegrator>(predict_preinteg_options_);

    Eigen::Isometry3d trans = frames_window.back()->node->estimate() * frames_window.back()->odom_scan2scan.inverse();
    Eigen::Isometry3d updated_map2base_trans = frames_window.back()->node->estimate();
    trans_odom2map_mutex.lock();
    trans_odom2map = trans.matrix();
    // map2base_incremental = map2base_last^(-1) * map2base_this 
    trans_aftmapped_incremental = trans_aftmapped.inverse() * updated_map2base_trans;
    trans_aftmapped = updated_map2base_trans;

    trans_odom2map_mutex.unlock();


    nav_msgs::Odometry aft = isometry2odom(frames_window.back()->stamp, trans_aftmapped, mapFrame, odometryFrame);
    last_aft_odom = aft;
    aftmapped_odom_pub.publish(aft);

    // Publish After-Mapped Odometry Incrementation
    nav_msgs::Odometry aft_incre = isometry2odom(frames_window.back()->stamp, trans_aftmapped_incremental, mapFrame, odometryFrame);
    aftmapped_odom_incremenral_pub.publish(aft_incre);

    // Publish /odom to /base_link
    if(odom2base_pub.getNumSubscribers()) {  // Returns the number of subscribers that are currently connected to this Publisher
      geometry_msgs::TransformStamped ts = matrix2transform(frames_window.back()->stamp, trans.matrix(), mapFrame, odometryFrame);
      odom2base_pub.publish(ts);
    }

    if(markers_pub.getNumSubscribers()) {
      auto markers = create_marker_array(ros::Time::now());
      markers_pub.publish(markers);
    }


    //pub optimized path for visualization
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = frames_window.back()->stamp;
    pose_stamped.header.frame_id = mapFrame;
    pose_stamped.pose = aft.pose.pose;


    optimized_path.poses.push_back(pose_stamped);
    optimized_path.header.stamp = frames_window.back()->stamp;
    optimized_path.header.frame_id = mapFrame;
    pubOptimizedPath.publish(optimized_path);

    last_nav_state_.p_(0) = aft.pose.pose.position.x;
    last_nav_state_.p_(1) = aft.pose.pose.position.y;
    last_nav_state_.p_(2) = aft.pose.pose.position.z;
    last_nav_state_.R_ = SO3::exp(Vector3d(aft.pose.pose.orientation.x, aft.pose.pose.orientation.y, aft.pose.pose.orientation.z));

    graph_slam.reset(new GraphSLAM(private_nh.param<std::string>("g2o_solver_type", "lm_var")));
  }

  void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg)
  {

    imu_mutex.lock();
    // send transform map to odom
    geometry_msgs::TransformStamped map2odom_transform; 
    map2odom_transform.header.stamp = imu_msg->header.stamp;
    map2odom_transform.header.frame_id = mapFrame;
    map2odom_transform.child_frame_id = odometryFrame;
    map2odom_transform.transform = isometry2transform( trans_aftmapped);
    map2odom_broadcaster.sendTransform(map2odom_transform);

    if(imu_msg->header.stamp.toSec() <= last_imu_time){
      ROS_WARN("imu message in disorder!");
    }
    sensor_msgs::Imu thisImu = imuConverter(*imu_msg);

    last_imu_time = imu_msg->header.stamp.toSec();
    imu_opt_queue.push_back(thisImu);
    double imu_time = imu_msg->header.stamp.toSec();
    double dt = ( last_imu_pre < 0) ? (1.0 / 200.0) : (imu_time - last_imu_pre);
    last_imu_pre = imu_time;
    preinteg_predict->propagate(dt, 
                                Eigen::Vector3d(thisImu.linear_acceleration.x,thisImu.linear_acceleration.y, thisImu.linear_acceleration.z), 
                                Eigen::Vector3d(thisImu.angular_velocity.x, thisImu.angular_velocity.y, thisImu.angular_velocity.z));

    //predict the imu preintegration
    NavStated current_pre_state = preinteg_predict->predict(last_nav_state_);


    SE3 pre_imu_odom_inc = current_pre_state.GetSE3().inverse() * last_nav_state_.GetSE3();
    Eigen::Isometry3d pre_odom_inc_isometry = Eigen::Isometry3d::Identity();
    pre_odom_inc_isometry.linear() = pre_imu_odom_inc.so3().matrix();
    pre_odom_inc_isometry.translation() = pre_imu_odom_inc.translation();
    
    nav_msgs::Odometry pre_odom_inc = isometry2odom(imu_msg->header.stamp, pre_odom_inc_isometry, odometryFrame, imuFrame);
    pre_odom_inc.twist.twist.linear.x = current_pre_state.v_(0);
    pre_odom_inc.twist.twist.linear.y = current_pre_state.v_(1);
    pre_odom_inc.twist.twist.linear.z = current_pre_state.v_(2);
    pre_odom_inc.twist.twist.angular.x = imu_msg->angular_velocity.x + last_nav_state_.bg_(0);
    pre_odom_inc.twist.twist.angular.y = imu_msg->angular_velocity.y + last_nav_state_.bg_(1);
    pre_odom_inc.twist.twist.angular.z = imu_msg->angular_velocity.z + last_nav_state_.bg_(2);
    pubImuOdometry.publish(pre_odom_inc);
    imu_mutex.unlock();
  }

 /**
   * @brief received floor coefficients are added to #floor_coeffs_queue
   * @param floor_coeffs_msg
   */
  void floor_coeffs_callback(const radar_graph_slam::FloorCoeffsConstPtr& floor_coeffs_msg) {
    if(floor_coeffs_msg->coeffs.empty()) {
      return;
    }

    std::lock_guard<std::mutex> lock(floor_coeffs_queue_mutex);
    floor_coeffs_queue.push_back(floor_coeffs_msg);
  }

  /**
   * @brief this method adds all the keyframes_ in #keyframe_queue to the pose graph (odometry edges)
   * @return if true, at least one keyframe_ was added to the pose graph
   */
  bool flush_keyframe_queue() {
    // std::lock_guard<std::mutex> lock(keyframe_queue_mutex);

    if(keyframe_queue.empty()) {
      return false;
    }

    // trans_odom2map_mutex.lock();
    // Eigen::Isometry3d odom2map(trans_odom2map.cast<double>());
    // trans_odom2map_mutex.unlock();

    int num_processed = 0;
    // ********** Select number of keyframess to be optimized **********
    for(int i = 0; i < std::min<int>(keyframe_queue.size(), max_keyframes_per_update); i++) {
      num_processed = i;

      const auto& keyframe = keyframe_queue[i];
      const auto this_odom = keyframe_queue[i]->riv_pose;
      // new_keyframess will be tested later for loop closure
      new_keyframes.push_back(keyframe);

      // add pose node
      // Eigen::Isometry3d odom = odom2map * keyframe->odom_scan2scan;
      // ********** Vertex of keyframess is contructed here ***********
      keyframe->node = loop_optimizer->add_se3_node(this_odom);
      keyframe_hash[keyframe->stamp] = keyframe;

      // fix the first node
      if(keyframes.empty() && new_keyframes.size() == 1) {
        if(private_nh.param<bool>("fix_first_node", false)) {
          Eigen::MatrixXd inf = Eigen::MatrixXd::Identity(6, 6);
          std::stringstream sst(private_nh.param<std::string>("fix_first_node_stddev", "1 1 1 1 1 1"));
          for(int i = 0; i < 6; i++) {
            double stddev = 1.0;
            sst >> stddev;
            inf(i, i) = 1.0 / stddev;
          }
          anchor_node = loop_optimizer->add_se3_node(Eigen::Isometry3d::Identity());
          anchor_node->setFixed(true);
          anchor_edge = loop_optimizer->add_se3_edge(anchor_node, keyframe->node, Eigen::Isometry3d::Identity(), inf);
        }
      }
      
      if(i == 0 && keyframes.empty()) {
        continue;
      }

      /***** Scan-to-Scan Add edge to between consecutive keyframes *****/
      const auto& prev_keyframe = i == 0 ? keyframes.back() : keyframe_queue[i - 1];
      // relative pose between odom of previous frame and this frame R2=R12*R1 => R12 = inv(R2) * R1
      Eigen::Isometry3d relative_pose = keyframe->odom_scan2scan.inverse() * prev_keyframe->odom_scan2scan;
      // calculate fitness score as information 
      Eigen::MatrixXd information = inf_calclator->calc_information_matrix(keyframe->cloud, prev_keyframe->cloud, relative_pose);
      auto edge = loop_optimizer->add_se3_edge(keyframe->node, prev_keyframe->node, relative_pose, information);
      // cout << information << endl;
      loop_optimizer->add_robust_kernel(edge, private_nh.param<std::string>("odometry_edge_robust_kernel", "NONE"), private_nh.param<double>("odometry_edge_robust_kernel_size", 1.0));
    }
    ROS_INFO("Added %d keyframes to the graph", num_processed + 1);

    keyframe_queue.erase(keyframe_queue.begin(), keyframe_queue.begin() + num_processed + 1);
    ROS_INFO("updated keyframe_queue size = %d", keyframe_queue.size());
    return true;
  }

  //this method optimize the pose graph at a fixed rate/duration(3s)
  /**
   * @brief Back-end Optimization. This methods adds all the data in the queues to the pose graph, and then optimizes the pose graph
   * @param event
   */
  void optimization_timer_callback(const ros::WallTimerEvent& event) {
    std::lock_guard<std::mutex> lock(main_thread_mutex);

    // add keyframes_ and floor coeffs in the queues to the pose graph
    // we just use the keyframe_queue to add keyframes_ to the pose graph for loop closure
    bool keyframe_updated = flush_keyframe_queue();

    if(!keyframe_updated) 
    {
      return;
    }
    
    // loop detection
    if(private_nh.param<bool>("enable_loop_closure", false)){
      ROS_INFO("Start to detect loop");
      std::vector<Loop::Ptr> loops = loop_detector->detect(keyframes, new_keyframes, *loop_optimizer);
    }

    // Copy "new_keyframes_" to vector  "keyframes_", "new_keyframes_" was used for loop detaction 
    //***************************keyframes are updated here**************************************************
    ROS_INFO("update new keyframes to keyframes");
    
    std::copy(new_keyframes.begin(), new_keyframes.end(), std::back_inserter(keyframes));
    new_keyframes.clear();
    bool loop_detected = false;
    ROS_INFO("start to add loop factors");

    if(private_nh.param<bool>("enable_loop_closure", false))
      loop_detected = addLoopFactor();
    if(loop_detected)
    {
      ROS_INFO("Loop detected! Start to optimize the pose graph");
      // optimize the pose graph
      int num_iterations = private_nh.param<int>("g2o_solver_num_iterations", 1024);
      clock_t start_ms = clock();
      loop_optimizer->optimize(num_iterations);
      clock_t end_ms = clock();
      double time_used = double(end_ms - start_ms) / CLOCKS_PER_SEC;
      opt_time.push_back(time_used);
    }
  }

  bool addLoopFactor()
  {
    ROS_INFO("Adding loop factor");
    if (loop_detector->loopIndexQueue.empty())
      return false;
    ROS_INFO("loop detected! Adding loop factor to the graph!");
    for (int i = 0; i < (int)loop_detector->loopIndexQueue.size(); ++i){
      int indexFrom = loop_detector->loopIndexQueue[i].first;
      int indexTo = loop_detector->loopIndexQueue[i].second;
      Eigen::Isometry3d poseBetween = loop_detector->loopPoseQueue[i];
      Eigen::MatrixXd information_matrix = loop_detector->loopInfoQueue[i];
      auto edge = loop_optimizer->add_se3_edge(keyframes[indexFrom]->node, keyframes[indexTo]->node, poseBetween, information_matrix);
      loop_optimizer->add_robust_kernel(edge, private_nh.param<std::string>("loop_closure_edge_robust_kernel", "NONE"), private_nh.param<double>("loop_closure_edge_robust_kernel_size", 1.0));
      return true;
    }
  }

  /**
   * @brief generate map point cloud and publish it
   * @param event
   */
  void map_points_publish_timer_callback(const ros::WallTimerEvent& event) {
    if(!map_points_pub.getNumSubscribers() || !graph_updated) {
      return;
    }
    std::vector<KeyFrameSnapshot::Ptr> snapshot;
    keyframes_snapshot_mutex.lock();
    snapshot = keyframes_snapshot;
    keyframes_snapshot_mutex.unlock();

    auto cloud = map_cloud_generator->generate(snapshot, map_cloud_resolution);
    if(!cloud) {
      return;
    }
    cloud->header.frame_id = mapFrame;
    cloud->header.stamp = snapshot.back()->cloud->header.stamp;

    sensor_msgs::PointCloud2Ptr cloud_msg(new sensor_msgs::PointCloud2());
    pcl::toROSMsg(*cloud, *cloud_msg);

    map_points_pub.publish(cloud_msg);
  }

  /**
   * @brief create visualization marker
   * @param stamp
   * @return
   */
  visualization_msgs::MarkerArray create_marker_array(const ros::Time& stamp) const {
    visualization_msgs::MarkerArray markers;
    if (show_sphere)
      markers.markers.resize(5);
    else
      markers.markers.resize(4);
    //add imu markers

    // to do

    // loop edges
    visualization_msgs::Marker& loop_marker = markers.markers[0];
    loop_marker.header.frame_id = "map";
    loop_marker.header.stamp = stamp;
    loop_marker.action = visualization_msgs::Marker::ADD;
    loop_marker.type = visualization_msgs::Marker::LINE_LIST;
    loop_marker.ns = "loop_edges";
    loop_marker.id = 1;
    loop_marker.pose.orientation.w = 1;
    loop_marker.scale.x = 0.1; loop_marker.scale.y = 0.1; loop_marker.scale.z = 0.1;
    loop_marker.color.r = 0.9; loop_marker.color.g = 0.9; loop_marker.color.b = 0;
    loop_marker.color.a = 1;
    // for (auto it = loop_detector->loopIndexContainer.begin(); it != loop_detector->loopIndexContainer.end(); ++it)
    // {
    //   int key_cur = it->first;
    //   int key_pre = it->second;
    //   geometry_msgs::Point p;
    //   Eigen::Vector3d pos = keyframes[key_cur]->node->estimate().translation();
    //   p.x = pos.x();
    //   p.y = pos.y();
    //   p.z = pos.z();
    //   loop_marker.points.push_back(p);
    //   pos = keyframes[key_pre]->node->estimate().translation();
    //   p.x = pos.x();
    //   p.y = pos.y();
    //   p.z = pos.z();
    //   loop_marker.points.push_back(p);
    // }

    // node markers
    visualization_msgs::Marker& traj_marker = markers.markers[1];
    traj_marker.header.frame_id = "map";
    traj_marker.header.stamp = stamp;
    traj_marker.ns = "nodes";
    traj_marker.id = 0;
    traj_marker.type = visualization_msgs::Marker::SPHERE_LIST;

    traj_marker.pose.orientation.w = 1.0;
    traj_marker.scale.x = traj_marker.scale.y = traj_marker.scale.z = 0.3;

    visualization_msgs::Marker& imu_marker = markers.markers[2];
    imu_marker.header = traj_marker.header;
    imu_marker.ns = "imu";
    imu_marker.id = 1;
    imu_marker.type = visualization_msgs::Marker::SPHERE_LIST;

    imu_marker.pose.orientation.w = 1.0;
    imu_marker.scale.x = imu_marker.scale.y = imu_marker.scale.z = 0.75;

    traj_marker.points.resize(keyframes.size());
    traj_marker.colors.resize(keyframes.size());
    for(size_t i = 0; i < keyframes.size(); i++) {
      Eigen::Vector3d pos = keyframes[i]->node->estimate().translation();
      traj_marker.points[i].x = pos.x();
      traj_marker.points[i].y = pos.y();
      traj_marker.points[i].z = pos.z();

      double p = static_cast<double>(i) / keyframes.size();
      traj_marker.colors[i].r = 0.0;//1.0 - p;
      traj_marker.colors[i].g = 1.0;//p;
      traj_marker.colors[i].b = 0.0;
      traj_marker.colors[i].a = 1.0;

      if(keyframes[i]->acceleration) {
        Eigen::Vector3d pos = keyframes[i]->node->estimate().translation();
        geometry_msgs::Point point;
        point.x = pos.x();
        point.y = pos.y();
        point.z = pos.z();

        std_msgs::ColorRGBA color;
        color.r = 0.0;
        color.g = 0.0;
        color.b = 1.0;
        color.a = 1.0;

        imu_marker.points.push_back(point);
        imu_marker.colors.push_back(color);
      }
    }

    // edge markers
    visualization_msgs::Marker& edge_marker = markers.markers[3];
    edge_marker.header.frame_id = "map";
    edge_marker.header.stamp = stamp;
    edge_marker.ns = "edges";
    edge_marker.id = 2;
    edge_marker.type = visualization_msgs::Marker::LINE_LIST;

    edge_marker.pose.orientation.w = 1.0;
    edge_marker.scale.x = 0.05;

    edge_marker.points.resize(graph_slam->graph->edges().size() * 2);
    edge_marker.colors.resize(graph_slam->graph->edges().size() * 2);

    auto edge_itr = graph_slam->graph->edges().begin();
    for(int i = 0; edge_itr != graph_slam->graph->edges().end(); edge_itr++, i++) {
      g2o::HyperGraph::Edge* edge = *edge_itr;
      g2o::EdgeSE3* edge_se3 = dynamic_cast<g2o::EdgeSE3*>(edge);
      if(edge_se3) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_se3->vertices()[0]);
        g2o::VertexSE3* v2 = dynamic_cast<g2o::VertexSE3*>(edge_se3->vertices()[1]);
        
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2 = v2->estimate().translation();

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z();
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z();

        double p1 = static_cast<double>(v1->id()) / graph_slam->graph->vertices().size();
        double p2 = static_cast<double>(v2->id()) / graph_slam->graph->vertices().size();
        edge_marker.colors[i * 2].r = 0.0;//1.0 - p1;
        edge_marker.colors[i * 2].g = 1.0;//p1;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].r = 0.0;//1.0 - p2;
        edge_marker.colors[i * 2 + 1].g = 1.0;//p2;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        if(std::abs(v1->id() - v2->id()) > 2) {
          // edge_marker.points[i * 2].z += 0.5;
          // edge_marker.points[i * 2 + 1].z += 0.5;
          edge_marker.colors[i * 2].r = 0.9;
          edge_marker.colors[i * 2].g = 0.9;
          edge_marker.colors[i * 2].b = 0.0;
          edge_marker.colors[i * 2 + 1].r = 0.9;
          edge_marker.colors[i * 2 + 1].g = 0.9;
          edge_marker.colors[i * 2 + 1].b = 0.0;
          edge_marker.colors[i * 2].a = 1.0;
          edge_marker.colors[i * 2 + 1].a = 1.0;
        }
        continue;
      }

      g2o::EdgeSE3Plane* edge_plane = dynamic_cast<g2o::EdgeSE3Plane*>(edge);
      if(edge_plane) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_plane->vertices()[0]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2(pt1.x(), pt1.y(), 0.0);

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z();
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z();

        edge_marker.colors[i * 2].b = 1.0;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].b = 1.0;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        continue;
      }

      g2o::EdgeSE3PriorXY* edge_priori_xy = dynamic_cast<g2o::EdgeSE3PriorXY*>(edge);
      if(edge_priori_xy) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_priori_xy->vertices()[0]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2 = Eigen::Vector3d::Zero();
        pt2.head<2>() = edge_priori_xy->measurement();

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z() + 0.5;
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z() + 0.5;

        edge_marker.colors[i * 2].r = 1.0;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].r = 1.0;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        continue;
      }

      g2o::EdgeSE3PriorXYZ* edge_priori_xyz = dynamic_cast<g2o::EdgeSE3PriorXYZ*>(edge);
      if(edge_priori_xyz) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_priori_xyz->vertices()[0]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2 = edge_priori_xyz->measurement();

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z() + 0.5;
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z();

        edge_marker.colors[i * 2].r = 1.0;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].r = 1.0;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        continue;
      }

      g2o::EdgeSE3Interial* edge_interial = dynamic_cast<g2o::EdgeSE3Interial*>(edge);
      if(edge_interial) {
        g2o::VertexSE3* p1 = dynamic_cast<g2o::VertexSE3*>(edge_interial->vertices()[4]);
        g2o::VertexVelocity* v1 = dynamic_cast<g2o::VertexVelocity*>(edge_interial->vertices()[5]);

        Eigen::Vector3d pt1 = p1->estimate().translation();
        Eigen::Vector3d vel = v1->estimate();

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z();
        edge_marker.points[i * 2 + 1].x = pt1.x() + vel.x();
        edge_marker.points[i * 2 + 1].y = pt1.y() + vel.y();
        edge_marker.points[i * 2 + 1].z = pt1.z() + vel.z();

        edge_marker.colors[i * 2].r = 1.0;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].r = 1.0;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        continue;
      }
    }

    if (show_sphere)
    {
      // sphere
      visualization_msgs::Marker& sphere_marker = markers.markers[4];
      sphere_marker.header.frame_id = "map";
      sphere_marker.header.stamp = stamp;
      sphere_marker.ns = "loop_close_radius";
      sphere_marker.id = 3;
      sphere_marker.type = visualization_msgs::Marker::SPHERE;

      if(!keyframes.empty()) {
        Eigen::Vector3d pos = keyframes.back()->node->estimate().translation();
        sphere_marker.pose.position.x = pos.x();
        sphere_marker.pose.position.y = pos.y();
        sphere_marker.pose.position.z = pos.z();
      }
      sphere_marker.pose.orientation.w = 1.0;
      sphere_marker.scale.x = sphere_marker.scale.y = sphere_marker.scale.z = loop_detector->get_distance_thresh() * 2.0;

      sphere_marker.color.r = 1.0;
      sphere_marker.color.a = 0.3;
    }

    return markers;
  }

/**
   * @brief load all data from a directory
   * @param req
   * @param res
   * @return
   */
  bool load_service(radar_graph_slam::LoadGraphRequest& req, radar_graph_slam::LoadGraphResponse& res) {
    std::lock_guard<std::mutex> lock(main_thread_mutex);

    std::string directory = req.path;

    std::cout << "loading data from:" << directory << std::endl;

    // Load graph.
    graph_slam->load(directory + "/graph.g2o");
    
    // Iterate over the items in this directory and count how many sub directories there are. 
    // This will give an upper limit on how many keyframe indexes we can expect to find.
    boost::filesystem::directory_iterator begin(directory), end;
    int max_directory_count = std::count_if(begin, end,
        [](const boost::filesystem::directory_entry & d) {
            return boost::filesystem::is_directory(d.path()); // only return true if a direcotry
    });

    // Load keyframes by looping through key frame indexes that we expect to see.
    for(int i = 0; i < max_directory_count; i++) {
      std::stringstream sst;
      sst << boost::format("%s/%06d") % directory % i;
      std::string key_frame_directory = sst.str();

      // If key_frame_directory doesnt exist, then we have run out so lets stop looking.
      if(!boost::filesystem::is_directory(key_frame_directory)) {
        break;
      }

      KeyFrame::Ptr keyframe(new KeyFrame(key_frame_directory, graph_slam->graph.get()));
      keyframes.push_back(keyframe);
    }
    std::cout << "loaded " << keyframes.size() << " keyframes" <<std::endl;
    
    // Load special nodes.
    std::ifstream ifs(directory + "/special_nodes.csv");
    if(!ifs) {
      return false;
    }
    while(!ifs.eof()) {
      std::string token;
      ifs >> token;
      if(token == "anchor_node") {
        int id = 0;
        ifs >> id;
        anchor_node = static_cast<g2o::VertexSE3*>(graph_slam->graph->vertex(id));
      } else if(token == "anchor_edge") {
        int id = 0;
        ifs >> id; 
        // We have no way of directly pulling the edge based on the edge ID that we have just read in.
        // Fortunatly anchor edges are always attached to the anchor node so we can iterate over 
        // the edges that listed against the node untill we find the one that matches our ID.
        if(anchor_node){
          auto edges = anchor_node->edges();

          for(auto e : edges) {
            int edgeID =  e->id();
            if (edgeID == id){
              anchor_edge = static_cast<g2o::EdgeSE3*>(e);

              break;
            }
          } 
        }
      } else if(token == "floor_node") {
        int id = 0;
        ifs >> id;
        floor_plane_node = static_cast<g2o::VertexPlane*>(graph_slam->graph->vertex(id));
      }
    }

    // check if we have any non null special nodes, if all are null then lets not bother.
    if(anchor_node->id() || anchor_edge->id() || floor_plane_node->id()) {
      std::cout << "loaded special nodes - ";

      // check exists before printing information about each special node
      if(anchor_node->id()) {
        std::cout << " anchor_node: " << anchor_node->id();
      }
      if(anchor_edge->id()) {
        std::cout << " anchor_edge: " << anchor_edge->id();
      }
      if(floor_plane_node->id()) {
        std::cout << " floor_node: " << floor_plane_node->id();
      }
      
      // finish with a new line
      std::cout << std::endl;
    }

    // Update our keyframe snapshot so we can publish a map update, trigger update with graph_updated = true.
    std::vector<KeyFrameSnapshot::Ptr> snapshot(keyframes.size());

    std::transform(keyframes.begin(), keyframes.end(), snapshot.begin(), [=](const KeyFrame::Ptr& k) { return std::make_shared<KeyFrameSnapshot>(k); });

    keyframes_snapshot_mutex.lock();
    keyframes_snapshot.swap(snapshot);
    keyframes_snapshot_mutex.unlock();
    graph_updated = true;

    res.success = true;

    std::cout << "snapshot updated" << std::endl << "loading successful" <<std::endl;

    return true;
  }

  /**
   * @brief dump all data to the current directory
   * @param req
   * @param res
   * @return
   */
  bool dump_service(radar_graph_slam::DumpGraphRequest& req, radar_graph_slam::DumpGraphResponse& res) {
    std::lock_guard<std::mutex> lock(main_thread_mutex);

    std::string directory = req.destination;

    if(directory.empty()) {
      std::array<char, 64> buffer;
      buffer.fill(0);
      time_t rawtime;
      time(&rawtime);
      const auto timeinfo = localtime(&rawtime);
      strftime(buffer.data(), sizeof(buffer), "%d-%m-%Y %H:%M:%S", timeinfo);
    }

    if(!boost::filesystem::is_directory(directory)) {
      boost::filesystem::create_directory(directory);
    }

    std::cout << "all data dumped to:" << directory << std::endl;

    graph_slam->save(directory + "/graph.g2o");
    for(size_t i = 0; i < keyframes.size(); i++) {
      std::stringstream sst;
      sst << boost::format("%s/%06d") % directory % i;

      keyframes[i]->save(sst.str());
    }

    if(zero_utm) {
      std::ofstream zero_utm_ofs(directory + "/zero_utm");
      zero_utm_ofs << boost::format("%.6f %.6f %.6f") % zero_utm->x() % zero_utm->y() % zero_utm->z() << std::endl;
    }

    std::ofstream ofs(directory + "/special_nodes.csv");
    ofs << "anchor_node " << (anchor_node == nullptr ? -1 : anchor_node->id()) << std::endl;
    ofs << "anchor_edge " << (anchor_edge == nullptr ? -1 : anchor_edge->id()) << std::endl;
    ofs << "floor_node " << (floor_plane_node == nullptr ? -1 : floor_plane_node->id()) << std::endl;

    res.success = true;
    return true;
  }

  /**
   * @brief save map data as pcd
   * @param req
   * @param res
   * @return
   */
  bool save_map_service(radar_graph_slam::SaveMapRequest& req, radar_graph_slam::SaveMapResponse& res) {
    std::vector<KeyFrameSnapshot::Ptr> snapshot;

    keyframes_snapshot_mutex.lock();
    snapshot = keyframes_snapshot;
    keyframes_snapshot_mutex.unlock();

    auto cloud = map_cloud_generator->generate(snapshot, req.resolution);
    if(!cloud) {
      res.success = false;
      return true;
    }

    if(zero_utm && req.utm) {
      for(auto& pt : cloud->points) {
        pt.getVector3fMap() += (*zero_utm).cast<float>();
      }
    }

    cloud->header.frame_id = mapFrame;
    cloud->header.stamp = snapshot.back()->cloud->header.stamp;

    if(zero_utm) {
      std::ofstream ofs(req.destination + ".utm");
      ofs << boost::format("%.6f %.6f %.6f") % zero_utm->x() % zero_utm->y() % zero_utm->z() << std::endl;
    }

    int ret = pcl::io::savePCDFileBinary(req.destination, *cloud);
    res.success = ret == 0;

    return true;
  }
  
  void command_callback(const std_msgs::String& str_msg) {
    if (str_msg.data == "output_aftmapped") {
      ofstream fout;
      fout.open("/home/wayne/stamped_pose_graph_estimate.txt", ios::out);
      fout << "# timestamp tx ty tz qx qy qz qw" << endl;
      fout.setf(ios::fixed, ios::floatfield);  // fixed mode，float
      fout.precision(8);  // Set precision 8
      for(size_t i = 0; i < keyframes.size(); i++) {
        Eigen::Vector3d pos_ = keyframes[i]->node->estimate().translation();
        Eigen::Matrix3d rot_ = keyframes[i]->node->estimate().rotation();
        Eigen::Quaterniond quat_(rot_);
        double timestamp = keyframes[i]->stamp.toSec();
        double tx = pos_(0), ty = pos_(1), tz = pos_(2);
        double qx = quat_.x(), qy = quat_.y(), qz = quat_.z(), qw = quat_.w();

        fout << timestamp << " "
          << tx << " " << ty << " " << tz << " "
          << qx << " " << qy << " " << qz << " " << qw << endl;
      }
      fout.close();
      ROS_INFO("Optimized edges have been output!");
    }
    else if (str_msg.data == "time") {
      if (loop_detector->pf_time.size() > 0) {
        std::sort(loop_detector->pf_time.begin(), loop_detector->pf_time.end());
        double median = loop_detector->pf_time.at(floor((double)loop_detector->pf_time.size() / 2));
        cout << "Pre-filtering Matching time cost (median): " << median << endl;
      }
      if (loop_detector->sc_time.size() > 0) {
        std::sort(loop_detector->sc_time.begin(), loop_detector->sc_time.end());
        double median = loop_detector->sc_time.at(floor((double)loop_detector->sc_time.size() / 2));
        cout << "Scan Context time cost (median): " << median << endl;
      }
      if (loop_detector->oc_time.size() > 0) {
        std::sort(loop_detector->oc_time.begin(), loop_detector->oc_time.end());
        double median = loop_detector->oc_time.at(floor((double)loop_detector->oc_time.size() / 2));
        cout << "Odometry Check time cost (median): " << median << endl;
      }
      if (opt_time.size() > 0) {
        std::sort(opt_time.begin(), opt_time.end());
        double median = opt_time.at(floor((double)opt_time.size() / 2));
        cout << "Optimization time cost (median): " << median << endl;
      }
    }
  }


  inline sensor_msgs::Imu imuConverter(const sensor_msgs::Imu& imu_in)
  {
    sensor_msgs::Imu imu_out = imu_in;
    Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
    acc = extRot * acc;
    imu_out.linear_acceleration.x = acc.x();
    imu_out.linear_acceleration.y = acc.y();
    imu_out.linear_acceleration.z = acc.z();
    // rotate gyroscope
    Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
    gyr = extRot * gyr;
    imu_out.angular_velocity.x = gyr.x();
    imu_out.angular_velocity.y = gyr.y();
    imu_out.angular_velocity.z = gyr.z();
    // rotate roll pitch yaw
    Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
    // Eigen::Quaterniond q_from(q_out.x(), q_out.y(), q_out.z(), q_out.w());

    Eigen::Quaterniond q_final = q_from * extQRPY;
    imu_out.orientation.x = q_final.x();
    imu_out.orientation.y = q_final.y();
    imu_out.orientation.z = q_final.z();
    imu_out.orientation.w = q_final.w();

    if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
    {
      ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
      ros::shutdown();
    }
    return imu_out;
  }

  bool failureDetection(const Vec3d& velCur, const Vec3d& gyr_bias, const Vec3d& acc_bias)
  {
      Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
      Eigen::Vector3f ba(acc_bias(0), acc_bias(1), acc_bias(2));
      Eigen::Vector3f bg(gyr_bias(0), gyr_bias(1), gyr_bias(2));
      // ROS_ERROR("failureDetection Velocity Inc Norm = %f", vel.norm());
      // ROS_WARN_THROTTLE(5, "Velocity Inc Norm = %f", vel.norm());
      if (vel.norm() > 100.0)
      {
          ROS_ERROR("Large velocity, v = %f, %f, %f, reset IMU-preintegration!", vel(0), vel(1), vel(2));
          return true;
      }

      if (ba.norm() > 0.5 || bg.norm() > 0.5)
      {

          ROS_ERROR("Large bias, ba = %f, %f, %f, bg = %f, %f, %f, reset IMU-preintegration!", ba(0), ba(1), ba(2), bg(0), bg(1), bg(2));
          return true;
      }
      return false;
  }

private:
  // ROS
  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;
  ros::WallTimer optimization_timer;
  ros::WallTimer map_publish_timer;

  std::unique_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_sub;
  std::unique_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> cloud_sub;
  std::unique_ptr<message_filters::Subscriber<radar_graph_slam::FloorCoeffs>> ground_sub;

  std::unique_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync;
  std::unique_ptr<message_filters::Subscriber<geometry_msgs::TwistWithCovarianceStamped>> ego_twist_sub;


  ros::Subscriber imu_odom_sub;
  ros::Subscriber imu_sub;
  ros::Subscriber floor_sub;
  ros::Subscriber command_sub;
  // ros::Subscriber ego_twist_sub;
  ros::Publisher pub_twist_stamped;
  ros::Publisher pubImuOdometry;
  // ros::Publisher pubImuIntegOdometry;
  // ros::Publisher pubImuOdometryInc;
  // ros::Publisher pubImuPath;
  ros::Publisher markers_pub;
  

  std::mutex trans_odom2map_mutex;
  std::mutex imu_mutex;

  Eigen::Matrix4d trans_odom2map; // keyframe->node->estimate() * keyframe->odom.inverse();
  Eigen::Isometry3d trans_aftmapped;  // Odometry from /map to /base_link
  Eigen::Isometry3d trans_aftmapped_incremental;
  ros::Publisher odom2base_pub;
  ros::Publisher aftmapped_odom_pub;
  ros::Publisher aftmapped_odom_incremenral_pub;
  ros::Publisher odom_frame2frame_pub;

  std::string points_topic;
  ros::Publisher read_until_pub;
  ros::Publisher map_points_pub;

  tf::TransformListener tf_listener;
  tf::TransformBroadcaster map2base_broadcaster; // odom_frame => base_frame
  tf::TransformBroadcaster map2odom_broadcaster; // map_frame => odom_frame

  ros::ServiceServer load_service_server;
  ros::ServiceServer dump_service_server;
  ros::ServiceServer save_map_service_server; 

  // keyframe queue
  std::mutex keyframe_queue_mutex;
  std::deque<KeyFrame::Ptr> keyframe_queue;
  std::deque<Eigen::Isometry3d> keyframe_pose_queue;
  std::mutex frame_queue_mutex;
  std::deque<KeyFrame::Ptr> frame_queue;

  std::deque<nav_msgs::OdometryConstPtr> imu_odom_queue;
  std::deque<sensor_msgs::Imu> imu_opt_queue;
  std::deque<std::deque<sensor_msgs::Imu>> frames_imu_queue;

  double thisKeyframeTime;
  double lastKeyframeTime;
  double last_imu_time = 0;
  size_t keyframe_index = 0;
  size_t frame_index = 0;


  Eigen::Matrix4d initial_pose;

  std::shared_ptr<IMUPreintegrator> preinteg_opt = nullptr;
  std::shared_ptr<IMUPreintegrator> preinteg_predict = nullptr;

  // floor_coeffs queue
  double floor_edge_stddev;
  std::mutex floor_coeffs_queue_mutex;
  std::deque<radar_graph_slam::FloorCoeffsConstPtr> floor_coeffs_queue;

  boost::optional<Eigen::Vector3d> zero_utm;

  // Marker coefficients
  bool show_sphere;

  // for map cloud generation
  std::atomic_bool graph_updated;
  double map_cloud_resolution;
  std::mutex keyframes_snapshot_mutex;
  std::vector<KeyFrameSnapshot::Ptr> keyframes_snapshot;
  std::unique_ptr<MapCloudGenerator> map_cloud_generator;

  // graph slam
  // all the below members must be accessed after locking main_thread_mutex
  std::mutex main_thread_mutex;

  int max_keyframes_per_update;
  //  Used for Loop Closure detection source, 
  //  pushed form keyframe_queue at "flush_keyframe_queue()", 
  //  inserted to "keyframes_" before optimization
  std::deque<KeyFrame::Ptr> new_keyframes;
  std::deque<KeyFrame::Ptr> frames_window;

  //  Previous keyframes_
  std::vector<KeyFrame::Ptr> keyframes;
  // std::vector<KeyFrame::Ptr> all_keyframes;

  std::unordered_map<ros::Time, KeyFrame::Ptr, RosTimeHash> keyframe_hash;
  g2o::VertexSE3* anchor_node;
  g2o::EdgeSE3* anchor_edge;
  g2o::VertexPlane* floor_plane_node;

  std::unique_ptr<GraphSLAM> graph_slam;
  std::unique_ptr<GraphSLAM> loop_optimizer;
  std::unique_ptr<LoopDetector> loop_detector;
  std::unique_ptr<KeyframeUpdater> keyframe_updater;
  std::unique_ptr<InformationMatrixCalculator> inf_calclator;

  // Registration Method
  pcl::Registration<PointT, PointT>::Ptr registration;
  pcl::KdTreeFLANN<PointT>::Ptr kdtreeHistoryKeyPoses;

  std::vector<double> opt_time;

  double lastIMU_opt_stamp = -1;
  double lastIMU_pre_stamp = -1;

  Eigen::Isometry3d last_imu_preinteg_transf;

  std::vector<IMUPreintegrator> imu_preintegrator_arr;

  Eigen::Matrix3d delta_R;
  Eigen::Vector3d delta_p;
  Eigen::Vector3d delta_v;  
  Eigen::Vector3d b_a;
  Eigen::Vector3d b_g;
  Eigen::Vector3d b_a_in;
  Eigen::Vector3d b_g_in;

  Eigen::Matrix<double, 15, 15> covariance;
  Eigen::Matrix<double, 15, 15> jacobian;


  std::vector<Eigen::Vector3d> acc_buf;
  std::vector<Eigen::Vector3d> gyr_buf;  
  std::vector<double> dt_buf;
  Eigen::Vector3d last_acc;
  Eigen::Vector3d last_gyr;
  Eigen::Matrix<double, 18, 18> noise;

  nav_msgs::Odometry last_aft_odom;
  nav_msgs::Path imuPath;

  NavStated last_nav_state_, current_nav_state_;  
  std::vector<NavStated> imu_states_;
  NavStated current_imu_predict_state;
  bool imu_preinteg_init_ = false;
  double lastImuT_imu = -1;
  SE3 last_pose_;
  IMUPreintegrator::Options preinteg_options_;  
  IMUPreintegrator::Options cur_preinteg_options_;  
  IMUPreintegrator::Options predict_preinteg_options_;  


  // std::shared_ptr<IMUPreintegrator> preinteg_ = nullptr;
  Eigen::Vector3d current_vel_map = Eigen::Vector3d::Zero();


  



  bool velocity_optimized = false;
  bool first_ego_vel = true;
  std::deque<Eigen::Vector3d> vel_queue;

  Mat15d prior_info_ = Mat15d::Identity(); 
  Eigen::Isometry3d last_odom;
  bool first_time = true;
  SE3 lastodomSE3;
  // double last_keyframe_stamp;
  // std::shared_ptr<KeyFrame> current_keyframe;
  // std::shared_ptr<KeyFrame> last_keyframe;
  // KeyFrame::Ptr& current_keyframe;
  // KeyFrame::Ptr& last_keyframe;
  nav_msgs::Path optimized_path;
  nav_msgs::Path SM_path;
  ros::Publisher pubSMPath;
  ros::Publisher pubOptimizedPath;
  double last_imu_stamp = -1;
  int num_of_preinteg = 0;

  //to do: add all deque to the keyframe class
  std::deque<g2o::VertexAccBias*> ba_ver_window;
  std::deque<g2o::VertexGyroBias*> bg_ver_window;
  std::deque<g2o::VertexVelocity*> vel_ver_window;
  std::deque<NavStated> navstates_window;
  std::deque<std::shared_ptr<radar_graph_slam::IMUPreintegrator>> preinteg_window;
  std::deque<geometry_msgs::TwistWithCovarianceStamped::Ptr> twist_window;
  std::deque<Eigen::Isometry3d> odom_window;
  std::deque<Vec3d> bg_window;
  std::deque<Vec3d> ba_window;
  bool optimize_window_init = false;


  double last_imu_pre = -1;
  double inertial_weight = 1.0;
  bool first_frame = true;
  std::string imuFrame{ "odom_imu" };

};

}  // namespace radar_graph_slam

PLUGINLIB_EXPORT_CLASS(radar_graph_slam::RadarGraphSlamNodelet, nodelet::Nodelet)
