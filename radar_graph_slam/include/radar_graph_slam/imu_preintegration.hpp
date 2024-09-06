/**
 * @brief 
 * 
*/

#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <ros/ros.h>
#include "radar_graph_slam/nav_state.hpp"
#include "sophus/se2.hpp"
#include "sophus/se3.hpp"
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>
#include "utility_radar.h"

namespace radar_graph_slam {

    class IMUPreintegrator {

    public:
        using SO3 = Sophus::SO3d;
        using Vec3d = Eigen::Vector3d;
        using Mat3d = Eigen::Matrix3d;
        struct Options {
        Options() {}
        Vec3d init_bg_ = Vec3d::Zero();  
        Vec3d init_ba_ = Vec3d::Zero();  
        double noise_gyro_ = 1e-2;      
        double noise_acce_ = 1e-1;       

        double bias_gyro_var_ = 1e-2;          
        double bias_acce_var_ = 1e-2;          
        Mat3d bg_rw_info_ = Mat3d::Identity();  
        Mat3d ba_rw_info_ = Mat3d::Identity();  
        };
        IMUPreintegrator(Options options = Options());

        NavStated predict(const NavStated &start);
        void propagate(const double dt, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr);
        SO3 GetDeltaRotation(const Vec3d &bg);
        Vec3d GetDeltaVelocity(const Vec3d &bg, const Vec3d &ba);
        Vec3d GetDeltaPosition(const Vec3d &bg, const Vec3d &ba);

        double sum_dt;
        double dt_ = 0;                       
        SO3 dR_;
        Eigen::Vector3d dv_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d dp_ = Eigen::Vector3d::Zero(); 
        Eigen::Vector3d bg_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d ba_ = Eigen::Vector3d::Zero();


        Eigen::Matrix3d dR_dbg_ = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d dV_dbg_ = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d dV_dba_ = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d dP_dbg_ = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d dP_dba_ = Eigen::Matrix3d::Zero();

        Eigen::Matrix<double, 9, 9> cov_ = Eigen::Matrix<double, 9, 9>::Zero();              
        Eigen::Matrix<double, 6, 6> noise_gyro_acce_ = Eigen::Matrix<double, 6, 6>::Zero(); 

        private:

        Eigen::Vector3d last_acc;
        Eigen::Vector3d last_gyr;
        Vec3d GravityVec = Vec3d(0, 0, 9.80511);

    };

}
