

#ifndef SAD_NAV_STATE_H
#define SAD_NAV_STATE_H

#include <sophus/so3.hpp>
#include <sophus/se3.hpp>


namespace radar_graph_slam {

using SE3 = Sophus::SE3d;

/**
 * 
 * @tparam T    
 *
 * 
 */
template <typename T>
struct NavState {

    using SO3 = Sophus::SO3<T>;

    NavState() = default;

    // from time, R, p, v, bg, ba
    explicit NavState(double time, const SO3& R = SO3(), const Eigen::Vector3d& t = Eigen::Vector3d::Zero(), const Eigen::Vector3d& v = Eigen::Vector3d::Zero(),
                      const Eigen::Vector3d& bg = Eigen::Vector3d::Zero(), const Eigen::Vector3d& ba = Eigen::Vector3d::Zero())
        : timestamp_(time), R_(R), p_(t), v_(v), bg_(bg), ba_(ba) {}

    // from pose and vel
    NavState(double time, const SE3& pose, const Eigen::Vector3d& vel = Eigen::Vector3d::Zero())
        : timestamp_(time), R_(pose.so3()), p_(pose.translation()), v_(vel) {}

    Sophus::SE3<T> GetSE3() const { return SE3(R_, p_); }

    double timestamp_ = 0;    // 
    SO3 R_;                   // 
    Eigen::Vector3d p_ = Eigen::Vector3d::Zero();   // 
    Eigen::Vector3d v_ = Eigen::Vector3d::Zero();   // 
    Eigen::Vector3d bg_ = Eigen::Vector3d::Zero();  // gyro 
    Eigen::Vector3d ba_ = Eigen::Vector3d::Zero();  // acce 
};

using NavStated = NavState<double>;
using NavStatef = NavState<float>;



}  // namespace radar_graph_slam

#endif
