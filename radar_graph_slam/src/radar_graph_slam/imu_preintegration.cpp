#include <radar_graph_slam/imu_preintegration.hpp>

namespace radar_graph_slam {

    using SO3 = Sophus::SO3d;
    using Vec3d = Eigen::Vector3d;
IMUPreintegrator::IMUPreintegrator(Options options){
    bg_ = options.init_bg_;
    ba_ = options.init_ba_;
    const float ng2 = options.noise_gyro_ * options.noise_gyro_;
    const float na2 = options.noise_acce_ * options.noise_acce_;
    noise_gyro_acce_.diagonal() << ng2, ng2, ng2, na2, na2, na2;
}
void IMUPreintegrator::propagate(const double dt, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr){
    if(dt_ == 0){
        last_acc = acc;
        last_gyr = gyr;
    }

    Eigen::Vector3d gyr_ = 0.5 * (last_gyr + gyr) - bg_;  
    Eigen::Vector3d acc_ = 0.5 * (last_acc + acc) - ba_;  

    dp_ = dp_ + dv_ * dt + 0.5f * dR_.matrix() * acc_ * dt * dt;
    dv_ = dv_ + dR_ * acc_ * dt;
    Eigen::Matrix<double, 9, 9> A_;
    A_.setIdentity();
    Eigen::Matrix<double, 9, 6> B;
    B.setZero();
    Eigen::Matrix3d acc_hat = SO3::hat(acc_);
    double dt2_ = dt * dt;

    A_.block<3, 3>(3, 0) = -dR_.matrix() * dt * acc_hat;
    A_.block<3, 3>(6, 0) = -0.5f * dR_.matrix() * acc_hat * dt2_;
    A_.block<3, 3>(6, 3) = dt * Eigen::Matrix3d::Identity();

    B.block<3, 3>(3, 3) = dR_.matrix() * dt;
    B.block<3, 3>(6, 3) = 0.5f * dR_.matrix() * dt2_;

    dP_dba_ = dP_dba_ + dV_dba_ * dt - 0.5f * dR_.matrix() * dt2_;                     
    dP_dbg_ = dP_dbg_ + dV_dbg_ * dt - 0.5f * dR_.matrix() * dt2_ * acc_hat * dR_dbg_;  
    dV_dba_ = dV_dba_ - dR_.matrix() * dt;                                              
    dV_dbg_ = dV_dbg_ - dR_.matrix() * dt * acc_hat * dR_dbg_;                          

    Eigen::Vector3d omega = gyr_ * dt;
    Eigen::Vector3d omega_ = -omega;        
    Eigen::Matrix3d rightJ;
    auto theta = omega_.norm();
    if (theta < 1e-6) {
        rightJ = SO3::Transformation::Identity();
    }
    else 
    {
        auto a = omega_;
        a.normalize();
        double sin_theta = std::sin(theta);
        double cos_theta = std::cos(theta);
        rightJ = (sin_theta / theta) * SO3::Transformation::Identity() + (1 - sin_theta / theta) * a * a.transpose() +
                (1 - cos_theta) / theta * SO3::hat(a);
    }

    SO3 deltaR = SO3::exp(omega);  
    dR_ = dR_ * deltaR;             
    A_.block<3, 3>(0, 0) = deltaR.matrix().transpose();
    B.block<3, 3>(0, 0) = rightJ * dt;

    cov_ = A_ * cov_ * A_.transpose() + B * noise_gyro_acce_ * B.transpose();

    dR_dbg_ = deltaR.matrix().transpose() * dR_dbg_ - rightJ * dt; 

    dt_ += dt;

}

SO3 IMUPreintegrator::GetDeltaRotation(const Vec3d &bg) { return dR_ * SO3::exp(dR_dbg_ * (bg - bg_)); }

Vec3d IMUPreintegrator::GetDeltaVelocity(const Vec3d &bg, const Vec3d &ba) {
    return dv_ + dV_dbg_ * (bg - bg_) + dV_dba_ * (ba - ba_);
}

Vec3d IMUPreintegrator::GetDeltaPosition(const Vec3d &bg, const Vec3d &ba) {
    return dp_ + dP_dbg_ * (bg - bg_) + dP_dba_ * (ba - ba_);
}
NavStated IMUPreintegrator::predict(const NavStated &start)  {

    SO3 this_R = start.R_*dR_;
    Eigen::Vector3d this_v = start.R_*dv_ + start.v_ - GravityVec * dt_;
    Eigen::Vector3d this_p = start.R_*dp_ + start.p_ + start.v_*dt_ - 0.5f*GravityVec*dt_*dt_;

    auto state = NavStated(start.timestamp_ + dt_, this_R, this_p, this_v);

    state.bg_ = bg_;
    state.ba_ = ba_;

    return state;
}
} // namespace radar_graph_slam