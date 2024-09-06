#pragma once

// #include <g2o/types/slam3d/edge_se3.h>
// #include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/robust_kernel.h>
#include <radar_graph_slam/imu_preintegration.hpp>
#include <g2o/g2o_types.hpp>
#include "utility_radar.h"

#include <Eigen/Dense>
#include <math.h>

namespace g2o {
  using Vec3d = Eigen::Vector3d;
  using Vec9d = Eigen::Matrix<double, 9, 1>;
  using Mat3d = Eigen::Matrix3d;
  

class EdgeSE3Interial : public g2o::BaseMultiEdge<9, Vec9d> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSE3Interial() = default;


  /**
   * 
   * @param preinteg  
   * @param gravity   
   * @param weight   
   */
  // EdgeIMU(std::shared_ptr<radar_graph_slam::IMUPreintegrator> preinteg, const Vec3d& gravity);
  EdgeSE3Interial(std::shared_ptr<radar_graph_slam::IMUPreintegrator> preinteg, double weight): preint_(preinteg), dt_(preinteg->dt_)
  {
      resize(6);  
      setInformation(preinteg->cov_.inverse()* weight);    
  }
  bool read(std::istream& is) override { return false; }
  bool write(std::ostream& os) const override { return false; }

  void computeError() override
  {
    const g2o::VertexSE3* p1 = dynamic_cast<const g2o::VertexSE3*>(_vertices[0]);
    auto* v1 = dynamic_cast<const VertexVelocity*>(_vertices[1]);
    auto* bg1 = dynamic_cast<const VertexGyroBias*>(_vertices[2]);
    auto* ba1 = dynamic_cast<const VertexAccBias*>(_vertices[3]);
    const g2o::VertexSE3* p2 = dynamic_cast<const g2o::VertexSE3*>(_vertices[4]);
    auto* v2 = dynamic_cast<const VertexVelocity*>(_vertices[5]);

    Vec3d bg = bg1->estimate();
    Vec3d ba = ba1->estimate();

    const SO3 dR = preint_->GetDeltaRotation(bg);
    const Vec3d dv = preint_->GetDeltaVelocity(bg, ba);
    const Vec3d dp = preint_->GetDeltaPosition(bg, ba);

    const SO3 p_dr_1(p1->estimate().linear());
    const SO3 p_dr_2(p2->estimate().linear());

    const Vec3d er = (dR.inverse() * p_dr_1.inverse() * p_dr_2).log();
    Mat3d RiT = p_dr_1.inverse().matrix();
    const Vec3d ev = RiT * (v2->estimate() - v1->estimate() + GravityVec * dt_) - dv;
    const Vec3d ep = RiT * (p2->estimate().translation() - p1->estimate().translation() - v1->estimate() * dt_ + 0.5 * GravityVec * dt_*dt_) - dp;
    _error << er, ev, ep;
  }

  void linearizeOplus() override 
  {
    const g2o::VertexSE3* p1 = dynamic_cast<const g2o::VertexSE3*>(_vertices[0]);
    auto* v1 = dynamic_cast<const VertexVelocity*>(_vertices[1]);
    auto* bg1 = dynamic_cast<const VertexGyroBias*>(_vertices[2]);
    auto* ba1 = dynamic_cast<const VertexAccBias*>(_vertices[3]);
    const g2o::VertexSE3* p2 = dynamic_cast<const g2o::VertexSE3*>(_vertices[4]);
    auto* v2 = dynamic_cast<const VertexVelocity*>(_vertices[5]);

    Vec3d bg = bg1->estimate();
    Vec3d ba = ba1->estimate();
    Vec3d dbg = bg - preint_->bg_;

    // 
    const SO3 R1 (p1->estimate().linear());
    const SO3 R1T = R1.inverse();
    const SO3 R2(p2->estimate().linear());

    auto dR_dbg = preint_->dR_dbg_;
    auto dv_dbg = preint_->dV_dbg_;
    auto dp_dbg = preint_->dP_dbg_;
    auto dv_dba = preint_->dV_dba_;
    auto dp_dba = preint_->dP_dba_;

    // 
    Vec3d vi = v1->estimate();
    Vec3d vj = v2->estimate();
    Vec3d pi = p1->estimate().translation();
    Vec3d pj = p2->estimate().translation();

    const SO3 dR = preint_->GetDeltaRotation(bg);
    const SO3 eR = SO3(dR).inverse() * R1T * R2;
    const Vec3d er = eR.log();
    auto Omega = -eR.log();
    auto theta = Omega.norm();
    Mat3d invJr;

    if(theta < 1e-6)
    {
      invJr = SO3::Transformation::Identity();
    }
    else
    {
      Vec3d a = Omega;
      a.normalize();
      double cot_half_theta = cos(0.5 * theta) / sin(0.5 * theta);
      invJr = 0.5 * theta * cot_half_theta * SO3::Transformation::Identity() +
               (1 - 0.5 * theta * cot_half_theta) * a * a.transpose() - 0.5 * theta * SO3::hat(a); 
    }


    // const Mat3d invJr = SO3::jr_inv(eR);

    _jacobianOplus[0].setZero();
    // dR/dR1
    _jacobianOplus[0].block<3, 3>(0, 0) = -invJr * (R2.inverse() * R1).matrix();
    // dv/dR1
    _jacobianOplus[0].block<3, 3>(3, 0) = SO3::hat(R1T * (vj - vi + GravityVec * dt_));
    // dp/dR1
    _jacobianOplus[0].block<3, 3>(6, 0) = SO3::hat(R1T * (pj - pi - v1->estimate() * dt_ + 0.5 * GravityVec * dt_ * dt_));


    // dp/dp1
    _jacobianOplus[0].block<3, 3>(6, 3) = -R1T.matrix();


    _jacobianOplus[1].setZero();
    // dv/dv1
    _jacobianOplus[1].block<3, 3>(3, 0) = -R1T.matrix();
    // dp/dv1
    _jacobianOplus[1].block<3, 3>(6, 0) = -R1T.matrix() * dt_;

  
    _jacobianOplus[2].setZero();
    // dR/dbg1

    Mat3d jr_;
    auto Omega_ = - (dR_dbg * dbg).eval();
    auto theta_ = Omega_.norm();

    if(theta_ < 1e-6)
    {
      jr_ = SO3::Transformation::Identity();
    }
    else
    {
      Vec3d a_ = Omega_;
      a_.normalize();
      double sin_theta = std::sin(theta_);
      double cos_theta = std::cos(theta_);
      jr_ = (sin_theta / theta_) * SO3::Transformation::Identity() + (1 - sin_theta / theta_) * a_ * a_.transpose() +
               (1 - cos_theta) / theta_ * SO3::hat(a_);
    }

    _jacobianOplus[2].block<3, 3>(0, 0) = -invJr * eR.inverse().matrix() * jr_ * dR_dbg;
    // dv/dbg1
    _jacobianOplus[2].block<3, 3>(3, 0) = -dv_dbg;
    // dp/dbg1
    _jacobianOplus[2].block<3, 3>(6, 0) = -dp_dbg;

    _jacobianOplus[3].setZero();
    // dv/dba1
    _jacobianOplus[3].block<3, 3>(3, 0) = -dv_dba;
    // dp/dba1
    _jacobianOplus[3].block<3, 3>(6, 0) = -dp_dba;


    _jacobianOplus[4].setZero();
    // dr/dr2
    _jacobianOplus[4].block<3, 3>(0, 0) = invJr;
    // dp/dp2
    _jacobianOplus[4].block<3, 3>(6, 3) = R1T.matrix();


    _jacobianOplus[5].setZero();
    // dv/dv2
    _jacobianOplus[5].block<3, 3>(3, 0) = R1T.matrix();  // OK
}
  Eigen::Matrix<double, 24, 24> GetHessian() {
      linearizeOplus();
      Eigen::Matrix<double, 9, 24> J;
      J.block<9, 6>(0, 0) = _jacobianOplus[0];
      J.block<9, 3>(0, 6) = _jacobianOplus[1];
      J.block<9, 3>(0, 9) = _jacobianOplus[2];
      J.block<9, 3>(0, 12) = _jacobianOplus[3];
      J.block<9, 6>(0, 15) = _jacobianOplus[4];
      J.block<9, 3>(0, 21) = _jacobianOplus[5];
      return J.transpose() * information() * J;
  }
  private:
  double dt_;
  std::shared_ptr<radar_graph_slam::IMUPreintegrator> preint_ = nullptr;
  Vec3d GravityVec = Vec3d(0, 0, 9.80511);

};
}  // namespace g2o

// #endif
