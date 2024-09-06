#pragma once

// #include <g2o/types/slam3d/edge_se3.h>
// #include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/robust_kernel.h>
// #include <radar_graph_slam/imu_preintegration.hpp>
#include <g2o/g2o_types.hpp>
// #include "utils.h"


// #include <g2o/types/sba/edge_se3_expmap.h>
// #include <g2o/types/sba/vertex_se3_expmap.h>

#include <Eigen/Dense>
#include <math.h>

namespace g2o {
  using Vec3d = Eigen::Vector3d;
  using Vec9d = Eigen::Matrix<double, 9, 1>;
  using Mat3d = Eigen::Matrix3d;

class EdgeRadar3DVelocity : public g2o::BaseUnaryEdge<3, Vec3d,g2o::VertexVelocity> {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // EdgeRadar3DVelocity() = default;


    /**
     * 
     * @param v0
     * @param speed
     */
  EdgeRadar3DVelocity() : g2o::BaseUnaryEdge<3, Vec3d,g2o::VertexVelocity>() {}

  virtual bool read(std::istream& is) override { return false; }
  virtual bool write(std::ostream& os) const override { return false; }

  // void linearizeOplus() override { _jacobianOplusXi.setIdentity(); }

  void computeError() override 
  {
    const g2o::VertexVelocity* v0 = static_cast<const g2o::VertexVelocity*>(_vertices[0]);
    _error = v0->estimate() - _measurement;
  }
  void setMeasurement(const Vec3d& m) override 
  {
    _measurement = m;
  }

};
}  // namespace g2o

// #endif
