// SPDX-License-Identifier: BSD-2-Clause

#ifndef GRAPH_SLAM_HPP
#define GRAPH_SLAM_HPP

#include <memory>
#include <ros/time.h>

#include <g2o/core/hyper_graph.h>
#include <g2o/edge_se3_priorz.hpp>
#include <g2o/edge_se3_z.hpp>
#include <g2o/edge_se3_se3.hpp>
#include <g2o/edge_se3_gt_utm.hpp>
// #include <g2o/edge_multi_imu.hpp>
#include <g2o/g2o_types.hpp>
#include <g2o/edge_se3_interial.hpp>
#include <g2o/edge_3d_velocity.hpp>


namespace g2o {
class VertexSE3;
class VertexPlane;
class VertexPointXYZ;
class EdgeSE3;
class EdgeSE3Plane;
class EdgeSE3PointXYZ;
class EdgeSE3PriorXY;
class EdgeSE3PriorXYZ;
class EdgeSE3PriorVec;
class EdgeSE3PriorQuat;
class EdgePlane;
class EdgePlaneIdentity;
class EdgePlaneParallel;
class EdgePlanePerpendicular;
class EdgePlanePriorNormal;
class EdgePlanePriorDistance;
// class EdgeIMU;
class VertexPose;
class VertexVelocity;
class VertexGyroBias;
class VertexAccBias;
class EdgeSE3Interial;
class EdgeRadar3DVelocity;
class EdgePriorPoseNavState;
class EdgeGyroRW;
class EdgeAccRW;
class EdgePose;
class RobustKernelFactory;


}  // namespace g2o

namespace radar_graph_slam {
  using Vec9d = Eigen::Matrix<double, 9, 1>;

  using Vec6d = Eigen::Matrix<double, 6, 1>;
  using Vec3d = Eigen::Vector3d;
  using Mat15d = Eigen::Matrix<double, 15, 15>;
  using Vec15d = Eigen::Matrix<double, 15, 15>;
class GraphSLAM {
public:
  GraphSLAM(const std::string& solver_type = "lm_var");
  virtual ~GraphSLAM();

  int num_vertices() const;
  int num_edges() const;

  void set_solver(const std::string& solver_type);

  /**
   * @brief add a SE3 node to the graph
   * @param pose
   * @return registered node
   */
  g2o::VertexSE3* add_se3_node(const Eigen::Isometry3d& pose);

  /**
   * @brief add a plane node to the graph
   * @param plane_coeffs
   * @return registered node
   */
  g2o::VertexPlane* add_plane_node(const Eigen::Vector4d& plane_coeffs);

  /**
   * @brief add a point_xyz node to the graph
   * @param xyz
   * @return registered node
   */
  g2o::VertexPointXYZ* add_point_xyz_node(const Eigen::Vector3d& xyz);

  /**
   * @brief 
   * 
   * @param vel 
   * @return g2o::VertexVelocity* 
   */
  g2o::VertexVelocity* add_velocity_node(const Eigen::Vector3d& vel);
  g2o::VertexGyroBias* add_gyro_bais_node(const Eigen::Vector3d& gyro_bias);
  g2o::VertexAccBias* add_acc_bias_node(const Eigen::Vector3d& acc_bias);

  /**
   * @brief add an edge between SE3 nodes
   * @param v1  node1
   * @param v2  node2
   * @param relative_pose  relative pose between node1 and node2
   * @param information_matrix  information matrix (it must be 6x6)
   * @return registered edge
   */
  g2o::EdgeSE3* add_se3_edge(g2o::VertexSE3* v1, g2o::VertexSE3* v2, const Eigen::Isometry3d& relative_pose, const Eigen::MatrixXd& information_matrix);

  g2o::EdgeSE3Prior* add_se3_prior_edge(g2o::VertexSE3* v_se3, const Eigen::Isometry3d& pose, const Eigen::MatrixXd& information_matrix);

  g2o::EdgeSE3SE3* add_se3_se3_edge(g2o::VertexSE3* v1, g2o::VertexSE3* v2, const g2o::SE3Quat& relative_pose, const Eigen::MatrixXd& information_matrix);
  /**
   * @brief add an edge between an SE3 node and a plane node
   * @param v_se3    SE3 node
   * @param v_plane  plane node
   * @param plane_coeffs  plane coefficients w.r.t. v_se3
   * @param information_matrix  information matrix (it must be 3x3)
   * @return registered edge
   */
  g2o::EdgeSE3Plane* add_se3_plane_edge(g2o::VertexSE3* v_se3, g2o::VertexPlane* v_plane, const Eigen::Vector4d& plane_coeffs, const Eigen::MatrixXd& information_matrix);

  /**
   * @brief add an edge between an SE3 node and a point_xyz node
   * @param v_se3        SE3 node
   * @param v_xyz        point_xyz node
   * @param xyz          xyz coordinate
   * @param information  information_matrix (it must be 3x3)
   * @return registered edge
   */
  g2o::EdgeSE3PointXYZ* add_se3_point_xyz_edge(g2o::VertexSE3* v_se3, g2o::VertexPointXYZ* v_xyz, const Eigen::Vector3d& xyz, const Eigen::MatrixXd& information_matrix);

  /**
   * @brief add a prior edge to an SE3 node
   * @param v_se3
   * @param xy
   * @param information_matrix
   * @return
   */
  g2o::EdgeSE3GtUTM* add_se3_gt_utm_edge(g2o::VertexSE3* v_se3, const Eigen::Vector3d& gt_xyz, const Eigen::Vector3d& utm_xyz, const Eigen::MatrixXd& information_matrix);
  g2o::EdgePlanePriorNormal* add_plane_normal_prior_edge(g2o::VertexPlane* v, const Eigen::Vector3d& normal, const Eigen::MatrixXd& information_matrix);

  g2o::EdgePlanePriorDistance* add_plane_distance_prior_edge(g2o::VertexPlane* v, double distance, const Eigen::MatrixXd& information_matrix);

  g2o::EdgeSE3PriorXY* add_se3_prior_xy_edge(g2o::VertexSE3* v_se3, const Eigen::Vector2d& xy, const Eigen::MatrixXd& information_matrix);

  g2o::EdgeSE3PriorXYZ* add_se3_prior_xyz_edge(g2o::VertexSE3* v_se3, const Eigen::Vector3d& xyz, const Eigen::MatrixXd& information_matrix);

  g2o::EdgeSE3PriorZ* add_se3_prior_z_edge(g2o::VertexSE3* v_se3, const Eigen::Vector1d& z, const Eigen::MatrixXd& information_matrix);

  g2o::EdgeSE3Z* add_se3_z_edge(g2o::VertexSE3* v1, g2o::VertexSE3* v2, const Eigen::Vector1d& z, const Eigen::MatrixXd& information_matrix);

  g2o::EdgeSE3PriorQuat* add_se3_prior_quat_edge(g2o::VertexSE3* v_se3, const Eigen::Quaterniond& quat, const Eigen::MatrixXd& information_matrix);

  g2o::EdgeSE3PriorVec* add_se3_prior_vec_edge(g2o::VertexSE3* v_se3, const Eigen::Vector3d& direction, const Eigen::Vector3d& measurement, const Eigen::MatrixXd& information_matrix);

  g2o::EdgePlane* add_plane_edge(g2o::VertexPlane* v_plane1, g2o::VertexPlane* v_plane2, const Eigen::Vector4d& measurement, const Eigen::Matrix4d& information);

  g2o::EdgePlaneIdentity* add_plane_identity_edge(g2o::VertexPlane* v_plane1, g2o::VertexPlane* v_plane2, const Eigen::Vector4d& measurement, const Eigen::Matrix4d& information);

  g2o::EdgePlaneParallel* add_plane_parallel_edge(g2o::VertexPlane* v_plane1, g2o::VertexPlane* v_plane2, const Eigen::Vector3d& measurement, const Eigen::Matrix3d& information);

  g2o::EdgePlanePerpendicular* add_plane_perpendicular_edge(g2o::VertexPlane* v_plane1, g2o::VertexPlane* v_plane2, const Eigen::Vector3d& measurement, const Eigen::MatrixXd& information);

  // g2o::EdgeIMU* add_imu_multi_edge(g2o::VertexPose* v0_pose, g2o::VertexVelocity* v0_vel,
  //                                 g2o::VertexGyroBias* v0_bg, g2o::VertexAccBias* v0_ba,
  //                                 g2o::VertexPose* v1_pose,g2o::VertexVelocity* v1_vel, 
  //                                 std::shared_ptr<radar_graph_slam::IMUPreintegrator> preinteg, const Eigen::Vector3d& gravity);
  g2o::EdgeSE3Interial* add_se3_interial_edge(g2o::VertexSE3* v1,g2o::VertexVelocity* v2, g2o::VertexGyroBias* v3, g2o::VertexAccBias* v4, g2o::VertexSE3* v5, g2o::VertexVelocity* v6,
                                              std::shared_ptr<radar_graph_slam::IMUPreintegrator> preinteg, double weight);

  g2o::EdgeRadar3DVelocity* add_3d_velocity_edge(g2o::VertexVelocity* v1,const Vec3d& measurement,const Eigen::MatrixXd& information);

  g2o::EdgePriorPoseNavState* add_prior_state_edge(g2o::VertexSE3* v1, g2o::VertexVelocity* v2, g2o::VertexGyroBias* v3, g2o::VertexAccBias* v4,
                                              const NavStated& state,const Mat15d& information); 

  g2o::EdgeGyroRW* add_gyro_rw_edge(g2o::VertexGyroBias* v1,g2o::VertexGyroBias* v2,const Eigen::MatrixXd& information);
  g2o::EdgeAccRW* add_acc_rw_edge(g2o::VertexAccBias* v1,g2o::VertexAccBias* v2,const Eigen::MatrixXd& information);
  g2o::EdgePose* add_pose_edge(g2o::VertexSE3* v1, const SE3& pose, const Eigen::MatrixXd& information_matrix);

  

  void add_robust_kernel(g2o::HyperGraph::Edge* edge, const std::string& kernel_type, double kernel_size);

  /**
   * @brief perform graph optimization
   */
  int optimize(int num_iterations);

  /**
   * @brief save the pose graph to a file
   * @param filename  output filename
   */
  void save(const std::string& filename);

  /**
   * @brief load the pose graph from file
   * @param filename  output filename
   */
  bool load(const std::string& filename);

public:
  g2o::RobustKernelFactory* robust_kernel_factory;
  std::unique_ptr<g2o::HyperGraph> graph;  // g2o graph
};

}  // namespace radar_graph_slam

#endif  // GRAPH_SLAM_HPP
