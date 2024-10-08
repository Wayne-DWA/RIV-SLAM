// SPDX-License-Identifier: BSD-2-Clause


#include <radar_graph_slam/graph_slam.hpp>

#include <boost/format.hpp>
#include <g2o/stuff/macros.h>
#include <g2o/core/factory.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/linear_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/solvers/pcg/linear_solver_pcg.h>
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/types/slam3d/edge_se3_pointxyz.h>
#include <g2o/types/slam3d_addons/types_slam3d_addons.h>
#include <g2o/edge_se3_plane.hpp>
#include <g2o/edge_se3_priorxy.hpp>
#include <g2o/edge_se3_priorxyz.hpp>
#include <g2o/edge_se3_priorz.hpp>
#include <g2o/edge_se3_z.hpp>
#include <g2o/edge_se3_se3.hpp>
#include <g2o/edge_se3_priorvec.hpp>
#include <g2o/edge_se3_priorquat.hpp>
#include <g2o/edge_plane_prior.hpp>
#include <g2o/edge_plane_identity.hpp>
#include <g2o/edge_plane_parallel.hpp>
#include <g2o/robust_kernel_io.hpp>
#include <g2o/edge_se3_gt_utm.hpp>

G2O_USE_OPTIMIZATION_LIBRARY(pcg)
G2O_USE_OPTIMIZATION_LIBRARY(cholmod)  // be aware of that cholmod brings GPL dependency
G2O_USE_OPTIMIZATION_LIBRARY(csparse)  // be aware of that csparse brings LGPL unless it is dynamically linked

namespace g2o {
G2O_REGISTER_TYPE(EDGE_SE3_PLANE, EdgeSE3Plane)
G2O_REGISTER_TYPE(EDGE_SE3_PRIORXY, EdgeSE3PriorXY)
G2O_REGISTER_TYPE(EDGE_SE3_PRIORXYZ, EdgeSE3PriorXYZ)
G2O_REGISTER_TYPE(EDGE_SE3_PRIORZ, EdgeSE3PriorZ)
G2O_REGISTER_TYPE(EDGE_SE3_Z, EdgeSE3Z)
G2O_REGISTER_TYPE(EDGE_SE3_SE3, EdgeSE3SE3)
G2O_REGISTER_TYPE(EDGE_SE3_PRIORVEC, EdgeSE3PriorVec)
G2O_REGISTER_TYPE(EDGE_SE3_PRIORQUAT, EdgeSE3PriorQuat)
G2O_REGISTER_TYPE(EDGE_PLANE_PRIOR_NORMAL, EdgePlanePriorNormal)
G2O_REGISTER_TYPE(EDGE_PLANE_PRIOR_DISTANCE, EdgePlanePriorDistance)
G2O_REGISTER_TYPE(EDGE_PLANE_IDENTITY, EdgePlaneIdentity)
G2O_REGISTER_TYPE(EDGE_PLANE_PARALLEL, EdgePlaneParallel)
G2O_REGISTER_TYPE(EDGE_PLANE_PAERPENDICULAR, EdgePlanePerpendicular)
G2O_REGISTER_TYPE(EDGE_SE3_GTUTM, EdgeSE3GtUTM)
// G2O_REGISTER_TYPE(EDGE_IMU, EdgeIMU)
G2O_REGISTER_TYPE(VERTEX_POSE, VertexPose)
G2O_REGISTER_TYPE(VERTEX_VELOCITY, VertexVelocity)
G2O_REGISTER_TYPE(VERTEX_GYRO_BIAS, VertexGyroBias)
G2O_REGISTER_TYPE(VERTEX_ACC_BIAS, VertexAccBias)
G2O_REGISTER_TYPE(EDGE_SE3_INTERIAL, EdgeSE3Interial)
G2O_REGISTER_TYPE(EDGE_RADAR_3D_VELOCITY, EdgeRadar3DVelocity)
G2O_REGISTER_TYPE(EDGE_PRIOR_POSE_NAV_STATE, EdgePriorPoseNavState)
G2O_REGISTER_TYPE(EDGE_GYRO_RW, EdgeGyroRW)
G2O_REGISTER_TYPE(EDGE_ACC_RW, EdgeAccRW)
G2O_REGISTER_TYPE(EDGE_POSE, EdgePose)


}  // namespace g2o

namespace radar_graph_slam {
    using Mat15d = Eigen::Matrix<double, 15, 15>;
    using Vec15d = Eigen::Matrix<double, 15, 15>;


/**
 * @brief constructor
 */
GraphSLAM::GraphSLAM(const std::string& solver_type) {
  graph.reset(new g2o::SparseOptimizer());
  g2o::SparseOptimizer* graph = dynamic_cast<g2o::SparseOptimizer*>(this->graph.get());

  std::cout << "construct solver: " << solver_type << std::endl;
  g2o::OptimizationAlgorithmFactory* solver_factory = g2o::OptimizationAlgorithmFactory::instance();
  g2o::OptimizationAlgorithmProperty solver_property;
  g2o::OptimizationAlgorithm* solver = solver_factory->construct(solver_type, solver_property);
  graph->setAlgorithm(solver);

  if(!graph->solver()) {
    std::cerr << std::endl;
    std::cerr << "error : failed to allocate solver!!" << std::endl;
    solver_factory->listSolvers(std::cerr);
    std::cerr << "-------------" << std::endl;
    std::cin.ignore(1);
    return;
  }
  std::cout << "done" << std::endl;

  robust_kernel_factory = g2o::RobustKernelFactory::instance();
}

/**
 * @brief destructor
 */
GraphSLAM::~GraphSLAM() {
  graph.reset();
}

void GraphSLAM::set_solver(const std::string& solver_type) {
  g2o::SparseOptimizer* graph = dynamic_cast<g2o::SparseOptimizer*>(this->graph.get());

  std::cout << "construct solver: " << solver_type << std::endl;
  g2o::OptimizationAlgorithmFactory* solver_factory = g2o::OptimizationAlgorithmFactory::instance();
  g2o::OptimizationAlgorithmProperty solver_property;
  g2o::OptimizationAlgorithm* solver = solver_factory->construct(solver_type, solver_property);
  graph->setAlgorithm(solver);

  if(!graph->solver()) {
    std::cerr << std::endl;
    std::cerr << "error : failed to allocate solver!!" << std::endl;
    solver_factory->listSolvers(std::cerr);
    std::cerr << "-------------" << std::endl;
    std::cin.ignore(1);
    return;
  }
  std::cout << "done" << std::endl;
}

int GraphSLAM::num_vertices() const {
  return graph->vertices().size();
}
int GraphSLAM::num_edges() const {
  return graph->edges().size();
}

g2o::VertexSE3* GraphSLAM::add_se3_node(const Eigen::Isometry3d& pose) {
  g2o::VertexSE3* vertex(new g2o::VertexSE3());
  vertex->setId(static_cast<int>(graph->vertices().size()));
  vertex->setEstimate(pose);
  graph->addVertex(vertex);
  return vertex;
}

g2o::VertexPlane* GraphSLAM::add_plane_node(const Eigen::Vector4d& plane_coeffs) {
  g2o::VertexPlane* vertex(new g2o::VertexPlane());
  vertex->setId(static_cast<int>(graph->vertices().size()));
  vertex->setEstimate(plane_coeffs);
  graph->addVertex(vertex);

  return vertex;
}

g2o::VertexPointXYZ* GraphSLAM::add_point_xyz_node(const Eigen::Vector3d& xyz) {
  g2o::VertexPointXYZ* vertex(new g2o::VertexPointXYZ());
  vertex->setId(static_cast<int>(graph->vertices().size()));
  vertex->setEstimate(xyz);
  graph->addVertex(vertex);

  return vertex;
}
g2o::VertexVelocity* GraphSLAM::add_velocity_node(const Eigen::Vector3d& vel) {
  g2o::VertexVelocity* vertex(new g2o::VertexVelocity());
  vertex->setId(static_cast<int>(graph->vertices().size()));
  vertex->setEstimate(vel);
  graph->addVertex(vertex);

  return vertex;
}

g2o::VertexGyroBias* GraphSLAM::add_gyro_bais_node(const Eigen::Vector3d& gyro_bias){
  g2o::VertexGyroBias* vertex(new g2o::VertexGyroBias());
  vertex->setId(static_cast<int>(graph->vertices().size()));
  vertex->setEstimate(gyro_bias);
  graph->addVertex(vertex);
  return vertex;
}
g2o::VertexAccBias* GraphSLAM::add_acc_bias_node(const Eigen::Vector3d& acc_bias){
  g2o::VertexAccBias* vertex(new g2o::VertexAccBias());
  vertex->setId(static_cast<int>(graph->vertices().size()));
  vertex->setEstimate(acc_bias);
  graph->addVertex(vertex);
  return vertex;
}
g2o::EdgeSE3* GraphSLAM::add_se3_edge(g2o::VertexSE3* v1, g2o::VertexSE3* v2, const Eigen::Isometry3d& relative_pose, const Eigen::MatrixXd& information_matrix) {
  g2o::EdgeSE3* edge(new g2o::EdgeSE3());
  edge->setMeasurement(relative_pose);
  edge->setInformation(information_matrix);
  edge->vertices()[0] = v1;
  edge->vertices()[1] = v2;
  graph->addEdge(edge);

  return edge;
}

g2o::EdgeSE3Prior* GraphSLAM::add_se3_prior_edge(g2o::VertexSE3* v_se3, const Eigen::Isometry3d& pose, const Eigen::MatrixXd& information_matrix) {
  g2o::EdgeSE3Prior* edge(new g2o::EdgeSE3Prior());
  edge->setMeasurement(pose);
  edge->setInformation(information_matrix);
  edge->vertices()[0] = v_se3;
  graph->addEdge(edge);

  return edge;
}

// Information matrix should be 6*6
g2o::EdgeSE3SE3* GraphSLAM::add_se3_se3_edge(g2o::VertexSE3* v1, g2o::VertexSE3* v2, const g2o::SE3Quat& relative_pose, const Eigen::MatrixXd& information_matrix) {
  g2o::EdgeSE3SE3* edge(new g2o::EdgeSE3SE3());
  edge->setMeasurement(relative_pose);
  edge->setInformation(information_matrix);
  edge->vertices()[0] = v1;
  edge->vertices()[1] = v2;
  graph->addEdge(edge);

  return edge;
}

g2o::EdgeSE3Plane* GraphSLAM::add_se3_plane_edge(g2o::VertexSE3* v_se3, g2o::VertexPlane* v_plane, const Eigen::Vector4d& plane_coeffs, const Eigen::MatrixXd& information_matrix) {
  g2o::EdgeSE3Plane* edge(new g2o::EdgeSE3Plane());
  edge->setMeasurement(plane_coeffs);
  edge->setInformation(information_matrix);
  edge->vertices()[0] = v_se3;
  edge->vertices()[1] = v_plane;
  graph->addEdge(edge);

  return edge;
}

g2o::EdgeSE3PointXYZ* GraphSLAM::add_se3_point_xyz_edge(g2o::VertexSE3* v_se3, g2o::VertexPointXYZ* v_xyz, const Eigen::Vector3d& xyz, const Eigen::MatrixXd& information_matrix) {
  g2o::EdgeSE3PointXYZ* edge(new g2o::EdgeSE3PointXYZ());
  edge->setMeasurement(xyz);
  edge->setInformation(information_matrix);
  edge->vertices()[0] = v_se3;
  edge->vertices()[1] = v_xyz;
  graph->addEdge(edge);

  return edge;
}

g2o::EdgeSE3GtUTM* GraphSLAM::add_se3_gt_utm_edge(g2o::VertexSE3* v_se3, const Eigen::Vector3d& gt_xyz, const Eigen::Vector3d& utm_xyz, const Eigen::MatrixXd& information_matrix) {
  g2o::EdgeSE3GtUTM* edge(new g2o::EdgeSE3GtUTM());
  edge->setMeasurement(gt_xyz);
  edge->setMeasurement_utm(utm_xyz);
  edge->setInformation(information_matrix);
  edge->vertices()[0] = v_se3;
  graph->addEdge(edge);
  return edge;
}

g2o::EdgePlanePriorNormal* GraphSLAM::add_plane_normal_prior_edge(g2o::VertexPlane* v, const Eigen::Vector3d& normal, const Eigen::MatrixXd& information_matrix) {
  g2o::EdgePlanePriorNormal* edge(new g2o::EdgePlanePriorNormal());
  edge->setMeasurement(normal);
  edge->setInformation(information_matrix);
  edge->vertices()[0] = v;
  graph->addEdge(edge);

  return edge;
}

g2o::EdgePlanePriorDistance* GraphSLAM::add_plane_distance_prior_edge(g2o::VertexPlane* v, double distance, const Eigen::MatrixXd& information_matrix) {
  g2o::EdgePlanePriorDistance* edge(new g2o::EdgePlanePriorDistance());
  edge->setMeasurement(distance);
  edge->setInformation(information_matrix);
  edge->vertices()[0] = v;
  graph->addEdge(edge);

  return edge;
}

g2o::EdgeSE3PriorXY* GraphSLAM::add_se3_prior_xy_edge(g2o::VertexSE3* v_se3, const Eigen::Vector2d& xy, const Eigen::MatrixXd& information_matrix) {
  g2o::EdgeSE3PriorXY* edge(new g2o::EdgeSE3PriorXY());
  edge->setMeasurement(xy);
  edge->setInformation(information_matrix);
  edge->vertices()[0] = v_se3;
  graph->addEdge(edge);

  return edge;
}

g2o::EdgeSE3PriorXYZ* GraphSLAM::add_se3_prior_xyz_edge(g2o::VertexSE3* v_se3, const Eigen::Vector3d& xyz, const Eigen::MatrixXd& information_matrix) {
  g2o::EdgeSE3PriorXYZ* edge(new g2o::EdgeSE3PriorXYZ());
  edge->setMeasurement(xyz);
  edge->setInformation(information_matrix);
  edge->vertices()[0] = v_se3;
  graph->addEdge(edge);

  return edge;
}

g2o::EdgeSE3PriorZ* GraphSLAM::add_se3_prior_z_edge(g2o::VertexSE3* v_se3, const Eigen::Vector1d& z, const Eigen::MatrixXd& information_matrix) {
  g2o::EdgeSE3PriorZ* edge(new g2o::EdgeSE3PriorZ());
  edge->setMeasurement(z);
  edge->setInformation(information_matrix);
  edge->vertices()[0] = v_se3;
  graph->addEdge(edge);

  return edge;
}

g2o::EdgeSE3Z* GraphSLAM::add_se3_z_edge(g2o::VertexSE3* v1, g2o::VertexSE3* v2, const Eigen::Vector1d& z, const Eigen::MatrixXd& information_matrix) {
  g2o::EdgeSE3Z* edge(new g2o::EdgeSE3Z());
  edge->setMeasurement(z);
  edge->setInformation(information_matrix);
  edge->vertices()[0] = v1;
  edge->vertices()[1] = v2;
  graph->addEdge(edge);

  return edge;
}

g2o::EdgeSE3PriorVec* GraphSLAM::add_se3_prior_vec_edge(g2o::VertexSE3* v_se3, const Eigen::Vector3d& direction, const Eigen::Vector3d& measurement, const Eigen::MatrixXd& information_matrix) {
  Eigen::Matrix<double, 6, 1> m;
  m.head<3>() = direction;
  m.tail<3>() = measurement;

  g2o::EdgeSE3PriorVec* edge(new g2o::EdgeSE3PriorVec());
  edge->setMeasurement(m);
  edge->setInformation(information_matrix);
  edge->vertices()[0] = v_se3;
  graph->addEdge(edge);

  return edge;
}

g2o::EdgeSE3PriorQuat* GraphSLAM::add_se3_prior_quat_edge(g2o::VertexSE3* v_se3, const Eigen::Quaterniond& quat, const Eigen::MatrixXd& information_matrix) {
  g2o::EdgeSE3PriorQuat* edge(new g2o::EdgeSE3PriorQuat());
  edge->setMeasurement(quat);
  edge->setInformation(information_matrix);
  edge->vertices()[0] = v_se3;
  graph->addEdge(edge);

  return edge;
}

g2o::EdgePlane* GraphSLAM::add_plane_edge(g2o::VertexPlane* v_plane1, g2o::VertexPlane* v_plane2, const Eigen::Vector4d& measurement, const Eigen::Matrix4d& information) {
  g2o::EdgePlane* edge(new g2o::EdgePlane());
  edge->setMeasurement(measurement);
  edge->setInformation(information);
  edge->vertices()[0] = v_plane1;
  edge->vertices()[1] = v_plane2;
  graph->addEdge(edge);

  return edge;
}

g2o::EdgePlaneIdentity* GraphSLAM::add_plane_identity_edge(g2o::VertexPlane* v_plane1, g2o::VertexPlane* v_plane2, const Eigen::Vector4d& measurement, const Eigen::Matrix4d& information) {
  g2o::EdgePlaneIdentity* edge(new g2o::EdgePlaneIdentity());
  edge->setMeasurement(measurement);
  edge->setInformation(information);
  edge->vertices()[0] = v_plane1;
  edge->vertices()[1] = v_plane2;
  graph->addEdge(edge);

  return edge;
}

g2o::EdgePlaneParallel* GraphSLAM::add_plane_parallel_edge(g2o::VertexPlane* v_plane1, g2o::VertexPlane* v_plane2, const Eigen::Vector3d& measurement, const Eigen::Matrix3d& information) {
  g2o::EdgePlaneParallel* edge(new g2o::EdgePlaneParallel());
  edge->setMeasurement(measurement);
  edge->setInformation(information);
  edge->vertices()[0] = v_plane1;
  edge->vertices()[1] = v_plane2;
  graph->addEdge(edge);

  return edge;
}

g2o::EdgePlanePerpendicular* GraphSLAM::add_plane_perpendicular_edge(g2o::VertexPlane* v_plane1, g2o::VertexPlane* v_plane2, const Eigen::Vector3d& measurement, const Eigen::MatrixXd& information) {
  g2o::EdgePlanePerpendicular* edge(new g2o::EdgePlanePerpendicular());
  edge->setMeasurement(measurement);
  edge->setInformation(information);
  edge->vertices()[0] = v_plane1;
  edge->vertices()[1] = v_plane2;
  graph->addEdge(edge);

  return edge;
}
g2o::EdgeSE3Interial* GraphSLAM::add_se3_interial_edge(g2o::VertexSE3* v1,g2o::VertexVelocity* v2, g2o::VertexGyroBias* v3, g2o::VertexAccBias* v4, g2o::VertexSE3* v5, g2o::VertexVelocity* v6,
                                              std::shared_ptr<radar_graph_slam::IMUPreintegrator> preinteg, double weight)
{
  g2o::EdgeSE3Interial* edge = new g2o::EdgeSE3Interial(preinteg,weight);
  edge->vertices()[0] = v1;
  edge->vertices()[1] = v2;
  edge->vertices()[2] = v3;
  edge->vertices()[3] = v4;
  edge->vertices()[4] = v5;
  edge->vertices()[5] = v6;
  graph->addEdge(edge);
  return edge;
}
g2o::EdgeRadar3DVelocity* GraphSLAM::add_3d_velocity_edge(g2o::VertexVelocity* v1,const Vec3d& measurement,const Eigen::MatrixXd& information)
{
  g2o::EdgeRadar3DVelocity* edge = new g2o::EdgeRadar3DVelocity();
  edge->setMeasurement(measurement);
  edge->setInformation(information);
  edge->vertices()[0] = v1;
  graph->addEdge(edge);
  return edge;
}
g2o::EdgePriorPoseNavState* GraphSLAM::add_prior_state_edge(g2o::VertexSE3* v1, g2o::VertexVelocity* v2, g2o::VertexGyroBias* v3, g2o::VertexAccBias* v4,
                                              const NavStated& state,const Mat15d& information)
{
  g2o::EdgePriorPoseNavState* edge = new g2o::EdgePriorPoseNavState(state,information);
  // edge->setMeasurement(p,v,bg,ba);
  edge->setInformation(information);
  edge->vertices()[0] = v1;
  edge->vertices()[1] = v2;
  edge->vertices()[2] = v3;
  edge->vertices()[3] = v4;
  graph->addEdge(edge);
  return edge;
}
g2o::EdgeGyroRW* GraphSLAM::add_gyro_rw_edge(g2o::VertexGyroBias* v1,g2o::VertexGyroBias* v2,const Eigen::MatrixXd& information)
{
  g2o::EdgeGyroRW* edge = new g2o::EdgeGyroRW();
  edge->setInformation(information);
  edge->vertices()[0] = v1;
  edge->vertices()[1] = v2;
  graph->addEdge(edge);
  return edge;
}
g2o::EdgeAccRW* GraphSLAM::add_acc_rw_edge(g2o::VertexAccBias* v1,g2o::VertexAccBias* v2,const Eigen::MatrixXd& information)
{
  g2o::EdgeAccRW* edge = new g2o::EdgeAccRW();
  edge->setInformation(information);
  edge->vertices()[0] = v1;
  edge->vertices()[1] = v2;
  graph->addEdge(edge);
  return edge;
}
g2o::EdgePose* GraphSLAM::add_pose_edge(g2o::VertexSE3* v1, const SE3& pose, const Eigen::MatrixXd& information_matrix)
{
  g2o::EdgePose* edge = new g2o::EdgePose(v1,pose);
  edge->setInformation(information_matrix);
  edge->vertices()[0] = v1;
  graph->addEdge(edge);
  return edge;
}

// g2o::EdgeIMU* GraphSLAM::add_imu_multi_edge(g2o::VertexPose* v0_pose, g2o::VertexVelocity* v0_vel,
//                                             g2o::VertexGyroBias* v0_bg, g2o::VertexAccBias* v0_ba,
//                                             g2o::VertexPose* v1_pose,g2o::VertexVelocity* v1_vel, 
//                                             std::shared_ptr<radar_graph_slam::IMUPreintegrator> preinteg, const Eigen::Vector3d& gravity)
// {
//     // auto edge = new g2o::EdgeIMU(preinteg,gravity);
//     // edge->setVertex(0, v0_pose);
//     // edge->setVertex(1, v0_vel);
//     // edge->setVertex(2, v0_bg);
//     // edge->setVertex(3, v0_ba);
//     // edge->setVertex(4, v1_pose);
//     // edge->setVertex(5, v1_vel);

//     g2o::EdgeIMU* edge = new g2o::EdgeIMU(preinteg,gravity);
//     edge->setVertex(0, v0_pose);
//     edge->setVertex(1, v0_vel);
//     edge->setVertex(2, v0_bg);
//     edge->setVertex(3, v0_ba);
//     edge->setVertex(4, v1_pose);
//     edge->setVertex(5, v1_vel);
//     // auto *rk = new g2o::RobustKernelHuber();
//     // rk->setDelta(200.0);
//     // edge->setRobustKernel(rk);
//     graph->addEdge(edge);
//     return edge;
// }


void GraphSLAM::add_robust_kernel(g2o::HyperGraph::Edge* edge, const std::string& kernel_type, double kernel_size) {
  if(kernel_type == "NONE") {
    return;
  }

  g2o::RobustKernel* kernel = robust_kernel_factory->construct(kernel_type);
  if(kernel == nullptr) {
    std::cerr << "warning : invalid robust kernel type: " << kernel_type << std::endl;
    return;
  }

  kernel->setDelta(kernel_size);

  g2o::OptimizableGraph::Edge* edge_ = dynamic_cast<g2o::OptimizableGraph::Edge*>(edge);
  edge_->setRobustKernel(kernel);
}

int GraphSLAM::optimize(int num_iterations) {
  g2o::SparseOptimizer* graph = dynamic_cast<g2o::SparseOptimizer*>(this->graph.get());
  // why should be more than 10?
  if(graph->edges().size() < 1) {
    return -1;
  }

  std::cout << std::endl;
  std::cout << "--- pose graph optimization ---" << std::endl;
  std::cout << "nodes: " << graph->vertices().size() << "   edges: " << graph->edges().size() << std::endl;
  std::cout << "optimizing... " << std::flush;

  std::cout << "init" << std::endl;
  graph->initializeOptimization();
  graph->setVerbose(false);  // Open/Close debug output

  std::cout << "chi2" << std::endl;
  double chi2 = graph->chi2();

  std::cout << "optimizing!!" << std::endl;
  auto t1 = ros::WallTime::now();
  int iterations = graph->optimize(num_iterations);

  auto t2 = ros::WallTime::now();
  std::cout << "done" << std::endl;
  std::cout << "iterations: " << iterations << " / " << num_iterations << std::endl;
  std::cout << "chi2: (before)" << chi2 << " -> (after)" << graph->chi2() << std::endl;
  std::cout << "time: " << boost::format("%.3f") % (t2 - t1).toSec() << "[sec]" << std::endl;

  return iterations;
}

void GraphSLAM::save(const std::string& filename) {
  g2o::SparseOptimizer* graph = dynamic_cast<g2o::SparseOptimizer*>(this->graph.get());

  std::ofstream ofs(filename);
  graph->save(ofs);

  g2o::save_robust_kernels(filename + ".kernels", graph);
}

bool GraphSLAM::load(const std::string& filename) {
  std::cout << "loading pose graph..." << std::endl;
  g2o::SparseOptimizer* graph = dynamic_cast<g2o::SparseOptimizer*>(this->graph.get());

  std::ifstream ifs(filename);
  if(!graph->load(ifs)) {
    return false;
  }

  std::cout << "nodes  : " << graph->vertices().size() << std::endl;
  std::cout << "edges  : " << graph->edges().size() << std::endl;

  if(!g2o::load_robust_kernels(filename + ".kernels", graph)) {
    return false;
  }

  return true;
}

}  // namespace radar_graph_slam
