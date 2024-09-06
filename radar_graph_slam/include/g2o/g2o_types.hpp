
#ifndef SLAM_IN_AUTO_DRIVING_COMMON_G2O_TYPES_H
#define SLAM_IN_AUTO_DRIVING_COMMON_G2O_TYPES_H

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/robust_kernel.h>

#include "radar_graph_slam/nav_state.hpp"


#include "g2o/core/robust_kernel_impl.h"

#include <glog/logging.h>
#include "sophus/se2.hpp"
#include "sophus/se3.hpp"

namespace g2o {
    using SE3 = Sophus::SE3d;
    using Vec3d = Eigen::Vector3d;
    using SO3 = Sophus::SO3d;
    using Quatd = Eigen::Quaterniond;
    using Quatf = Eigen::Quaternionf;
    using Mat3d = Eigen::Matrix3d;
    using Mat6d = Eigen::Matrix<double, 6, 6>;
    using Mat15d = Eigen::Matrix<double, 15, 15>;
    using Vec15d = Eigen::Matrix<double, 15, 15>;
    using NavStated = radar_graph_slam::NavStated;


class VertexPose : public g2o::BaseVertex<6, SE3> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPose() {}

    bool read(std::istream& is) override {
        double data[7];
        for (int i = 0; i < 7; i++) {
            is >> data[i];
        }
        setEstimate(SE3(Quatd(data[6], data[3], data[4], data[5]), Vec3d(data[0], data[1], data[2])));
        return true;
    }

    bool write(std::ostream& os) const override {
        os << "VERTEX_SE3:QUAT ";
        os << id() << " ";
        Quatd q = _estimate.unit_quaternion();
        os << _estimate.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << std::endl;
        return true;
    }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double* update_) {
        _estimate.so3() = _estimate.so3() * SO3::exp(Eigen::Map<const Vec3d>(&update_[0]));  // 
        _estimate.translation() += Eigen::Map<const Vec3d>(&update_[3]);                     // 
        updateCache();
    }
};

/**
 * @brief 
 * 
 */
class VertexVelocity : public g2o::BaseVertex<3, Vec3d> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexVelocity() {}

    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    virtual void setToOriginImpl() { _estimate.setZero(); }

    virtual void oplusImpl(const double* update_) { _estimate += Eigen::Map<const Vec3d>(update_); }
};

/**

 */
class VertexGyroBias : public VertexVelocity {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexGyroBias() {}
};

/**
 */
class VertexAccBias : public VertexVelocity {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexAccBias() {}
};

/**
 * 
 */
class EdgeGyroRW : public g2o::BaseBinaryEdge<3, Vec3d, VertexGyroBias, VertexGyroBias> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeGyroRW() {}

    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    void computeError() {
        const auto* VG1 = dynamic_cast<const VertexGyroBias*>(_vertices[0]);
        const auto* VG2 = dynamic_cast<const VertexGyroBias*>(_vertices[1]);
        _error = VG2->estimate() - VG1->estimate();
    }

    virtual void linearizeOplus() {
        _jacobianOplusXi = -Mat3d::Identity();
        _jacobianOplusXj.setIdentity();
    }

    Eigen::Matrix<double, 6, 6> GetHessian() {
        linearizeOplus();
        Eigen::Matrix<double, 3, 6> J;
        J.block<3, 3>(0, 0) = _jacobianOplusXi;
        J.block<3, 3>(0, 3) = _jacobianOplusXj;
        return J.transpose() * information() * J;
    }
};

/**
 * 
 */
class EdgeAccRW : public g2o::BaseBinaryEdge<3, Vec3d, VertexAccBias, VertexAccBias> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeAccRW() {}

    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    void computeError() {
        const auto* VA1 = dynamic_cast<const VertexAccBias*>(_vertices[0]);
        const auto* VA2 = dynamic_cast<const VertexAccBias*>(_vertices[1]);
        _error = VA2->estimate() - VA1->estimate();
    }

    virtual void linearizeOplus() {
        _jacobianOplusXi = -Mat3d::Identity();
        _jacobianOplusXj.setIdentity();
    }

    Eigen::Matrix<double, 6, 6> GetHessian() {
        linearizeOplus();
        Eigen::Matrix<double, 3, 6> J;
        J.block<3, 3>(0, 0) = _jacobianOplusXi;
        J.block<3, 3>(0, 3) = _jacobianOplusXj;
        return J.transpose() * information() * J;
    }
};

/**
 */
class EdgePriorPoseNavState : public g2o::BaseMultiEdge<15, Vec15d> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgePriorPoseNavState() = default;
    EdgePriorPoseNavState(const NavStated& state, const Mat15d& info) 
    {
        resize(4);
        state_ = state;
        setInformation(info);
    }

    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    void computeError() override
    {    
        auto* vp = dynamic_cast<const g2o::VertexSE3*>(_vertices[0]);
        auto* vv = dynamic_cast<const VertexVelocity*>(_vertices[1]);
        auto* vg = dynamic_cast<const VertexGyroBias*>(_vertices[2]);
        auto* va = dynamic_cast<const VertexAccBias*>(_vertices[3]);
        const SO3 vp_1(vp->estimate().linear());


        const Vec3d er = SO3(state_.R_.matrix().transpose() * vp_1.matrix()).log();
        const Vec3d ep = vp->estimate().translation() - state_.p_;
        const Vec3d ev = vv->estimate() - state_.v_;
        const Vec3d ebg = vg->estimate() - state_.bg_;
        const Vec3d eba = va->estimate() - state_.ba_;

        _error << er, ep, ev, ebg, eba;

    }
    // void setMeasurement(const Vec15d& m) override {
    //     _measurement = m;
    // }
    virtual void linearizeOplus() override
    {
        const auto* vp = dynamic_cast<const g2o::VertexSE3*>(_vertices[0]);
        const SO3 vp_1(vp->estimate().linear());

        const Vec3d er = SO3(state_.R_.matrix().transpose() * vp_1.matrix()).log();

        _jacobianOplus[0].setZero();
        auto omega = -er;
        auto theta = omega.norm();
        if (theta < 1e-6) {
            _jacobianOplus[0].block<3, 3>(0, 0) = SO3::Transformation::Identity();  // dr/dr
        } else {
            Vec3d a = omega;
            a.normalize();
            double cot_half_theta = cos(0.5 * theta) / sin(0.5 * theta);
            Mat3d Jr_inv = 0.5 * theta * cot_half_theta *SO3::Transformation::Identity() + (1 - 0.5 * theta * cot_half_theta) * a * a.transpose() - 0.5 * theta * SO3::hat(a);
            _jacobianOplus[0].block<3, 3>(0, 0) = Jr_inv;
        }
        _jacobianOplus[0].block<3, 3>(3, 3) = Mat3d::Identity();  // dp/dp
        _jacobianOplus[1].setZero();
        _jacobianOplus[1].block<3, 3>(6, 0) = Mat3d::Identity();  // dv/dv
        _jacobianOplus[2].setZero();
        _jacobianOplus[2].block<3, 3>(9, 0) = Mat3d::Identity();  // dbg/dbg
        _jacobianOplus[3].setZero();
        _jacobianOplus[3].block<3, 3>(12, 0) = Mat3d::Identity();  // dba/dba
    }

    Eigen::Matrix<double, 15, 15> GetHessian() {
        linearizeOplus();
        Eigen::Matrix<double, 15, 15> J;
        J.block<15, 6>(0, 0) = _jacobianOplus[0];
        J.block<15, 3>(0, 6) = _jacobianOplus[1];
        J.block<15, 3>(0, 9) = _jacobianOplus[2];
        J.block<15, 3>(0, 12) = _jacobianOplus[3];
        return J.transpose() * information() * J;
    }

    NavStated state_;
};

/**
 */
class EdgePose : public g2o::BaseUnaryEdge<6, SE3, g2o::VertexSE3> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgePose() = default;
    EdgePose(g2o::VertexSE3* v, const SE3& obs) {
        setVertex(0, v);
        setMeasurement(obs);
    }

    void computeError() override {
        g2o::VertexSE3* v = (g2o::VertexSE3*)_vertices[0];
        const SO3 v_1(v->estimate().linear());

        _error.head<3>() = (_measurement.so3().inverse() * v_1).log();
        _error.tail<3>() = v->estimate().translation() - _measurement.translation();
    };

    void linearizeOplus() override {
        g2o::VertexSE3* v = (g2o::VertexSE3*)_vertices[0];
        const SO3 v_1(v->estimate().linear());

        // jacobian 6x6
        _jacobianOplusXi.setZero();
        //jr inv
        Mat3d invJr;
        auto omega_ = _measurement.so3().inverse() * v_1;
        auto omega = -omega_.log();
        auto theta = omega.norm();
        if(theta < 1e-6)
        {
            invJr = SO3::Transformation::Identity();
        }
        else
        {
            Vec3d a = omega;
            a.normalize();
            double cot_half_theta = cos(0.5 * theta) / sin(0.5 * theta);
            invJr = 0.5 * theta * cot_half_theta * SO3::Transformation ::Identity() +
                    (1 - 0.5 * theta * cot_half_theta) * a * a.transpose() - 0.5 * theta * SO3::hat(a);
        }
        _jacobianOplusXi.block<3, 3>(0, 0) = invJr;  // dR/dR
        _jacobianOplusXi.block<3, 3>(3, 3) = Mat3d::Identity();                                              // dp/dp
    }

    Mat6d GetHessian() {
        linearizeOplus();
        return _jacobianOplusXi.transpose() * information() * _jacobianOplusXi;
    }

    virtual bool read(std::istream& in) { return true; }
    virtual bool write(std::ostream& out) const { return true; }

   private:
};

/**
 */
class EdgeGNSSTransOnly : public g2o::BaseUnaryEdge<3, Vec3d, VertexPose> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * @param v
     * @param obs
     */
    EdgeGNSSTransOnly(VertexPose* v, const Vec3d& obs, const SE3& TBG) : TBG_(TBG) {
        setVertex(0, v);
        setMeasurement(obs);
    }

    void computeError() override {
        VertexPose* v = (VertexPose*)_vertices[0];
        _error = (v->estimate() * TBG_).translation() - _measurement;
    };

    // void linearizeOplus() override {
    //     VertexPose* v = (VertexPose*)_vertices[0];
    //     // jacobian 6x6
    //     _jacobianOplusXi.setZero();
    //     _jacobianOplusXi.block<3, 3>(0, 0) = (_measurement.so3().inverse() * v->estimate().so3()).jr_inv();  // dR/dR
    //     _jacobianOplusXi.block<3, 3>(3, 3) = Mat3d::Identity();                                              // dp/dp
    // }

    virtual bool read(std::istream& in) { return true; }
    virtual bool write(std::ostream& out) const { return true; }

   private:
    SE3 TBG_;
};

/**
 */
class EdgeRelativeMotion : public g2o::BaseBinaryEdge<6, SE3, VertexPose, VertexPose> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeRelativeMotion() = default;
    EdgeRelativeMotion(VertexPose* v1, VertexPose* v2, const SE3& obs) {
        setVertex(0, v1);
        setVertex(1, v2);
        setMeasurement(obs);
    }

    void computeError() override {
        VertexPose* v1 = (VertexPose*)_vertices[0];
        VertexPose* v2 = (VertexPose*)_vertices[1];
        SE3 T12 = v1->estimate().inverse() * v2->estimate();
        _error = (_measurement.inverse() * v1->estimate().inverse() * v2->estimate()).log();
    };

    virtual bool read(std::istream& is) override {
        double data[7];
        for (int i = 0; i < 7; i++) {
            is >> data[i];
        }
        Quatd q(data[6], data[3], data[4], data[5]);
        q.normalize();
        setMeasurement(SE3(q, Vec3d(data[0], data[1], data[2])));
        for (int i = 0; i < information().rows() && is.good(); i++) {
            for (int j = i; j < information().cols() && is.good(); j++) {
                is >> information()(i, j);
                if (i != j) information()(j, i) = information()(i, j);
            }
        }
        return true;
    }

    virtual bool write(std::ostream& os) const override {
        os << "EDGE_SE3:QUAT ";
        auto* v1 = static_cast<VertexPose*>(_vertices[0]);
        auto* v2 = static_cast<VertexPose*>(_vertices[1]);
        os << v1->id() << " " << v2->id() << " ";
        SE3 m = _measurement;
        Eigen::Quaterniond q = m.unit_quaternion();
        os << m.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << " ";

        // information matrix
        for (int i = 0; i < information().rows(); i++) {
            for (int j = i; j < information().cols(); j++) {
                os << information()(i, j) << " ";
            }
        }
        os << std::endl;
        return true;
    }

   private:
};

/**
 */
class EdgeEncoder3D : public g2o::BaseUnaryEdge<3, Vec3d, VertexVelocity> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeEncoder3D() = default;

    /**
     * 
     * @param v0
     * @param speed
     */
    EdgeEncoder3D(VertexVelocity* v0, const Vec3d& speed) {
        setVertex(0, v0);
        setMeasurement(speed);
    }

    void computeError() override {
        VertexVelocity* v0 = (VertexVelocity*)_vertices[0];
        _error = v0->estimate() - _measurement;
    }

    void linearizeOplus() override { _jacobianOplusXi.setIdentity(); }

    virtual bool read(std::istream& in) { return true; }
    virtual bool write(std::ostream& out) const { return true; }
};

/**
 */
class EdgeNDT : public g2o::BaseUnaryEdge<3, Vec3d, VertexPose> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeNDT() = default;

   
    using QueryVoxelFunc = std::function<bool(const Vec3d& query_pt, Vec3d& mu, Mat3d& info)>;

    EdgeNDT(VertexPose* v0, const Vec3d& pt, QueryVoxelFunc func) {
        setVertex(0, v0);
        pt_ = pt;
        query_ = func;

        Vec3d q = v0->estimate().so3() * pt_ + v0->estimate().translation();
        if (query_(q, mu_, info_)) {
            setInformation(info_);
            valid_ = true;
        } else {
            valid_ = false;
        }
    }

    bool IsValid() const { return valid_; }

    Mat6d GetHessian() {
        linearizeOplus();
        return _jacobianOplusXi.transpose() * info_ * _jacobianOplusXi;
    }

    void computeError() override {
        VertexPose* v0 = (VertexPose*)_vertices[0];
        Vec3d q = v0->estimate().so3() * pt_ + v0->estimate().translation();

        if (query_(q, mu_, info_)) {
            _error = q - mu_;
            setInformation(info_);
            valid_ = true;
        } else {
            valid_ = false;
            _error.setZero();
            setLevel(1);
        }
    }


    void linearizeOplus() override {
        if (valid_) {
            VertexPose* v0 = (VertexPose*)_vertices[0];
            SO3 R = v0->estimate().so3();

            _jacobianOplusXi.setZero();
            _jacobianOplusXi.block<3, 3>(0, 0) = -R.matrix() * SO3::hat(pt_);  // 
            _jacobianOplusXi.block<3, 3>(0, 3) = Mat3d::Identity();            // 
        } else {
            _jacobianOplusXi.setZero();
        }
    }

    virtual bool read(std::istream& in) { return true; }
    virtual bool write(std::ostream& out) const { return true; }

   private:
    QueryVoxelFunc query_;
    Vec3d pt_ = Vec3d::Zero();
    Vec3d mu_ = Vec3d::Zero();
    Mat3d info_ = Mat3d::Identity();
    bool valid_ = false;
};
}  // namespace sad

#endif  // SLAM_IN_AUTO_DRIVING_G2O_TYPES_H
