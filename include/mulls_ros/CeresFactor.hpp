// Author:   Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk

#ifndef CERES_FACTOR_HPP
#define CERES_FACTOR_HPP

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/cubic_interpolation.h>

#include "assert.h"

#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <vikit/pinhole_camera.h>
#include <vikit/vision.h>
#include <vikit/math_utils.h>
#include <vikit/patch_score.h>

#include <ceres/local_parameterization.h>
#include "Plane3d.hpp"

#include "Common.h"

class PoseSE3Parameterization : public ceres::LocalParameterization
{
public:
	PoseSE3Parameterization() {}
	virtual ~PoseSE3Parameterization() {}
	virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const
	{
		Eigen::Map<const Eigen::Vector3d> trans(x + 4);

		Eigen::Quaterniond delta_q;
		Eigen::Vector3d delta_t;
		ORB_SLAM2::getTransformFromSe3(Eigen::Map<const Eigen::Matrix<double, 6, 1>>(delta), delta_q, delta_t);
		Eigen::Map<const Eigen::Quaterniond> quater(x);
		Eigen::Map<Eigen::Quaterniond> quater_plus(x_plus_delta);
		Eigen::Map<Eigen::Vector3d> trans_plus(x_plus_delta + 4);

		quater_plus = delta_q * quater;
		trans_plus = delta_q * trans + delta_t;

		return true;
	}

	virtual bool ComputeJacobian(const double *x, double *jacobian) const
	{
		Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
		(j.topRows(6)).setIdentity();
		(j.bottomRows(1)).setZero();

		return true;
	}

	virtual int GlobalSize() const { return 7; }
	virtual int LocalSize() const { return 6; }
};

//平面特征参数化(球坐标形式)
class LocalPlaneParameterization : public ceres::LocalParameterization
{
public:
	LocalPlaneParameterization() {}

	virtual ~LocalPlaneParameterization() {}

	//更新(Hesse 对半球坐标(a,e,d)的导数
	virtual bool Plus(const double *x,
					  const double *delta,
					  double *x_plus_delta) const
	{
		ORB_SLAM2::Plane3D<double> plane_old(Eigen::Vector4d(x[0], x[1], x[2], x[3]));
		plane_old.oplus(Eigen::Vector3d(delta[0], delta[1], delta[2]));

		Eigen::Map<Eigen::Vector4d> plane_new(x_plus_delta);
		plane_new = plane_old.coeffs();
	}

	//雅克比计算(theta, phi)
	virtual bool ComputeJacobian(const double *x,
								 double *jacobian) const
	{
		double cos_theta = ceres::cos(x[0]);
		double sin_theta = ceres::sin(x[0]);

		double cos_phi = ceres::cos(x[1]);
		double sin_phi = ceres::sin(x[1]);

		jacobian[0] = -1.0 * cos_phi * sin_theta;
		jacobian[1] = -1.0 * sin_phi * cos_theta;
		jacobian[2] = 0.0;

		jacobian[3] = cos_phi * sin_theta;
		jacobian[4] = -1.0 * sin_phi * sin_theta;
		jacobian[5] = 0.0;

		jacobian[6] = 0.0;
		jacobian[7] = cos_phi;
		jacobian[8] = 0.0;

		jacobian[9] = 0.0;
		jacobian[10] = 0.0;
		jacobian[11] = 1.0;

		return true;
	}

	virtual int GlobalSize() const { return 4; }
	virtual int LocalSize() const { return 3; }
};

/********************************************************************************/
/*                  　 激光点误差（边缘点到线的误差）  自动求导模式                    */
/********************************************************************************/
struct LidarEdgeFactor
{
	LidarEdgeFactor(Eigen::Vector3d ref_point_, Eigen::Vector3d cur_point_a_,
					Eigen::Vector3d cur_point_b_, double weight_)
		: ref_point(ref_point_), cur_point_a(cur_point_a_), cur_point_b(cur_point_b_), weight(weight_)
	{
	}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> rp{T(ref_point.x()), T(ref_point.y()), T(ref_point.z())};
		Eigen::Matrix<T, 3, 1> cpa{T(cur_point_a.x()), T(cur_point_a.y()), T(cur_point_a.z())};
		Eigen::Matrix<T, 3, 1> cpb{T(cur_point_b.x()), T(cur_point_b.y()), T(cur_point_b.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_cur_ref{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_cur_ref{t[0], t[1], t[2]};

		// Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		// q_cur_ref = q_identity.slerp(T(s), q_cur_ref);
		// Eigen::Matrix<T, 3, 1> t_cur_ref{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> cp;
		cp = q_cur_ref * rp + t_cur_ref;

		Eigen::Matrix<T, 3, 1> nu = (cp - cpa).cross(cp - cpb);
		Eigen::Matrix<T, 3, 1> de = cpa - cpb;

		residual[0] = nu.x() / de.norm();
		residual[1] = nu.y() / de.norm();
		residual[2] = nu.z() / de.norm();

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d ref_point_, const Eigen::Vector3d cur_point_a_,
									   const Eigen::Vector3d cur_point_b_, const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarEdgeFactor, 3, 4, 3>(
			new LidarEdgeFactor(ref_point_, cur_point_a_, cur_point_b_, s_)));
	}

	Eigen::Vector3d ref_point, cur_point_a, cur_point_b;
	double weight;
};

/********************************************************************************/
/*                  　 平坦点到平面的误差  自动求导模式                    */
/********************************************************************************/
struct LidarPlaneNormFactor
{
	LidarPlaneNormFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_, double negative_OA_dot_norm_, double weight_)
		: curr_point(curr_point_), plane_unit_norm(plane_unit_norm_), negative_OA_dot_norm(negative_OA_dot_norm_), weight(weight_)
	{
	}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;

		Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
		residual[0] = norm.dot(point_w) + T(negative_OA_dot_norm);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d plane_unit_norm_,
									   const double negative_OA_dot_norm_, double weight_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneNormFactor, 1, 4, 3>(
			new LidarPlaneNormFactor(curr_point_, plane_unit_norm_, negative_OA_dot_norm_, weight_)));
	}

private:
	double weight;
	Eigen::Vector3d curr_point;
	Eigen::Vector3d plane_unit_norm;
	double negative_OA_dot_norm;
};

struct LidarGroundPlaneFactor
{
	LidarGroundPlaneFactor(const Eigen::Vector4d ground_plane_c, double weight)
		: plane_c_(ground_plane_c), weight_(weight)
	{
	}

	template <typename T>
	bool operator()(const T *const qwc, const T *const twc, const T *const p_w, T *residuals) const
	{
		//相机姿态
		Eigen::Quaternion<T> q_w_curr{qwc[3], qwc[0], qwc[1], qwc[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{twc[0], twc[1], twc[2]};

		//将局部平面转到世界坐标系中
		Eigen::Matrix<T, 3, 1> p_c_n = Eigen::Matrix<T, 3, 1>::Zero();
		p_c_n << T(plane_c_[0]), T(plane_c_[1]), T(plane_c_[2]);

		Eigen::Matrix<T, 3, 1> p_w_n = q_w_curr * p_c_n;
		T p_w_d = T(plane_c_[3]) - 1.0 * (t_w_curr.transpose() * q_w_curr.toRotationMatrix() * p_c_n).x();

		//构造平面
		ORB_SLAM2::Plane3D<T> plane_w_trans =
			ORB_SLAM2::Plane3D<T>(Eigen::Matrix<T, 4, 1>(p_w_n(0), p_w_n(1), p_w_n(2), p_w_d));
		ORB_SLAM2::Plane3D<T> plane_w_measure =
			ORB_SLAM2::Plane3D<T>(Eigen::Matrix<T, 4, 1>(T(p_w[0]), T(p_w[1]), T(p_w[2]), T(p_w[3])));

		Eigen::Matrix<T, 3, 1> err = plane_w_trans.ominus(plane_w_measure);

		residuals[0] = err(0);
		residuals[1] = err(1);
		residuals[2] = err(2);

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector4d &plane_c, double weight)
	{
		//误差维度、参数块维度
		return (new ceres::AutoDiffCostFunction<LidarGroundPlaneFactor, 3, 4, 3, 4>(
			new LidarGroundPlaneFactor(plane_c, weight)));
	}

private:
	double weight_;
	Eigen::Vector4d plane_c_; //当前帧提取到的平面
};

class ReprojectionErrorSE3Analytic : public ceres::SizedCostFunction<2, 7>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	ReprojectionErrorSE3Analytic(const double u, const double v,
								 double fx, double fy, double cx, double cy,
								 const Eigen::Vector3d &wpt, const double inv_sigma = 1.)
		: unpx_(u, v), wpt_(wpt), fx_(fx), fy_(fy), cx_(cx), cy_(cy)
	{
		sqrt_info_ = inv_sigma * Eigen::Matrix2d::Identity(); //信息矩阵
	}

	virtual bool Evaluate(double const *const *parameters,
						  double *residuals,
						  double **jacobians) const
	{
		// [tx, ty, tz, qw, qx, qy, qz]
		Eigen::Map<const Eigen::Vector3d> twc(parameters[0]);
		Eigen::Map<const Eigen::Quaterniond> qwc(parameters[0] + 3);

		Sophus::SE3 Twc(qwc, twc);
		Sophus::SE3 Tcw = Twc.inverse();

		// Compute left/right reproj err
		Eigen::Vector2d pred;

		Eigen::Vector3d lcampt = Tcw * wpt_;

		const double linvz = 1. / lcampt.z();

		pred << fx_ * lcampt.x() * linvz + cx_,
			fy_ * lcampt.y() * linvz + cy_;

		Eigen::Map<Eigen::Vector2d> werr(residuals);
		werr = sqrt_info_ * (pred - unpx_);

		// Update chi2err and depthpos info for
		// post optim checking
		chi2err_ = 0.;
		for (int i = 0; i < 2; i++)
			chi2err_ += std::pow(residuals[i], 2);

		isdepthpositive_ = true;
		if (lcampt.z() <= 0)
			isdepthpositive_ = false;

		if (jacobians != NULL)
		{
			const double linvz2 = linvz * linvz;

			Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_lcam;
			J_lcam << linvz * fx_, 0., -lcampt.x() * linvz2 * fx_,
				0., linvz * fy_, -lcampt.y() * linvz2 * fy_;

			Eigen::Matrix<double, 2, 3> J_lRcw;
			J_lRcw.noalias() = J_lcam * Tcw.rotation_matrix();

			if (jacobians[0] != NULL)
			{
				Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_se3pose(jacobians[0]);
				J_se3pose.setZero();

				J_se3pose.block<2, 3>(0, 0).noalias() = -1. * J_lRcw;
				J_se3pose.block<2, 3>(0, 3).noalias() = J_lRcw * Sophus::SO3::hat(wpt_);

				J_se3pose = sqrt_info_ * J_se3pose.eval();
			}
		}

		return true;
	}

	// Mutable var. that will be updated in const Evaluate()
	mutable double chi2err_;
	mutable bool isdepthpositive_;
	Eigen::Matrix2d sqrt_info_;

private:
	Eigen::Vector2d unpx_;
	Eigen::Vector3d wpt_;
	double fx_, fy_, cx_, cy_;
};

//视觉光束法平差误差函数
struct ReprojectionFactor
{
	ReprojectionFactor(double observed_x_, double observed_y_, Eigen::Vector3d pw_, vk::PinholeCamera *pinhole_cam_, double inv_sigma_)
		: observed_x(observed_x_), observed_y(observed_y_), pw(pw_), mpPinhole_cam(pinhole_cam_), inv_sigma(inv_sigma_)
	{
		sqrt_info = inv_sigma * Eigen::Matrix2d::Identity();
	}

	template <typename T>
	bool operator()(const T *qwc, const T *twc, T *residuals) const
	{
		Eigen::Quaternion<T> q_w_curr{qwc[3], qwc[0], qwc[1], qwc[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{twc[0], twc[1], twc[2]};

		Eigen::Quaternion<T> q_curr_w = q_w_curr.inverse();
		q_curr_w.normalize();
		Eigen::Matrix<T, 3, 1> t_curr_w = q_curr_w * t_w_curr;

		Eigen::Matrix<T, 3, 1> p_w{T(pw[0]), T(pw[1]), T(pw[2])};
		Eigen::Matrix<T, 3, 1> p_c = q_curr_w * p_w - t_curr_w;

		T predicted_x = p_c[0] / p_c[2] * T(mpPinhole_cam->fx()) + T(mpPinhole_cam->cx());
		T predicted_y = p_c[1] / p_c[2] * T(mpPinhole_cam->fy()) + T(mpPinhole_cam->cy());

		// The error is the difference between the predicted and observed position.
		Eigen::Matrix<T, 2, 1> err;
		err[0] = predicted_x - T(observed_x);
		err[1] = predicted_y - T(observed_y);

		Eigen::Map<Eigen::Matrix<T, 2, 1>> w_err(residuals);
		w_err = sqrt_info.cast<T>() * err;

		return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction *Create(const double observed_x_,
									   const double observed_y_,
									   Eigen::Vector3d pw_,
									   vk::PinholeCamera *pinhole_cam_,
									   double inv_sigma_)
	{
		return (new ceres::AutoDiffCostFunction<ReprojectionFactor, 2, 4, 3>(
			new ReprojectionFactor(observed_x_, observed_y_, pw_, pinhole_cam_, inv_sigma_)));
	}

	Eigen::Vector3d pw;
	double inv_sigma;
	Eigen::Matrix2d sqrt_info;

	double observed_x;
	double observed_y;
	vk::PinholeCamera *mpPinhole_cam;
};

class SnavelyReprojectionFactorPoseOnly : public ceres::SizedCostFunction<2, 7>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	SnavelyReprojectionFactorPoseOnly(const double u, const double v,
									  double fx, double fy, double cx, double cy,
									  const Eigen::Vector3d &wpt, const double inv_sigma = 1.)
		: unpx_(u, v), wpt_(wpt), fx_(fx), fy_(fy), cx_(cx), cy_(cy)
	{
		sqrt_info_ = inv_sigma * Eigen::Matrix2d::Identity(); //信息矩阵
	}

	virtual bool Evaluate(double const *const *parameters,
						  double *residuals,
						  double **jacobians) const
	{
		// [qx, qy, qz,qw,tx, ty, tz,]
		Eigen::Map<const Eigen::Quaterniond> qwc(parameters[0]);
		Eigen::Map<const Eigen::Vector3d> twc(parameters[0] + 4);

		Sophus::SE3 Twc(qwc, twc);
		Sophus::SE3 Tcw = Twc.inverse();

		// Compute left/right reproj err
		Eigen::Vector2d pred;

		Eigen::Vector3d lcampt = Tcw * wpt_;

		const double linvz = 1. / lcampt.z();

		pred << fx_ * lcampt.x() * linvz + cx_,
			fy_ * lcampt.y() * linvz + cy_;

		Eigen::Map<Eigen::Vector2d> werr(residuals);
		werr = sqrt_info_ * (pred - unpx_);

		// Update chi2err and depthpos info for
		// post optim checking
		chi2err_ = 0.;
		for (int i = 0; i < 2; i++)
			chi2err_ += std::pow(residuals[i], 2);

		isdepthpositive_ = true;
		if (lcampt.z() <= 0)
			isdepthpositive_ = false;

		if (jacobians != NULL)
		{
			const double linvz2 = linvz * linvz;

			Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_lcam;
			J_lcam << linvz * fx_, 0., -lcampt.x() * linvz2 * fx_,
				0., linvz * fy_, -lcampt.y() * linvz2 * fy_;

			Eigen::Matrix<double, 2, 3> J_lRcw;
			J_lRcw.noalias() = J_lcam * Tcw.rotation_matrix();

			if (jacobians[0] != NULL)
			{
				Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_se3pose(jacobians[0]);
				J_se3pose.setZero();

				J_se3pose.block<2, 3>(0, 0).noalias() = -1. * J_lRcw * Sophus::SO3::hat(wpt_);
				J_se3pose.block<2, 3>(0, 3).noalias() = J_lRcw;

				J_se3pose = sqrt_info_ * J_se3pose.eval();
			}
		}

		return true;
	}

	// Mutable var. that will be updated in const Evaluate()
	mutable double chi2err_;
	mutable bool isdepthpositive_;
	Eigen::Matrix2d sqrt_info_;

private:
	Eigen::Vector2d unpx_;
	Eigen::Vector3d wpt_;
	double fx_, fy_, cx_, cy_;
};


class PhotometricFactorPoseOnly : public ceres::SizedCostFunction<1, 7>
{
public:

};



//激光边缘点到空间线段的距离
class EdgeAnalyticCostFunction : public ceres::SizedCostFunction<3, 7>
{
public:
	EdgeAnalyticCostFunction(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_, Eigen::Vector3d last_point_b_, double &weight_, double *cost_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), weight(weight_), cost(cost_)
	{
	}

	virtual ~EdgeAnalyticCostFunction() {}

	virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
	{
		Eigen::Map<const Eigen::Quaterniond> q_last_curr(parameters[0]);
		Eigen::Map<const Eigen::Vector3d> t_last_curr(parameters[0] + 4);
		Eigen::Vector3d lp;
		lp = q_last_curr * curr_point + t_last_curr;

		Eigen::Vector3d nu = (lp - last_point_a).cross(lp - last_point_b);
		Eigen::Vector3d de = last_point_a - last_point_b;
		double de_norm = de.norm();

		residuals[0] = nu.x() / de_norm * weight;
		residuals[1] = nu.y() / de_norm * weight;
		residuals[2] = nu.z() / de_norm * weight;

		*cost = residuals[0] * residuals[0] + residuals[1] * residuals[1] + residuals[2] * residuals[2];

		if (jacobians != NULL)
		{
			if (jacobians[0] != NULL)
			{
				Eigen::Matrix3d skew_lp = ORB_SLAM2::skew(lp);
				Eigen::Matrix<double, 3, 6> dp_by_se3;
				dp_by_se3.block<3, 3>(0, 0) = -skew_lp * weight;
				dp_by_se3.block<3, 3>(0, 3).setIdentity() * weight;

				Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J_se3(jacobians[0]);
				J_se3.setZero();
				Eigen::Matrix3d skew_de = ORB_SLAM2::skew(de);
				J_se3.block<3, 6>(0, 0) = -skew_de * dp_by_se3 / de_norm;
			}
		}

		return true;
	}

private:
	Eigen::Vector3d curr_point;
	Eigen::Vector3d last_point_a;
	Eigen::Vector3d last_point_b;

	double weight;
	mutable double *cost;
};

//激光平坦点到空间平面的距离
class SurfNormAnalyticCostFunction : public ceres::SizedCostFunction<1, 7>
{
public:
	SurfNormAnalyticCostFunction(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_, double negative_OA_dot_norm_, double &weight_, double *cost_)
		: curr_point(curr_point_), plane_unit_norm(plane_unit_norm_), negative_OA_dot_norm(negative_OA_dot_norm_), weight(weight_), cost(cost_)
	{
	}
	virtual ~SurfNormAnalyticCostFunction() {}
	virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
	{
		Eigen::Map<const Eigen::Quaterniond> q_w_curr(parameters[0]);
		Eigen::Map<const Eigen::Vector3d> t_w_curr(parameters[0] + 4);
		Eigen::Vector3d point_w = q_w_curr * curr_point + t_w_curr;
		residuals[0] = plane_unit_norm.dot(point_w) + negative_OA_dot_norm;
		*cost = residuals[0] * residuals[0];

		if (jacobians != NULL)
		{
			if (jacobians[0] != NULL)
			{
				Eigen::Matrix3d skew_point_w = ORB_SLAM2::skew(point_w);
				Eigen::Matrix<double, 3, 6> dp_by_se3;
				dp_by_se3.block<3, 3>(0, 0) = -skew_point_w * weight;
				dp_by_se3.block<3, 3>(0, 3).setIdentity() * weight;
				Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> J_se3(jacobians[0]);
				J_se3.setZero();
				J_se3.block<1, 6>(0, 0) = plane_unit_norm.transpose() * dp_by_se3;
			}
		}
		return true;
	}

private:
	double weight;
	mutable double *cost;

	Eigen::Vector3d curr_point;
	Eigen::Vector3d plane_unit_norm;
	double negative_OA_dot_norm;
};

#endif //CERES_FACTOR_HPP