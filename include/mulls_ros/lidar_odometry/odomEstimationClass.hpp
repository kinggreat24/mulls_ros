/*
 * @Author: kinggreat24
 * @Date: 2021-04-06 12:41:42
 * @LastEditTime: 2024-04-22 17:38:29
 * @LastEditors: kinggreat24 kinggreat24@whu.edu.cn
 * @Description:
 * @FilePath: /floam/include/odomEstimationClass.h
 * 可以输入预定的版权声明、个性签名、空行等
 */
// Author of FLOAM: Wang Han
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro
#ifndef _ODOM_ESTIMATION_CLASS_H_
#define _ODOM_ESTIMATION_CLASS_H_

// std lib
#include <string>
#include <math.h>
#include <vector>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_box.h>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

// eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

// LOCAL LIB
#include "CeresFactor.hpp"

template <typename PointT>
class OdomEstimationClass
{
public:
	OdomEstimationClass(){}

	void init(double map_resolution)
	{
		// init local map
		laserCloudCornerMap = typename pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());
		laserCloudSurfMap = typename pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());

		// downsampling size
		downSizeFilterEdge.setLeafSize(map_resolution, map_resolution, map_resolution);
		downSizeFilterSurf.setLeafSize(map_resolution * 2, map_resolution * 2, map_resolution * 2);

		// kd-tree
		kdtreeEdgeMap = typename pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>());
		kdtreeSurfMap = typename pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>());

		odom = Eigen::Isometry3d::Identity();
		last_odom = Eigen::Isometry3d::Identity();
		optimization_count = 2;
	}
	void initMapWithPoints(const typename pcl::PointCloud<PointT>::Ptr &edge_in, const typename pcl::PointCloud<PointT>::Ptr &surf_in)
	{
		*laserCloudCornerMap += *edge_in;
		*laserCloudSurfMap += *surf_in;
		optimization_count = 12;
	}
	void updatePointsToMap(const typename pcl::PointCloud<PointT>::Ptr &edge_in, const typename pcl::PointCloud<PointT>::Ptr &surf_in)
	{
		if (optimization_count > 2)
			optimization_count--;

		Eigen::Isometry3d odom_prediction = odom * (last_odom.inverse() * odom);
		last_odom = odom;
		odom = odom_prediction;

		q_w_curr = Eigen::Quaterniond(odom.rotation());
		t_w_curr = odom.translation();

		typename pcl::PointCloud<PointT>::Ptr downsampledEdgeCloud(new pcl::PointCloud<PointT>());
		typename pcl::PointCloud<PointT>::Ptr downsampledSurfCloud(new pcl::PointCloud<PointT>());
		downSamplingToMap(edge_in, downsampledEdgeCloud, surf_in, downsampledSurfCloud);
		// ROS_WARN("point nyum%d,%d",(int)downsampledEdgeCloud->points.size(), (int)downsampledSurfCloud->points.size());
		if (laserCloudCornerMap->points.size() > 10 && laserCloudSurfMap->points.size() > 50)
		{
			kdtreeEdgeMap->setInputCloud(laserCloudCornerMap);
			kdtreeSurfMap->setInputCloud(laserCloudSurfMap);

			double t_lidar_association = 0.0;
			for (int iterCount = 0; iterCount < optimization_count; iterCount++)
			{
				ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
				ceres::Problem::Options problem_options;
				ceres::Problem problem(problem_options);

				problem.AddParameterBlock(parameters, 7, new PoseSE3Parameterization());

				std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

				addEdgeCostFactor(downsampledEdgeCloud, laserCloudCornerMap, problem, loss_function);
				addSurfCostFactor(downsampledSurfCloud, laserCloudSurfMap, problem, loss_function);

				std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
				double tlidar_association = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
				t_lidar_association += tlidar_association;

				ceres::Solver::Options options;
				options.linear_solver_type = ceres::DENSE_QR;
				options.max_num_iterations = 4;
				options.minimizer_progress_to_stdout = false;
				options.check_gradients = false;
				options.gradient_check_relative_precision = 1e-4;
				ceres::Solver::Summary summary;

				ceres::Solve(options, &problem, &summary);
			}
			// std::cout << "t_lidar_association: " << t_lidar_association << std::endl;
		}
		else
		{
			printf("not enough points in map to associate, map error");
		}
		odom = Eigen::Isometry3d::Identity();
		odom.linear() = q_w_curr.toRotationMatrix();
		odom.translation() = t_w_curr;
		addPointsToMap(downsampledEdgeCloud, downsampledSurfCloud);
	}

	void getMap(typename pcl::PointCloud<PointT>::Ptr &laserCloudMap)
	{
		*laserCloudMap += *laserCloudSurfMap;
		*laserCloudMap += *laserCloudCornerMap;
	}

	Eigen::Isometry3d odom;
	typename pcl::PointCloud<PointT>::Ptr laserCloudCornerMap;
	typename pcl::PointCloud<PointT>::Ptr laserCloudSurfMap;

private:
	// optimization variable
	double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
	Eigen::Map<Eigen::Quaterniond> q_w_curr = Eigen::Map<Eigen::Quaterniond>(parameters);
	Eigen::Map<Eigen::Vector3d> t_w_curr = Eigen::Map<Eigen::Vector3d>(parameters + 4);

	Eigen::Isometry3d last_odom;

	// kd-tree
	typename pcl::KdTreeFLANN<PointT>::Ptr kdtreeEdgeMap;
	typename pcl::KdTreeFLANN<PointT>::Ptr kdtreeSurfMap;

	// points downsampling before add to map
	typename pcl::VoxelGrid<PointT> downSizeFilterEdge;
	typename pcl::VoxelGrid<PointT> downSizeFilterSurf;

	// local map
	typename pcl::CropBox<PointT> cropBoxFilter;

	// optimization count
	int optimization_count;

	// function
	void addEdgeCostFactor(const typename pcl::PointCloud<PointT>::Ptr &pc_in, const typename pcl::PointCloud<PointT>::Ptr &map_in, ceres::Problem &problem, ceres::LossFunction *loss_function)
	{
		int scan_edge_size = (int)pc_in->points.size();
		Eigen::Matrix<double, 1, Eigen::Dynamic> edge_weights(1, scan_edge_size);
		Eigen::Matrix<double, 1, Eigen::Dynamic> edge_residuals(1, scan_edge_size);
		edge_weights.setOnes();
		edge_residuals.setZero();

		int corner_num = 0;
		for (int i = 0; i < (int)pc_in->points.size(); i++)
		{
			PointT point_temp;
			pointAssociateToMap(&(pc_in->points[i]), &point_temp);

			std::vector<int> pointSearchInd;
			std::vector<float> pointSearchSqDis;
			kdtreeEdgeMap->nearestKSearch(point_temp, 5, pointSearchInd, pointSearchSqDis);
			if (pointSearchSqDis[4] < 1.0)
			{
				std::vector<Eigen::Vector3d> nearCorners;
				Eigen::Vector3d center(0, 0, 0);
				for (int j = 0; j < 5; j++)
				{
					Eigen::Vector3d tmp(map_in->points[pointSearchInd[j]].x,
										map_in->points[pointSearchInd[j]].y,
										map_in->points[pointSearchInd[j]].z);
					center = center + tmp;
					nearCorners.push_back(tmp);
				}
				center = center / 5.0;

				Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
				for (int j = 0; j < 5; j++)
				{
					Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
					covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
				}

				Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

				Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
				Eigen::Vector3d curr_point(pc_in->points[i].x, pc_in->points[i].y, pc_in->points[i].z);
				if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
				{
					Eigen::Vector3d point_on_line = center;
					Eigen::Vector3d point_a, point_b;
					point_a = 0.1 * unit_direction + point_on_line;
					point_b = -0.1 * unit_direction + point_on_line;

					double weight = 1.0;
					double cost = 0.0;

					ceres::CostFunction *cost_function = new EdgeAnalyticCostFunction(curr_point, point_a, point_b, weight, &cost);
					problem.AddResidualBlock(cost_function, loss_function, parameters);
					corner_num++;
				}
			}
		}
		if (corner_num < 20)
		{
			printf("not enough correct points");
		}

		// std::cout<<"corner_measurements: "<<corner_num<<std::endl;
	}
	void addSurfCostFactor(const typename pcl::PointCloud<PointT>::Ptr &pc_in, const typename pcl::PointCloud<PointT>::Ptr &map_in, ceres::Problem &problem, ceres::LossFunction *loss_function)
	{
		int surf_num = 0;
		for (int i = 0; i < (int)pc_in->points.size(); i++)
		{
			PointT point_temp;
			pointAssociateToMap(&(pc_in->points[i]), &point_temp);
			std::vector<int> pointSearchInd;
			std::vector<float> pointSearchSqDis;
			kdtreeSurfMap->nearestKSearch(point_temp, 5, pointSearchInd, pointSearchSqDis);

			Eigen::Matrix<double, 5, 3> matA0;
			Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
			if (pointSearchSqDis[4] < 1.0)
			{

				for (int j = 0; j < 5; j++)
				{
					matA0(j, 0) = map_in->points[pointSearchInd[j]].x;
					matA0(j, 1) = map_in->points[pointSearchInd[j]].y;
					matA0(j, 2) = map_in->points[pointSearchInd[j]].z;
				}
				// find the norm of plane
				Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
				double negative_OA_dot_norm = 1 / norm.norm();
				norm.normalize();

				bool planeValid = true;
				for (int j = 0; j < 5; j++)
				{
					// if OX * n > 0.2, then plane is not fit well
					if (fabs(norm(0) * map_in->points[pointSearchInd[j]].x +
							 norm(1) * map_in->points[pointSearchInd[j]].y +
							 norm(2) * map_in->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
					{
						planeValid = false;
						break;
					}
				}
				Eigen::Vector3d curr_point(pc_in->points[i].x, pc_in->points[i].y, pc_in->points[i].z);
				if (planeValid)
				{
					double weight = 1.0;
					double cost = 0.0;

					ceres::CostFunction *cost_function = new SurfNormAnalyticCostFunction(curr_point, norm, negative_OA_dot_norm, weight, &cost);
					problem.AddResidualBlock(cost_function, loss_function, parameters);

					surf_num++;
				}
			}
		}
		if (surf_num < 20)
		{
			printf("not enough correct points");
		}

		// std::cout<<"surf_measurements: "<<surf_num<<std::endl;
	}

	void addPointsToMap(const typename pcl::PointCloud<PointT>::Ptr &downsampledEdgeCloud, const typename pcl::PointCloud<PointT>::Ptr &downsampledSurfCloud)
	{
		std::chrono::steady_clock::time_point t1_raw = std::chrono::steady_clock::now();
		typename pcl::PointCloud<PointT>::Ptr surf_downsample(new pcl::PointCloud<PointT>());
		typename pcl::PointCloud<PointT>::Ptr edge_downsample(new pcl::PointCloud<PointT>());
		// pcl::PointCloud<PointT>::Ptr ground_downsample(new pcl::PointCloud<PointT>());
		pcl::transformPointCloud(
			*downsampledEdgeCloud,
			*edge_downsample,
			odom.matrix());
		*laserCloudCornerMap += *edge_downsample;

		pcl::transformPointCloud(
			*downsampledSurfCloud,
			*surf_downsample,
			odom.matrix());
		*laserCloudSurfMap += *surf_downsample;

		// pcl::transformPointCloud(
		//     *(downsampledGroundCloud),
		//     *ground_downsample,
		//     odom.matrix());

		double x_min = +odom.translation().x() - 100;
		double y_min = +odom.translation().y() - 100;
		double z_min = +odom.translation().z() - 100;
		double x_max = +odom.translation().x() + 100;
		double y_max = +odom.translation().y() + 100;
		double z_max = +odom.translation().z() + 100;

		// ROS_INFO("size : %f,%f,%f,%f,%f,%f", x_min, y_min, z_min,x_max, y_max, z_max);
		cropBoxFilter.setMin(Eigen::Vector4f(x_min, y_min, z_min, 1.0));
		cropBoxFilter.setMax(Eigen::Vector4f(x_max, y_max, z_max, 1.0));
		cropBoxFilter.setNegative(false);

		typename pcl::PointCloud<PointT>::Ptr tmpCorner(new pcl::PointCloud<PointT>());
		typename pcl::PointCloud<PointT>::Ptr tmpSurf(new pcl::PointCloud<PointT>());
		cropBoxFilter.setInputCloud(laserCloudSurfMap);
		cropBoxFilter.filter(*tmpSurf);
		cropBoxFilter.setInputCloud(laserCloudCornerMap);
		cropBoxFilter.filter(*tmpCorner);

		downSizeFilterSurf.setInputCloud(tmpSurf);
		downSizeFilterSurf.filter(*laserCloudSurfMap);
		downSizeFilterEdge.setInputCloud(tmpCorner);
		downSizeFilterEdge.filter(*laserCloudCornerMap);

		std::chrono::steady_clock::time_point t2_raw = std::chrono::steady_clock::now();
		double tlidar_update_raw = std::chrono::duration_cast<std::chrono::duration<double>>(t2_raw - t1_raw).count();
		// std::cout << "original local map update time: " << tlidar_update_raw << std::endl;
	}

	void pointAssociateToMap(PointT const *const pi, PointT *const po)
	{
		Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
		Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
		po->x = point_w.x();
		po->y = point_w.y();
		po->z = point_w.z();
		po->intensity = pi->intensity;
		// po->intensity = 1.0;
	}

	void downSamplingToMap(const typename pcl::PointCloud<PointT>::Ptr &edge_pc_in, typename pcl::PointCloud<PointT>::Ptr &edge_pc_out, const typename pcl::PointCloud<PointT>::Ptr &surf_pc_in, typename pcl::PointCloud<PointT>::Ptr &surf_pc_out)
	{
		downSizeFilterEdge.setInputCloud(edge_pc_in);
		downSizeFilterEdge.filter(*edge_pc_out);
		downSizeFilterSurf.setInputCloud(surf_pc_in);
		downSizeFilterSurf.filter(*surf_pc_out);
	}
};

#endif // _ODOM_ESTIMATION_CLASS_H_
