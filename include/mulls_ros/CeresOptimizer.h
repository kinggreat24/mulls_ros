/*
 * @Author: kinggreat24
 * @Date: 2020-11-30 09:26:31
 * @LastEditTime: 2022-06-06 13:36:44
 * @LastEditors: kinggreat24
 * @Description: 
 * @FilePath: /mulls_ros/include/mulls_ros/CeresOptimizer.h
 * @可以输入预定的版权声明、个性签名、空行等
 */
#ifndef CERES_OPTIMIZER_H
#define CERES_OPTIMIZER_H

#include <iostream>
#include <vector>
#include <chrono>

#include "CeresFactor.hpp"
#include "utility.hpp"

namespace ORB_SLAM2
{

    class CeresOptimizer
    {
    public:
        inline double line_weight_evaluate(const Eigen::Vector3d &eigen_val)
        {
            double weight = std::sqrt((eigen_val[2] * eigen_val[2] - eigen_val[1] * eigen_val[1]) / (eigen_val[2] * eigen_val[2]));
            return weight;
        }

        inline double plane_weight_evaluate(const Eigen::Vector3d &eigen_val)
        {
            double weight = std::sqrt((eigen_val[2] * eigen_val[2] - eigen_val[0] * eigen_val[0]) / (eigen_val[2] * eigen_val[2]));
            return weight;
        }

        int static PoseOptimization(lo::cloudblock_Ptr pF,
                                    pcl::PointCloud<Point_T>::Ptr &pLocalCornerMap,
                                    pcl::PointCloud<Point_T>::Ptr &pLocalSurfMap,
                                    pcl::PointCloud<Point_T>::Ptr &pCurCornerScan,
                                    pcl::PointCloud<Point_T>::Ptr &pCurSurfScan,
                                    int max_iteration = 3);
    };

}

#endif //CERES_OPTIMIZER_H