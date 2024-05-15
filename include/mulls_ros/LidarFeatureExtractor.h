/*
 * @Author: kinggreat24
 * @Date: 2022-06-05 21:19:32
 * @LastEditTime: 2022-06-05 23:08:05
 * @LastEditors: kinggreat24
 * @Description: 
 * @FilePath: /mulls_ros/include/mulls_ros/LidarFeatureExtractor.h
 * 可以输入预定的版权声明、个性签名、空行等
 */

#ifndef LIDAR_FEATURE_EXTRACTOR_H
#define LIDAR_FEATURE_EXTRACTOR_H

#include "utility.hpp"

#include "MultiScanRegistration.h"
#include "patchwork.hpp"

namespace ORB_SLAM2
{

    class LidarFeatureExtractor
    {
    public:
        LidarFeatureExtractor(const std::string &lidar_type);

        void ExtractLoamFeatures(const pcl::PointCloud<Point_T> &pointcloud_in,
                                 const double sweep_time,
                                 pcl::PointCloud<Point_T>::Ptr cornerPointsSharp,
                                 pcl::PointCloud<Point_T>::Ptr cornerPointsLessSharp,
                                 pcl::PointCloud<Point_T>::Ptr surfPointsFlat,
                                 pcl::PointCloud<Point_T>::Ptr surfPointsLessFlat);

        void PatchWorkGroundSegmentation(
            const pcl::PointCloud<Point_T>::Ptr &lidar_in, pcl::PointCloud<Point_T>::Ptr &ground_cloud,
            pcl::PointCloud<Point_T>::Ptr &obstacle_cloud, double time_takens);

    private:
        MultiScanRegistration *multiScanRegistraton_; //LOAM方式提取激光特征

        boost::shared_ptr<PatchWork<Point_T>> mpPatchworkGroundSeg; //地面点云提取
    };

}

#endif //LIDAR_FEATURE_EXTRACTOR_H