/*
 * @Author: kinggreat24
 * @Date: 2022-06-05 21:25:50
 * @LastEditTime: 2022-06-06 14:33:27
 * @LastEditors: kinggreat24
 * @Description: 
 * @FilePath: /mulls_ros/src/LidarFeatureExtractor.cpp
 * 可以输入预定的版权声明、个性签名、空行等
 */
#include "LidarFeatureExtractor.h"

namespace ORB_SLAM2
{
    LidarFeatureExtractor::LidarFeatureExtractor(const std::string &setting_file)
    {
        cv::FileStorage fSettings(setting_file, cv::FileStorage::READ);

        /****************          LOAM 方式提取激光点云特征         ****************/
        std::string lidar_type = fSettings["LidarType"];
        multiScanRegistraton_ = new MultiScanRegistration();
        RegistrationParams registrationParams;
        multiScanRegistraton_->setupRegistrationParams(lidar_type, registrationParams);

        /***************         地面提取PatchWork参数设置               *******************/
        _patchwork_param_t patchwork_param;
        patchwork_param.sensor_height_ = fSettings["patchwork.GPF.sensor_height"];
        fSettings["patchwork.verbose"] >> patchwork_param.verbose_;

        patchwork_param.num_iter_ = fSettings["patchwork.GPF.num_iter"];
        patchwork_param.num_lpr_ = fSettings["patchwork.GPF.num_lpr"];
        patchwork_param.num_min_pts_ = fSettings["patchwork.GPF.num_min_pts"];
        patchwork_param.th_seeds_ = fSettings["patchwork.GPF.th_seeds"];
        patchwork_param.th_dist_ = fSettings["patchwork.GPF.th_dist"];
        patchwork_param.max_range_ = fSettings["patchwork.GPF.max_r"];
        patchwork_param.min_range_ = fSettings["patchwork.GPF.min_r"];
        patchwork_param.num_rings_ = fSettings["patchwork.uniform.num_rings"];
        patchwork_param.num_sectors_ = fSettings["patchwork.uniform.num_sectors"];
        patchwork_param.uprightness_thr_ = fSettings["patchwork.GPF.uprightness_thr"];
        patchwork_param.adaptive_seed_selection_margin_ = fSettings["patchwork.adaptive_seed_selection_margin"];

        // For global threshold
        fSettings["patchwork.using_global_elevation"] >> patchwork_param.using_global_thr_;
        patchwork_param.global_elevation_thr_ = fSettings["patchwork.global_elevation_threshold"];

        patchwork_param.num_zones_ = fSettings["patchwork.czm.num_zones"];
        std::cout << "patchwork_param.num_zones_: " << patchwork_param.num_zones_ << std::endl;

        //num_sectors_each_zone
        cv::FileNode czm_num_sectors = fSettings["patchwork.czm.num_sectors_each_zone"];
        if (czm_num_sectors.type() != cv::FileNode::SEQ)
        {
            std::cerr << "num_sectors_each_zone is not a sequence" << std::endl;
            return;
        }
        cv::FileNodeIterator it_sector = czm_num_sectors.begin(), it_sector_end = czm_num_sectors.end();
        for (; it_sector != it_sector_end; it_sector++)
        {
            patchwork_param.num_sectors_each_zone_.push_back(*it_sector);
        }

        //num_rings_each_zone
        cv::FileNode czm_num_rings = fSettings["patchwork.czm.num_rings_each_zone"];
        if (czm_num_rings.type() != cv::FileNode::SEQ)
        {
            std::cerr << "num_rings_each_zone is not a sequence" << std::endl;
            return;
        }
        cv::FileNodeIterator it_ring = czm_num_rings.begin(), it_ring_end = czm_num_rings.end();
        for (; it_ring != it_ring_end; it_ring++)
        {
            patchwork_param.num_rings_each_zone_.push_back(*it_ring);
        }

        //min_ranges_
        cv::FileNode min_ranges = fSettings["patchwork.czm.min_ranges_each_zone"];
        if (min_ranges.type() != cv::FileNode::SEQ)
        {
            std::cerr << "min_ranges_each_zone is not a sequence" << std::endl;
            return;
        }
        cv::FileNodeIterator it_min_range = min_ranges.begin(), it_min_range_end = min_ranges.end();
        for (; it_min_range != it_min_range_end; it_min_range++)
        {
            patchwork_param.min_ranges_.push_back(*it_min_range);
        }

        // elevation_thr_
        cv::FileNode elevation_thresholds = fSettings["patchwork.czm.elevation_thresholds"];
        if (elevation_thresholds.type() != cv::FileNode::SEQ)
        {
            std::cerr << "elevation_thresholds is not a sequence" << std::endl;
            return;
        }
        cv::FileNodeIterator it_elevation_threshold = elevation_thresholds.begin(), it_elevation_threshold_end = elevation_thresholds.end();
        for (; it_elevation_threshold != it_elevation_threshold_end; it_elevation_threshold++)
        {
            patchwork_param.elevation_thr_.push_back(*it_elevation_threshold);
        }

        //flatness_thr_
        cv::FileNode flatness_thresholds = fSettings["patchwork.czm.flatness_thresholds"];
        if (flatness_thresholds.type() != cv::FileNode::SEQ)
        {
            std::cerr << "flatness_thresholds is not a sequence" << std::endl;
            return;
        }
        cv::FileNodeIterator it_flatness_threshold = flatness_thresholds.begin(), it_flatness_threshold_end = flatness_thresholds.end();
        for (; it_flatness_threshold != it_flatness_threshold_end; it_flatness_threshold++)
        {
            patchwork_param.flatness_thr_.push_back(*it_flatness_threshold);
        }

        mpPatchworkGroundSeg.reset(new PatchWork<Point_T>(patchwork_param));
    }

    void LidarFeatureExtractor::ExtractLoamFeatures(const pcl::PointCloud<Point_T> &pointcloud_in,
                                                    const double sweep_time,
                                                    pcl::PointCloud<Point_T>::Ptr cornerPointsSharp,
                                                    pcl::PointCloud<Point_T>::Ptr cornerPointsLessSharp,
                                                    pcl::PointCloud<Point_T>::Ptr surfPointsFlat,
                                                    pcl::PointCloud<Point_T>::Ptr surfPointsLessFlat)
    {
        unsigned long int sec = (unsigned long int)sweep_time;
        unsigned long int nsec = (sweep_time - sec) * 10e9;
        auto epoch = std::chrono::system_clock::time_point();
        auto since_epoch = std::chrono::seconds(sec) + std::chrono::nanoseconds(nsec);

        multiScanRegistraton_->handleCloudMessage(
            pointcloud_in,
            Time(since_epoch + epoch),
            *cornerPointsSharp,
            *cornerPointsLessSharp,
            *surfPointsFlat,
            *surfPointsLessFlat);
    }

    void LidarFeatureExtractor::PatchWorkGroundSegmentation(
        const pcl::PointCloud<Point_T>::Ptr &lidar_in, pcl::PointCloud<Point_T>::Ptr &ground_cloud,
        pcl::PointCloud<Point_T>::Ptr &obstacle_cloud, double time_takens)
    {
        mpPatchworkGroundSeg->estimate_ground(*lidar_in, *ground_cloud, *obstacle_cloud, time_takens);
    }

}
