/*
 * @Author: kinggreat24
 * @Date: 2022-06-01 19:33:33
 * @LastEditTime: 2022-07-26 00:09:07
 * @LastEditors: kinggreat24
 * @Description: 
 * @FilePath: /mulls_ros/include/mulls_ros/VisualFeatureExtractor.h
 * 可以输入预定的版权声明、个性签名、空行等
 */
#ifndef VISUAL_FEATURE_EXTRACTOR_H
#define VISUAL_FEATURE_EXTRACTOR_H

#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "ORBVocabulary.h"
#include "ELSED.h"

#include <vikit/pinhole_camera.h>
#include <vikit/vision.h>

#include "cfilter.hpp"

#include "utility.hpp"

namespace ORB_SLAM2
{

    class VisualFeatureExtractor
    {

    public:
        VisualFeatureExtractor(const lo::orb_params_t &orb_params, const lo::tracker_t &dvl_tracking_params, 
            vk::PinholeCamera *pinhole_camera_, lo::CFilter<Point_T>* cfilter, Eigen::Matrix4d& mTcam_lidar,  const std::string &voc_file);

        void ImagePreprocessing(const cv::Mat &imGray, lo::cloudblock_Ptr &clockcloud_ptr);
        void ExtractORB(const cv::Mat &imGray, lo::cloudblock_Ptr &clockcloud_ptr);
        void ShowLidarSamplePoints(cv::Mat &image_out, lo::cloudblock_Ptr &clockcloud_ptr, const int step);
    protected:
        void ComputeImageBounds(const cv::Mat &imLeft);
        bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);
        void AssignFeaturesToGrid(lo::cloudblock_Ptr &clockcloud_ptr);
        void UndistortKeyPoints(lo::cloudblock_Ptr &clockcloud_ptr);
        void ComputeBoW(lo::cloudblock_Ptr &clockcloud_ptr);

        void LidarPointsSampling(const cv::Mat &im, const pcTPtr &pc_camera, lo::cloudblock_Ptr &clockcloud_ptr);
                
        template <typename T>
        void PyrDownMeanSmooth(const cv::Mat &in, cv::Mat &out)
        {
            out.create(cv::Size(in.size().width / 2, in.size().height / 2), in.type());

#pragma omp parallel for //Multi-thread
            for (int y = 0; y < out.rows; ++y)
            {
                for (int x = 0; x < out.cols; ++x)
                {
                    int x0 = x * 2;
                    int x1 = x0 + 1;
                    int y0 = y * 2;
                    int y1 = y0 + 1;

                    out.at<T>(y, x) = (T)((in.at<T>(y0, x0) + in.at<T>(y0, x1) + in.at<T>(y1, x0) + in.at<T>(y1, x1)) / 4.0f);
                }
            }
        }

    private:
        ORB_SLAM2::ORBextractor *mpORBExtractor;
        ORB_SLAM2::ORBVocabulary *mpVocabulary;
        lo::CFilter<Point_T>* mpCfilter;


        lo::tracker_t tracker_params_;

        vk::PinholeCamera *mpPinholeCamModel;
        cv::Mat mDistCoef;
        cv::Mat mK;
        
        Eigen::Matrix4d mTcam_lidar;           //激光雷达到相机的外参

        // Undistorted Image Bounds (computed once).
        static float mnMinX;
        static float mnMaxX;
        static float mnMinY;
        static float mnMaxY;
        static bool mbInitialComputations;

        static float mfGridElementWidthInv;
        static float mfGridElementHeightInv;
    };

}

#endif //VISUAL_FEATURE_EXTRACTOR_H