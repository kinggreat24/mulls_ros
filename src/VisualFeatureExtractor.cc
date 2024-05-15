/*
 * @Author: kinggreat24
 * @Date: 2022-06-01 21:29:39
 * @LastEditTime: 2022-09-16 15:29:26
 * @LastEditors: kinggreat24 kinggreat24@whu.edu.cn
 * @Description: 
 * @FilePath: /mulls_ros/src/VisualFeatureExtractor.cc
 * 可以输入预定的版权声明、个性签名、空行等
 */
#include "VisualFeatureExtractor.h"

namespace ORB_SLAM2
{
    bool VisualFeatureExtractor::mbInitialComputations = true;
    float VisualFeatureExtractor::mnMinX, VisualFeatureExtractor::mnMinY, VisualFeatureExtractor::mnMaxX, VisualFeatureExtractor::mnMaxY;
    float VisualFeatureExtractor::mfGridElementWidthInv, VisualFeatureExtractor::mfGridElementHeightInv;

    VisualFeatureExtractor::VisualFeatureExtractor(const lo::orb_params_t &orb_params,
                                                   const lo::tracker_t &dvl_tracking_params,
                                                   vk::PinholeCamera *pinhole_camera_,
                                                   lo::CFilter<Point_T> *cfilter,
                                                   Eigen::Matrix4d &Tcamlidar,
                                                   const std::string &voc_file)
        : tracker_params_(dvl_tracking_params), mpPinholeCamModel(pinhole_camera_), mpCfilter(cfilter), mTcam_lidar(Tcamlidar)
    {
        int nFeatures = orb_params.n_features;
        float scale_factor = orb_params.scale_factor;
        int nLevels = orb_params.nLevels;
        int ini_ThFast = orb_params.iniThFast;
        int min_ThFast = orb_params.minThFast;
        mpORBExtractor = new ORB_SLAM2::ORBextractor(nFeatures, scale_factor, nLevels, ini_ThFast, min_ThFast);

        mpVocabulary = new ORB_SLAM2::ORBVocabulary();
        mpVocabulary->load_frombin(voc_file);

        mK = cv::Mat::zeros(3, 3, CV_32F);
        mK.at<float>(0, 0) = mpPinholeCamModel->fx();
        mK.at<float>(0, 2) = mpPinholeCamModel->cx();
        mK.at<float>(1, 1) = mpPinholeCamModel->fy();
        mK.at<float>(1, 2) = mpPinholeCamModel->cy();
        mK.at<float>(2, 2) = 1.0;

        std::cout << "mK: " << std::endl
                  << mK << std::endl;

        mDistCoef = cv::Mat::zeros(1, 5, CV_32F);
        mDistCoef.at<float>(0) = mpPinholeCamModel->d0();
        mDistCoef.at<float>(1) = mpPinholeCamModel->d1();
        mDistCoef.at<float>(2) = mpPinholeCamModel->d2();
        mDistCoef.at<float>(3) = mpPinholeCamModel->d3();
        mDistCoef.at<float>(4) = mpPinholeCamModel->d4();

        std::cout << "mDistCoef: " << mDistCoef << std::endl;
    }

    void VisualFeatureExtractor::ExtractORB(const cv::Mat &imGray, lo::cloudblock_Ptr &clockcloud_ptr)
    {
        (*mpORBExtractor)(imGray, cv::Mat(), clockcloud_ptr->mvKeys, clockcloud_ptr->mDescriptors);
        clockcloud_ptr->NP = clockcloud_ptr->mvKeys.size();
        ComputeBoW(clockcloud_ptr);
    }

    void VisualFeatureExtractor::ImagePreprocessing(const cv::Mat &imGray, lo::cloudblock_Ptr &clockcloud_ptr)
    {
        // Scale Level Info
        // clockcloud_ptr->mnScaleLevels = mpORBExtractor->GetLevels();
        // clockcloud_ptr->mfScaleFactor = mpORBExtractor->GetScaleFactor();
        // clockcloud_ptr->mfLogScaleFactor = log(clockcloud_ptr->mfScaleFactor);
        // clockcloud_ptr->mvScaleFactors = mpORBExtractor->GetScaleFactors();
        // clockcloud_ptr->mvInvScaleFactors = mpORBExtractor->GetInverseScaleFactors();
        // clockcloud_ptr->mvLevelSigma2 = mpORBExtractor->GetScaleSigmaSquares();
        // clockcloud_ptr->mvInvLevelSigma2 = mpORBExtractor->GetInverseScaleSigmaSquares();

        //图像金字塔生成
        cv::Mat original_img_ = imGray.clone();
        original_img_.convertTo(original_img_, CV_32FC1, 1.0 / 255);
        clockcloud_ptr->mvImgPyramid.resize(tracker_params_.levels);
        clockcloud_ptr->mvImgPyramid[0] = original_img_;
        for (int i = 1; i < tracker_params_.levels; i++)
        {
            clockcloud_ptr->mvImgPyramid[i] = cv::Mat(cv::Size(clockcloud_ptr->mvImgPyramid[i - 1].size().width / 2, clockcloud_ptr->mvImgPyramid[i - 1].size().height / 2), clockcloud_ptr->mvImgPyramid[i - 1].type());
            PyrDownMeanSmooth<float>(clockcloud_ptr->mvImgPyramid[i - 1], clockcloud_ptr->mvImgPyramid[i]);
        }

        // This is done only for the first Frame (or after a change in the calibration)
        if (VisualFeatureExtractor::mbInitialComputations)
        {
            ComputeImageBounds(imGray);

            VisualFeatureExtractor::mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(VisualFeatureExtractor::mnMaxX - VisualFeatureExtractor::mnMinX);
            VisualFeatureExtractor::mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(VisualFeatureExtractor::mnMaxY - VisualFeatureExtractor::mnMinY);
            VisualFeatureExtractor::mbInitialComputations = false;
        }

        //激光点云特征处理
        // pcTPtr pc_raw_filter(new pcT());
        // mpCfilter->pointcloud_depthfilter(clockcloud_ptr->pc_raw, pc_raw_filter, "x", 5, 80);

        // pcTPtr pc_raw_cam(new pcT());
        // pcl::transformPointCloudWithNormals(*pc_raw_filter, *pc_raw_cam, mTcam_lidar);

        // //激光点采样
        // LidarPointsSampling(original_img_, pc_raw_cam, clockcloud_ptr);
    }

    void VisualFeatureExtractor::UndistortKeyPoints(lo::cloudblock_Ptr &clockcloud_ptr)
    {
        if (mDistCoef.at<float>(0) == 0.0)
        {
            clockcloud_ptr->mvKeysUn = clockcloud_ptr->mvKeys;
            return;
        }

        // Fill matrix with points
        cv::Mat mat(clockcloud_ptr->NP, 2, CV_32F);
        for (int i = 0; i < clockcloud_ptr->NP; i++)
        {
            mat.at<float>(i, 0) = clockcloud_ptr->mvKeys[i].pt.x;
            mat.at<float>(i, 1) = clockcloud_ptr->mvKeys[i].pt.y;
        }

        // Undistort points
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

        // Fill undistorted keypoint vector
        clockcloud_ptr->mvKeysUn.resize(clockcloud_ptr->NP);
        for (int i = 0; i < clockcloud_ptr->NP; i++)
        {
            cv::KeyPoint kp = clockcloud_ptr->mvKeys[i];
            kp.pt.x = mat.at<float>(i, 0);
            kp.pt.y = mat.at<float>(i, 1);
            clockcloud_ptr->mvKeysUn[i] = kp;
        }
    }

    void VisualFeatureExtractor::ComputeImageBounds(const cv::Mat &imLeft)
    {
        if (mDistCoef.at<float>(0) != 0.0)
        {
            cv::Mat mat(4, 2, CV_32F);
            mat.at<float>(0, 0) = 0.0;
            mat.at<float>(0, 1) = 0.0;
            mat.at<float>(1, 0) = imLeft.cols;
            mat.at<float>(1, 1) = 0.0;
            mat.at<float>(2, 0) = 0.0;
            mat.at<float>(2, 1) = imLeft.rows;
            mat.at<float>(3, 0) = imLeft.cols;
            mat.at<float>(3, 1) = imLeft.rows;

            // Undistort corners
            mat = mat.reshape(2);
            cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
            mat = mat.reshape(1);

            VisualFeatureExtractor::mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
            VisualFeatureExtractor::mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
            VisualFeatureExtractor::mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
            VisualFeatureExtractor::mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1));
        }
        else
        {
            VisualFeatureExtractor::mnMinX = 0.0f;
            VisualFeatureExtractor::mnMaxX = imLeft.cols;
            VisualFeatureExtractor::mnMinY = 0.0f;
            VisualFeatureExtractor::mnMaxY = imLeft.rows;
        }
    }

    void VisualFeatureExtractor::AssignFeaturesToGrid(lo::cloudblock_Ptr &clockcloud_ptr)
    {
        int nReserve = 0.5f * clockcloud_ptr->NP / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
        for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
            for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
                clockcloud_ptr->mGrid[i][j].reserve(nReserve);

        for (int i = 0; i < clockcloud_ptr->NP; i++)
        {
            const cv::KeyPoint &kp = clockcloud_ptr->mvKeysUn[i];
            int nGridPosX, nGridPosY;
            if (PosInGrid(kp, nGridPosX, nGridPosY))
                clockcloud_ptr->mGrid[nGridPosX][nGridPosY].push_back(i);
        }
    }

    bool VisualFeatureExtractor::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
    {
        posX = round((kp.pt.x - VisualFeatureExtractor::mnMinX) * VisualFeatureExtractor::mfGridElementWidthInv);
        posY = round((kp.pt.y - VisualFeatureExtractor::mnMinY) * VisualFeatureExtractor::mfGridElementHeightInv);

        //Keypoint's coordinates are undistorted, which could cause to go out of the image
        if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
            return false;

        return true;
    }

    void VisualFeatureExtractor::ComputeBoW(lo::cloudblock_Ptr &clockcloud_ptr)
    {
        if (clockcloud_ptr->mBowVec.empty())
        {
            std::vector<cv::Mat> vCurrentDesc = lo::toDescriptorVector(clockcloud_ptr->mDescriptors);
            mpVocabulary->transform(vCurrentDesc, clockcloud_ptr->mBowVec, clockcloud_ptr->mFeatVec, 4);
        }
    }

    void VisualFeatureExtractor::LidarPointsSampling(const cv::Mat &im, const pcTPtr &pc_camera, lo::cloudblock_Ptr &clockcloud_ptr)
    {
        //激光点投影到图像
        int num_bucket_size = 20;
        std::vector<std::pair<float, Point_T>> mag_point_bucket;
        mag_point_bucket.reserve(num_bucket_size);
        clockcloud_ptr->mpMagLidarPointCloud.reset(new pcT());

        for (auto pt = pc_camera->begin(); pt != pc_camera->end(); pt++)
        {
            Eigen::Vector3d xyz(pt->x, pt->y, pt->z);
            if (pt->z < 0)
                continue;

            Eigen::Vector2d uv = mpPinholeCamModel->world2cam(xyz);
            int u = static_cast<int>(uv(0));
            int v = static_cast<int>(uv(1));

            if (mpPinholeCamModel->isInFrame(Eigen::Vector2i(u, v), 4))
            {
                float dx = 0.5f * (im.at<float>(v, u + 1) - im.at<float>(v, u - 1));
                float dy = 0.5f * (im.at<float>(v + 1, u) - im.at<float>(v - 1, u));

                std::pair<float, Point_T> mag_point;
                mag_point = make_pair((dx * dx + dy * dy), (*pt));

                mag_point_bucket.push_back(mag_point);
                if (mag_point_bucket.size() == num_bucket_size)
                {

                    float max = -1;
                    int idx;
                    for (int i = 0; i < mag_point_bucket.size(); ++i)
                    {
                        if (mag_point_bucket[i].first > max)
                        {
                            max = mag_point_bucket[i].first;
                            idx = i;
                        }
                    }

                    if (max > (6.25 / (255.0 * 255.0))) // 16.25
                        clockcloud_ptr->mpMagLidarPointCloud->push_back(mag_point_bucket[idx].second);

                    mag_point_bucket.clear();
                }
            }
        }
    }

    void VisualFeatureExtractor::ShowLidarSamplePoints(cv::Mat &image_out, lo::cloudblock_Ptr &clockcloud_ptr, const int step)
    {
        // std::cout << "mnMinX: " << mnMinX << " mnMaxX: " << mnMaxX << " mnMinY: " << mnMinY << " mnMaxY: " << mnMaxY << std::endl;
        if (image_out.channels() == 1)
            cvtColor(image_out, image_out, cv::COLOR_GRAY2BGR);

        const int num_level = 0;
        const float scale = 1.0f / (1 << num_level);

        float v_min = 5.0;
        float v_max = 80.0;
        float dv = v_max - v_min;

        int n = 0;
        for (auto iter = clockcloud_ptr->mpMagLidarPointCloud->begin(); iter != clockcloud_ptr->mpMagLidarPointCloud->end(); ++iter)
        {
            n++;
            if (n % step != 0)
                continue;

            int u_ref_i = 0;
            int v_ref_i = 0;

            if (iter->z < 0)
                continue;

            double inv_z = 1.0 / iter->z;
            double u_ = mpPinholeCamModel->fx() * iter->x * inv_z + mpPinholeCamModel->cx();
            double v_ = mpPinholeCamModel->fy() * iter->y * inv_z + mpPinholeCamModel->cy();

            if (u_ < mnMinX || u_ >= mnMaxX || v_ < mnMinY || v_ >= mnMaxY)
                continue;

            u_ref_i = static_cast<int>(u_);
            v_ref_i = static_cast<int>(v_);

            float v = iter->z;
            float r = 1.0;
            float g = 1.0;
            float b = 1.0;
            if (v < v_min)
                v = v_min;
            if (v > v_max)
                v = v_max;

            if (v < v_min + 0.25 * dv)
            {
                r = 0.0;
                g = 4 * (v - v_min) / dv;
            }
            else if (v < (v_min + 0.5 * dv))
            {
                r = 0.0;
                b = 1 + 4 * (v_min + 0.25 * dv - v) / dv;
            }
            else if (v < (v_min + 0.75 * dv))
            {
                r = 4 * (v - v_min - 0.5 * dv) / dv;
                b = 0.0;
            }
            else
            {
                g = 1 + 4 * (v_min + 0.75 * dv - v) / dv;
                b = 0.0;
            }

            cv::circle(image_out, cv::Point(u_ref_i, v_ref_i), 2.5, 255 * cv::Scalar(r, g, b), -1);
        }
    }

}