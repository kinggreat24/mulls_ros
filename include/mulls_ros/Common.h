/*
 * @Author: your name
 * @Date: 2020-04-09 10:36:44
 * @LastEditTime: 2022-06-06 13:38:27
 * @LastEditors: kinggreat24
 * @Description: In User Settings Edit
 * @FilePath: /mulls_ros/include/mulls_ros/Common.h
 */
// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

#ifndef ORB_SLAM2_COMMON_H
#define ORB_SLAM2_COMMON_H

#include <chrono>
#include <vector>

//PCL
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h> //条件滤波
#include <pcl/filters/voxel_grid.h>          //体素滤波器头文件
#include <pcl/filters/crop_box.h>
#include <pcl/filters/statistical_outlier_removal.h> //统计滤波
#include <pcl/filters/passthrough.h>
#include <pcl/common/common.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h> //NDT(正态分布)配准类头文件

//OpenCV
#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{

    using Time = std::chrono::system_clock::time_point;

    ////////////////////////////////////////////////////////////////////////
    //                            Lego-LOAM TYPES
    /////////////////////////////////////////////////////////////////////////
    typedef struct Cloud_info
    {
        std::vector<int> startRingIndex;
        std::vector<int> endRingIndex;
        float startOrientation;
        float endOrientation;
        float orientationDiff;
        std::vector<bool> segmentedCloudGroundFlag;
        std::vector<unsigned int> segmentedCloudColInd;
        std::vector<float> segmentedCloudRange;
    } Cloud_info;

  
    typedef Eigen::Matrix<double, 4, 1> Vector4d;
    typedef Eigen::Matrix<float, 6, 1> Vector6;
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    typedef Eigen::Matrix<float, 2, 6> Matrix2x6;
    typedef Eigen::Matrix<double, 4, 4> Matrix4d;
    typedef Eigen::Matrix<double, 6, 6> Matrix6d;

    //地面点云提取参数
    typedef struct _patchwork_param_t patchwork_param_t;
    struct _patchwork_param_t
    {
        double sensor_height_;
        bool verbose_;

        int num_iter_;
        int num_lpr_;
        int num_min_pts_;
        double th_seeds_;
        double th_dist_;
        double max_range_;
        double min_range_;
        int num_rings_;
        int num_sectors_;
        double uprightness_thr_;
        double adaptive_seed_selection_margin_;

        // For global threshold
        bool using_global_thr_;
        double global_elevation_thr_;

        int num_zones_;
        std::vector<int> num_sectors_each_zone_;
        std::vector<int> num_rings_each_zone_;
        std::vector<double> min_ranges_;
        std::vector<double> elevation_thr_;
        std::vector<double> flatness_thr_;
    };

    //动态点云滤出的参数
    typedef struct _erasor_param_t erasor_param_t;
    struct _erasor_param_t
    {
        double max_r_;
        double num_rings_;
        int num_sectors_;
        double max_h_;
        double min_h_;
        double th_bin_max_h_;
        double scan_ratio_threshold_;

        int num_lowest_pts_;
        int minimum_num_pts_;
        double rejection_ratio_;
        double th_dist_;
        double iter_groundfilter_;
        double num_lprs_;
        double th_seeds_heights_;
        double map_voxel_size_;

        int version_;

        //距离图参数
        double v_fov_;
        double h_fov_;
        std::vector<float> range_img_resolution_;
    };

    typedef pcl::PointXYZINormal Point_T;
    
    typedef pcl::PointXYZI PointType;

    typedef pcl::PointXYZRGB PointColor;

    // typedef pcl::PointXYZRGBL PointColor;

    typedef struct smoothness_t
    {
        float value;
        size_t ind;
    } smoothness_t;

    struct by_value
    {
        bool operator()(smoothness_t const &left, smoothness_t const &right)
        {
            return left.value < right.value;
        }
    };

    //将上一帧的点投影到当前帧
    inline void TransformToEnd(const Point_T *point_last, const Eigen::Quaterniond &q_cur_last, const Eigen::Vector3d &t_cur_last, Point_T *point_cur)
    {
        Eigen::Vector3d pl(point_last->x, point_last->y, point_last->z);
        Eigen::Vector3d pc = q_cur_last.toRotationMatrix() * pl + t_cur_last;
        point_cur->x = pc[0];
        point_cur->y = pc[1];
        point_cur->z = pc[2];
        point_cur->intensity = point_last->intensity;
    }

    //将当前帧的点投影到上一帧
    inline void TransformToStart(const Point_T &point_cur, const Eigen::Quaterniond &q_last_cur, const Eigen::Vector3d &t_last_cur, Point_T &point_last)
    {
        Eigen::Vector3d pc(point_cur.x, point_cur.y, point_cur.z);
        Eigen::Vector3d pl = q_last_cur.toRotationMatrix() * pc + t_last_cur;
        point_last.x = pl[0];
        point_last.y = pl[1];
        point_last.z = pl[2];
        point_last.intensity = point_cur.intensity;
    }

    inline void TransformPoint(const Point_T *point_cam, const Eigen::Quaterniond &q_wc, const Eigen::Vector3d &t_wc, Point_T *point_world)
    {
        Eigen::Vector3d pc(point_cam->x, point_cam->y, point_cam->z);
        Eigen::Vector3d pw = q_wc.toRotationMatrix() * pc + t_wc;
        point_world->x = pw[0];
        point_world->y = pw[1];
        point_world->z = pw[2];
        point_world->intensity = point_cam->intensity;
    }

    inline void pointAssociateToMap(
        const Point_T *pointOri,
        const Eigen::Quaterniond &q_wc, const Eigen::Vector3d &t_wc,
        Point_T *pointSel)
    {
        Eigen::Vector3d pointOriVec(pointOri->x, pointOri->y, pointOri->z);
        Eigen::Vector3d pointSelVec = q_wc.toRotationMatrix() * pointOriVec + t_wc;
        pointSel->x = pointSelVec[0];
        pointSel->y = pointSelVec[1];
        pointSel->z = pointSelVec[2];
        pointSel->intensity = pointOri->intensity;
    }

    int TransformPointCloud(const pcl::PointCloud<PointType>::Ptr &pointcloud_in, const Eigen::Quaterniond &q, const Eigen::Vector3d &t, pcl::PointCloud<PointType>::Ptr &pointcloud_out);
    int TransformPointCloud(const pcl::PointCloud<PointColor>::Ptr &pointcloud_in, const Eigen::Quaterniond &q, const Eigen::Vector3d &t, pcl::PointCloud<PointColor>::Ptr &pointcloud_out);

    int DepthFilterPointCloud(const pcl::PointCloud<PointType>::Ptr &pointcloud_in, const char *field, const float min, const float max, pcl::PointCloud<PointType>::Ptr &pointcloud_out);
    int DepthFilterPointCloud(const pcl::PointCloud<PointColor>::Ptr &pointcloud_in, const char *field, const float min, const float max, pcl::PointCloud<PointColor>::Ptr &pointcloud_out);

    void SavePointCloudPly(const std::string &file_name, const pcl::PointCloud<PointType>::Ptr pc);
    void SavePointCloudPly(const std::string &file_name, const pcl::PointCloud<PointColor>::Ptr pc);

    /** \brief A standard non-ROS alternative to ros::Time.*/
    // helper function
    inline double toSec(Time::duration duration)
    {
        return std::chrono::duration<double>(duration).count();
    }

    //计算内点阈值
    double compute_inlier_residual_threshold(const std::vector<double> &residuals, const int residual_size, const float m_inlier_ratio);

    inline double compute_inlier_residual_threshold_corner(std::vector<double> residuals, double ratio)
    {
        std::set<double> dis_vec;
        for (size_t i = 0; i < (size_t)(residuals.size() / 3); i++)
        {
            dis_vec.insert(fabs(residuals[3 * i + 0]) + fabs(residuals[3 * i + 1]) + fabs(residuals[3 * i + 2]));
        }
        return *(std::next(dis_vec.begin(), (int)((ratio)*dis_vec.size())));
    }

    inline double compute_inlier_residual_threshold_surf(std::vector<double> residuals, double ratio)
    {
        return *(std::next(residuals.begin(), (int)((ratio)*residuals.size())));
    }

    inline double compute_inlier_residual_threshold_mappoints(std::vector<double> residuals, double ratio)
    {
        std::set<double> dis_vec;
        for (size_t i = 0; i < (size_t)(residuals.size() / 2); i++)
        {
            dis_vec.insert(fabs(residuals[2 * i + 0]) + fabs(residuals[2 * i + 1]));
        }
        return *(std::next(dis_vec.begin(), (int)((ratio)*dis_vec.size())));
    }

    // Calculates rotation matrix to euler angles
    // The result is the same as MATLAB except the order
    // of the euler angles ( x and z are swapped ).
    // The order is roll pitch yaw
    inline Eigen::Vector3d rotationMatrixToEulerAngles(Eigen::Matrix3d &R)
    {
        // assert(isRotationMatrix(R));
        float sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));

        bool singular = sy < 1e-6; // If

        float x, y, z;
        if (!singular)
        {
            x = atan2(R(2, 1), R(2, 2));
            y = atan2(-R(2, 0), sy);
            z = atan2(R(1, 0), R(0, 0));
        }
        else
        {
            x = atan2(-R(1, 2), R(1, 1));
            y = atan2(-R(2, 0), sy);
            z = 0;
        }
        return Eigen::Vector3d(x, y, z);
    }

    inline Eigen::Matrix3d eulerAnglesToRotationMatrix(Eigen::Vector3d &theta_rpy, bool use_axis = false)
    {
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        if (!use_axis)
        {
            // Calculate rotation about x axis
            Eigen::Matrix3d R_x = Eigen::Matrix3d::Identity();
            R_x << 1, 0, 0,
                0, cos(theta_rpy[0]), -sin(theta_rpy[0]),
                0, sin(theta_rpy[0]), cos(theta_rpy[0]);

            // Calculate rotation about y axis
            Eigen::Matrix3d R_y = Eigen::Matrix3d::Identity();
            R_y << cos(theta_rpy[1]), 0, sin(theta_rpy[1]),
                0, 1, 0,
                -sin(theta_rpy[1]), 0, cos(theta_rpy[1]);

            // Calculate rotation about z axis
            Eigen::Matrix3d R_z = Eigen::Matrix3d::Identity();
            R_z << cos(theta_rpy[2]), -sin(theta_rpy[2]), 0,
                sin(theta_rpy[2]), cos(theta_rpy[2]), 0,
                0, 0, 1;

            // Combined rotation matrix
            R = R_z * R_y * R_x;
        }
        else
        {
            Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(theta_rpy[0], Eigen::Matrix<double, 3, 1>::UnitX()));
            Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(theta_rpy[1], Eigen::Matrix<double, 3, 1>::UnitY()));
            Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(theta_rpy[2], Eigen::Matrix<double, 3, 1>::UnitZ()));

            R = yawAngle * pitchAngle * rollAngle;
        }

        return R;
    }

    //Line

    // 比较线特征距离的两种方式，自己添加的
    struct compare_descriptor_by_NN_dist
    {
        inline bool operator()(const std::vector<cv::DMatch> &a, const std::vector<cv::DMatch> &b)
        {
            return (a[0].distance < b[0].distance);
        }
    };

    struct conpare_descriptor_by_NN12_dist
    {
        inline bool operator()(const std::vector<cv::DMatch> &a, const std::vector<cv::DMatch> &b)
        {
            return ((a[1].distance - a[0].distance) > (b[1].distance - b[0].distance));
        }
    };

    // 按描述子之间距离的从小到大方式排序
    struct sort_descriptor_by_queryIdx
    {
        inline bool operator()(const std::vector<cv::DMatch> &a, const std::vector<cv::DMatch> &b)
        {
            return (a[0].queryIdx < b[0].queryIdx);
        }
    };

    inline cv::Mat SkewSymmetricMatrix(const cv::Mat &v)
    {
        return (cv::Mat_<float>(3, 3) << 0, -v.at<float>(2), v.at<float>(1),
                v.at<float>(2), 0, -v.at<float>(0),
                -v.at<float>(1), v.at<float>(0), 0);
    }

    inline Eigen::Matrix3d SkewSymmetricMatrix(const Eigen::Vector3d &v)
    {
        Eigen::Matrix3d SkewMatrix = Eigen::Matrix3d::Zero();
        SkewMatrix << 0.0, -v.z(), v.y(),
            v.z(), 0, -v.x(),
            -v.y(), v.x(), 0.0;
        return SkewMatrix;
    }

    /**
 * @brief 求一个vector数组的中位数绝对偏差MAD
 * 中位数绝对偏差MAD——median absolute deviation, 是单变量数据集中样本差异性的稳健度量。
 * MAD是一个健壮的统计量，对于数据集中异常值的处理比标准差更具有弹性，可以大大减少异常值对于数据集的影响
 * 对于单变量数据集 X={X1,X2,X3,...,Xn}, MAD的计算公式为：MAD(X)=median(|Xi-median(X)|)
 * @param residues
 * @return
 */
    inline double vector_mad(std::vector<double> residues)
    {
        if (residues.size() != 0)
        {
            // Return the standard deviation of vector with MAD estimation
            int n_samples = residues.size();
            std::sort(residues.begin(), residues.end());
            double median = residues[n_samples / 2];
            for (int i = 0; i < n_samples; i++)
                residues[i] = fabs(residues[i] - median);
            std::sort(residues.begin(), residues.end());
            double MAD = residues[n_samples / 2];
            return 1.4826 * MAD;
        }
        else
            return 0.0;
    }

    inline double robustWeightCauchy(double norm_res)
    {
        // Cauchy
        return 1.0 / (1.0 + norm_res * norm_res);

        // Smooth Truncated Parabola
        /*if( norm_res <= 1.0 )
        return 1.0 - norm_res * norm_res;
    else
        return 0.0;*/

        // Tukey's biweight
        /*if( norm_res <= 1.0 )
        return pow( 1.0 - norm_res*norm_res ,2.0);
    else
        return 0.0;*/

        // Huber loss function
        /*if( norm_res <= 1.0 )
        return 1.0;
    else
        return 1.0 / norm_res;*/

        // Welsch
        //return exp( - norm_res*norm_res );
    }

    inline bool is_finite(const Eigen::MatrixXd x)
    {
        return ((x - x).array() == (x - x).array()).all();
    }

    inline bool is_nan(const Eigen::MatrixXd x)
    {
        for (unsigned int i = 0; i < x.rows(); i++)
        {
            for (unsigned int j = 0; j < x.cols(); j++)
            {
                if (std::isnan(x(i, j)))
                    return true;
            }
        }
        return false;
    }

    inline Eigen::Matrix3d skew(Eigen::Vector3d v)
    {
        Eigen::Matrix3d skew;

        skew(0, 0) = 0;
        skew(1, 1) = 0;
        skew(2, 2) = 0;

        skew(0, 1) = -v(2);
        skew(0, 2) = v(1);
        skew(1, 2) = -v(0);

        skew(1, 0) = v(2);
        skew(2, 0) = -v(1);
        skew(2, 1) = v(0);

        return skew;
    }

    inline Eigen::Vector3d skewcoords(Eigen::Matrix3d M)
    {
        Eigen::Vector3d skew;
        skew << M(2, 1), M(0, 2), M(1, 0);
        return skew;
    }

    inline Eigen::Matrix3d skewlog(Eigen::Matrix3d M)
    {
        Eigen::Matrix3d skew;
        double val = (M.trace() - 1.f) / 2.f;
        if (val > 1.f)
            val = 1.f;
        else if (val < -1.f)
            val = -1.f;
        double theta = acos(val);
        if (theta == 0.f)
            skew << 0, 0, 0, 0, 0, 0, 0, 0, 0;
        else
            skew << (M - M.transpose()) / (2.f * sin(theta)) * theta;
        return skew;
    }

    inline Eigen::Matrix4d inverse_se3(Eigen::Matrix4d T)
    {
        Eigen::Matrix4d Tinv = Eigen::Matrix4d::Identity();
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        t = T.block(0, 3, 3, 1);
        R = T.block(0, 0, 3, 3);
        Tinv.block(0, 0, 3, 3) = R.transpose();
        Tinv.block(0, 3, 3, 1) = -R.transpose() * t;
        return Tinv;
    }

    inline Eigen::Matrix4d expmap_se3(Vector6d x)
    {
        Eigen::Matrix3d R, V, s, I = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t, w;
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        w = x.tail(3);
        t = x.head(3);
        double theta = w.norm();
        if (theta < 0.000001)
            R = I;
        else
        {
            s = skew(w) / theta;
            R = I + s * sin(theta) + s * s * (1.0f - cos(theta));
            V = I + s * (1.0f - cos(theta)) / theta + s * s * (theta - sin(theta)) / theta;
            t = V * t;
        }
        T.block(0, 0, 3, 4) << R, t;
        return T;
    }

    inline Vector6d logmap_se3(Eigen::Matrix4d T)
    {
        Eigen::Matrix3d R, Id3 = Eigen::Matrix3d::Identity();
        Eigen::Vector3d Vt, t, w;
        Eigen::Matrix3d V = Eigen::Matrix3d::Identity(), w_hat = Eigen::Matrix3d::Zero();
        Vector6d x;
        Vt << T(0, 3), T(1, 3), T(2, 3);
        w << 0.f, 0.f, 0.f;
        R = T.block(0, 0, 3, 3);
        double cosine = (R.trace() - 1.f) / 2.f;
        if (cosine > 1.f)
            cosine = 1.f;
        else if (cosine < -1.f)
            cosine = -1.f;
        double sine = sqrt(1.0 - cosine * cosine);
        if (sine > 1.f)
            sine = 1.f;
        else if (sine < -1.f)
            sine = -1.f;
        double theta = acos(cosine);
        if (theta > 0.000001)
        {
            w_hat = theta * (R - R.transpose()) / (2.f * sine);
            w = skewcoords(w_hat);
            Eigen::Matrix3d s;
            s = skew(w) / theta;
            V = Id3 + s * (1.f - cosine) / theta + s * s * (theta - sine) / theta;
        }
        t = V.inverse() * Vt;
        x.head(3) = t;
        x.tail(3) = w;
        return x;
    }

    inline double diffManifoldError(Eigen::Matrix4d T1, Eigen::Matrix4d T2)
    {
        return (logmap_se3(T1) - logmap_se3(T2)).norm();
    }

    inline bool isGoodSolution(Eigen::Matrix4d DT, Matrix6d DTcov, double err)
    {
        Eigen::SelfAdjointEigenSolver<Matrix6d> eigensolver(DTcov);
        Vector6d DT_cov_eig = eigensolver.eigenvalues();

        if (DT_cov_eig(0) < 0.0 || DT_cov_eig(5) > 1.0 || err < 0.0 || err > 1.0 || !is_finite(DT))
        {
            cout << endl
                 << DT_cov_eig(0) << "\t" << DT_cov_eig(5) << "\t" << err << endl;
            return false;
        }

        return true;
    }

    inline void vector_mean_stdv_mad(std::vector<double> residues, double &mean, double &stdv)
    {
        mean = 0.f;
        stdv = 0.f;

        if (residues.size() != 0)
        {
            // Return the standard deviation of vector with MAD estimation
            int n_samples = residues.size();
            std::vector<double> residues_ = residues;
            sort(residues.begin(), residues.end());
            double median = residues[n_samples / 2];
            for (int i = 0; i < n_samples; i++)
                residues[i] = fabsf(residues[i] - median);
            std::sort(residues.begin(), residues.end());
            stdv = 1.4826 * residues[n_samples / 2];

            // return the mean with only the best samples
            int k = 0;
            for (int i = 0; i < n_samples; i++)
                if (residues_[i] < 2.0 * stdv)
                {
                    mean += residues_[i];
                    k++;
                }

            if (k >= int(0.2 * residues.size()))
                mean /= double(k);
            else
            {
                k = 0;
                mean = 0.f;
                for (int i = 0; i < n_samples; i++)
                {
                    mean += residues_[i];
                    k++;
                }
                mean /= double(k);
            }
        }
    }

    inline void getTransformFromSe3(const Eigen::Matrix<double, 6, 1> &se3, Eigen::Quaterniond &q, Eigen::Vector3d &t)
    {
        Eigen::Vector3d omega(se3.data());
        Eigen::Vector3d upsilon(se3.data() + 3);
        Eigen::Matrix3d Omega = skew(omega);

        double theta = omega.norm();
        double half_theta = 0.5 * theta;

        double imag_factor;
        double real_factor = cos(half_theta);
        if (theta < 1e-10)
        {
            double theta_sq = theta * theta;
            double theta_po4 = theta_sq * theta_sq;
            imag_factor = 0.5 - 0.0208333 * theta_sq + 0.000260417 * theta_po4;
        }
        else
        {
            double sin_half_theta = sin(half_theta);
            imag_factor = sin_half_theta / theta;
        }

        q = Eigen::Quaterniond(real_factor, imag_factor * omega.x(), imag_factor * omega.y(), imag_factor * omega.z());

        Eigen::Matrix3d J;
        if (theta < 1e-10)
        {
            J = q.matrix();
        }
        else
        {
            Eigen::Matrix3d Omega2 = Omega * Omega;
            J = (Eigen::Matrix3d::Identity() + (1 - cos(theta)) / (theta * theta) * Omega + (theta - sin(theta)) / (pow(theta, 3)) * Omega2);
        }

        t = J * upsilon;
    }

} // namespace ORB_SLAM2

#endif // ORB_SLAM2_COMMON_H
