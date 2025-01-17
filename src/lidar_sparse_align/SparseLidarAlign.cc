/*
 * @Author: Kinggreat24
 * @Date: 2020-08-13 11:19:31
 * @LastEditors: kinggreat24
 * @LastEditTime: 2022-05-31 15:47:20
 * @Description: 
 */
#include "lidar_sparse_align/SparseLidarAlign.h"

namespace lo
{

    SparseLidarAlign::SparseLidarAlign(const vk::PinholeCamera *pinhole_model, const lo::_tracker_t &tracker_info)
        : pinhole_model_(pinhole_model), tracker_info_(tracker_info)
    {
        // tracker_info_ = dvlo::Config::cfg()->tracker();
        min_level_ = tracker_info_.min_level;
        max_level_ = tracker_info_.max_level;
        use_weight_scale_ = tracker_info_.use_weight_scale;
        set_weightfunction();
    }

    SparseLidarAlign::~SparseLidarAlign()
    {
    }

    void SparseLidarAlign::set_weightfunction()
    {
        if (tracker_info_.use_weight_scale)
        {
            switch (tracker_info_.scale_estimator_type)
            {
            case ScaleEstimatorType::TDistributionScale:
                scale_estimator_.reset(new TDistributionScaleEstimator());
                break;
            default:
                cerr << "Do not use scale estimator." << endl;
            }
        }

        switch (tracker_info_.weight_function_type)
        {
        case WeightFunctionType::TDistributionWeight:
            weight_function_.reset(new TDistributionWeightFunction());
            break;
        default:
            cerr << "Do not use weight function." << endl;
        }
    }

    bool SparseLidarAlign::tracking(cloudblock_Ptr pReference, cloudblock_Ptr pCurrent, Sophus::SE3 &transformation)
    {
        bool status = true;

        pointcloud_ref_    = pReference->pointcloud();
        ref_image_pyramid_ = pReference->mvImgPyramid;
        cur_image_pyramid_ = pCurrent->mvImgPyramid;

        affine_a_ = 1.0f;
        affine_b_ = 0.0f;

        for (current_level_ = max_level_; current_level_ >= min_level_; current_level_--)
        {
            is_precomputed_ = false;
            stop_ = false;
            optimize(transformation);
        }

        return status;
    }

    void SparseLidarAlign::startIteration() {}
    void SparseLidarAlign::finishIteration() {}


    void SparseLidarAlign::precompute_patches(cv::Mat &img, pcl::PointCloud<Point_T> &pointcloud, cv::Mat &patch_buf, bool is_derivative)
    {
        const int border = patch_halfsize_ + 2 + 2;
        const int stride = img.cols;
        const float scale = 1.0f / (1 << current_level_);

        std::vector<Eigen::Vector2d> uv_set;
        for (auto pt = pointcloud.begin(); pt != pointcloud.end(); pt++)
        {
            Eigen::Vector3d xyz(pt->x, pt->y, pt->z);
            Eigen::Vector2d uv = scale * pinhole_model_->world2cam(xyz);
            uv_set.push_back(uv);
        }

        patch_buf = cv::Mat(pointcloud.size(), pattern_length_, CV_32F);

        //是否已经计算过导数
        if (is_derivative)
        {
            dI_buf_.resize(Eigen::NoChange, patch_buf.rows * pattern_length_);
            dI_buf_.setZero();

            jacobian_buf_.resize(Eigen::NoChange, patch_buf.rows * pattern_length_);
            jacobian_buf_.setZero();
        }

        auto pc_iter = pointcloud.begin();
        size_t point_counter = 0;

        for (auto uv_iter = uv_set.begin(); uv_iter != uv_set.end(); ++uv_iter, ++pc_iter, ++point_counter)
        {
            Eigen::Vector2d &uv = *uv_iter;
            float u_f = uv(0);
            float v_f = uv(1);
            const int u_i = static_cast<int>(u_f);
            const int v_i = static_cast<int>(v_f);

            if (u_i - border < 0 || u_i + border > img.cols || v_i - border < 0 || v_i + border > img.rows || pc_iter->z <= 0.0)
            {
                float *patch_buf_ptr = reinterpret_cast<float *>(patch_buf.data) + pattern_length_ * point_counter;
                for (int i = 0; i < pattern_length_; ++i, ++patch_buf_ptr)
                    *patch_buf_ptr = std::numeric_limits<float>::quiet_NaN();
                continue;
            }

            const float subpix_u = u_f - u_i;
            const float subpix_v = v_f - v_i;
            const float w_tl = (1.0 - subpix_u) * (1.0 - subpix_v);
            const float w_tr = subpix_u * (1.0 - subpix_v);
            const float w_bl = (1.0 - subpix_u) * subpix_v;
            const float w_br = subpix_u * subpix_v;

            size_t pixel_counter = 0;

            float *patch_buf_ptr = reinterpret_cast<float *>(patch_buf.data) + pattern_length_ * point_counter;

            for (int i = 0; i < pattern_length_; ++i, ++pixel_counter, ++patch_buf_ptr)
            {
                int x = pattern_[i][0];
                int y = pattern_[i][1];

                float *img_ptr = (float *)img.data + (v_i + y) * stride + (u_i + x);
                *patch_buf_ptr = w_tl * img_ptr[0] + w_tr * img_ptr[1] + w_bl * img_ptr[stride] + w_br * img_ptr[stride + 1];

                if (is_derivative)
                {
                    // precompute image gradient
                    float dx = 0.5f * ((w_tl * img_ptr[1] + w_tr * img_ptr[2] + w_bl * img_ptr[stride + 1] + w_br * img_ptr[stride + 2]) - (w_tl * img_ptr[-1] + w_tr * img_ptr[0] + w_bl * img_ptr[stride - 1] + w_br * img_ptr[stride]));
                    float dy = 0.5f * ((w_tl * img_ptr[stride] + w_tr * img_ptr[1 + stride] + w_bl * img_ptr[stride * 2] + w_br * img_ptr[stride * 2 + 1]) - (w_tl * img_ptr[-stride] + w_tr * img_ptr[1 - stride] + w_bl * img_ptr[0] + w_br * img_ptr[1]));

                    Matrix2x6 frame_jac;
                    Eigen::Vector3f xyz(pc_iter->x, pc_iter->y, pc_iter->z);
                    cloudblock_t::jacobian_xyz2uv(xyz, frame_jac);

                    Eigen::Vector2f dI_xy(dx, dy);
                    dI_buf_.col(point_counter * pattern_length_ + i) = dI_xy;
                    jacobian_buf_.col(point_counter * pattern_length_ + pixel_counter) = (dx * pinhole_model_->fx() * frame_jac.row(0) + dy * pinhole_model_->fy() * frame_jac.row(1)) / (1 << current_level_);
                }
            }
        }
    }

    double SparseLidarAlign::compute_residuals(const Sophus::SE3 &transformation)
    {
        errors_.clear();
        J_.clear();
        weight_.clear();

        if (!is_precomputed_)
        {
            cv::Mat reference_img = ref_image_pyramid_[current_level_].clone();
            precompute_patches(reference_img, pointcloud_ref_, ref_patch_buf_, true);
            is_precomputed_ = true;
        }

        // cv::Mat &current_img = current_->level(current_level_);
        cv::Mat current_img = cur_image_pyramid_[current_level_].clone();
        pcl::PointCloud<Point_T> pointcloud_cur;
        pcl::transformPointCloud(pointcloud_ref_, pointcloud_cur, transformation.matrix());
        precompute_patches(current_img, pointcloud_cur, cur_patch_buf_, false);

        cv::Mat errors = cv::Mat(pointcloud_cur.size(), pattern_length_, CV_32F);
        //errors = cur_patch_buf_ - (affine_a_ * ref_patch_buf_ + affine_b_);
        errors = cur_patch_buf_ - ref_patch_buf_;

        scale_ = scale_estimator_->compute(errors);

        float chi2 = 0.0f;
        n_measurement_ = 0;

        float *errors_ptr = errors.ptr<float>();
        float *ref_patch_buf_ptr = ref_patch_buf_.ptr<float>();
        float *cur_patch_buf_ptr = cur_patch_buf_.ptr<float>();

        float IiIj = 0.0f;
        float IiIi = 0.0f;
        float sum_Ii = 0.0f;
        float sum_Ij = 0.0f;

        for (int i = 0; i < errors.size().area(); ++i, ++errors_ptr, ++ref_patch_buf_ptr, ++cur_patch_buf_ptr)
        {

            float &res = *errors_ptr;

            float &Ii = *ref_patch_buf_ptr;
            float &Ij = *cur_patch_buf_ptr;

            if (std::isfinite(res))
            {

                n_measurement_++;

                Vector6f J(jacobian_buf_.col(i));

                errors_.push_back(res);
                J_.push_back(J);

                IiIj += Ii * Ij;
                IiIi += Ii * Ii;
                sum_Ii += Ii;
                sum_Ij += Ij;
            }
        }

        affine_a_ = IiIj / IiIi;
        affine_b_ = (sum_Ij - affine_a_ * sum_Ii) / n_measurement_;

        vector<float> sorted_errors;
        sorted_errors.resize(errors_.size());
        copy(errors_.begin(), errors_.end(), sorted_errors.begin());
        sort(sorted_errors.begin(), sorted_errors.end());

        float median_mu = sorted_errors[sorted_errors.size() / 2];

        std::vector<float> absolute_res_error;
        for (auto error : errors_)
        {
            absolute_res_error.push_back(fabs(error - median_mu));
        }
        sort(absolute_res_error.begin(), absolute_res_error.end());
        float median_abs_deviation = 1.4826 * absolute_res_error[absolute_res_error.size() / 2];

        for (auto error : errors_)
        {
            float weight = 1.0;
            weight = weight_function_->weight((error - median_mu) / median_abs_deviation);
            weight_.push_back(weight);

            chi2 += error * error * weight;
        }
        return chi2 / n_measurement_;
    }

    // implementation for LSQNonlinear class
    void SparseLidarAlign::update(const ModelType &old_model, ModelType &new_model)
    {
        Eigen::Matrix<double, 6, 1> update_;
        for (int i = 0; i < 6; i++)
            update_[i] = -x_[i];
        // new_model = old_model * Sophus::SE3::exp(-x_);
        new_model = old_model * Sophus::SE3::exp(update_);
    }

    void SparseLidarAlign::max_level(int level) { max_level_ = level; }

    double SparseLidarAlign::build_LinearSystem(Sophus::SE3 &model)
    {
        double res = compute_residuals(model);

        //是否显示残差图像
        // if(display_)
        //     resimg_ = cv::Mat(current_->level(current_level_).size(), CV_32F, cv::Scalar(0));

        H_.setZero();
        Jres_.setZero();

        for (int i = 0; i < errors_.size(); ++i)
        {
            float &res = errors_[i];
            Vector6f &J = J_[i];
            float &weight = weight_[i];

            H_.noalias() += J * J.transpose() * weight;
            Jres_.noalias() -= J * res * weight;

            // if(display_)
            //     resimg_.at<float>((int) v_cur + y - patch_halfsize_, (int) u_cur + x - patch_halfsize_) =
            //             res / 255.0;
        }

        // std::cout<<"Hessian matrix: "<<std::endl<<H_.matrix()<<std::endl;

        return res;
    }

} // namespace lo