/*
 * @Author: kinggreat24
 * @Date: 2020-11-30 09:28:20
 * @LastEditTime: 2022-06-06 13:40:28
 * @LastEditors: kinggreat24
 * @Description: 
 * @FilePath: /mulls_ros/src/CeresOptimizer.cc
 * @可以输入预定的版权声明、个性签名、空行等
 */
#include "CeresOptimizer.h"
#include "Common.h"
using namespace std;

namespace ORB_SLAM2
{

    int CeresOptimizer::PoseOptimization(lo::cloudblock_Ptr pFrame,
                                         pcl::PointCloud<Point_T>::Ptr &pLocalCornerMap,
                                         pcl::PointCloud<Point_T>::Ptr &pLocalSurfMap,
                                         pcl::PointCloud<Point_T>::Ptr &pCurCornerScan,
                                         pcl::PointCloud<Point_T>::Ptr &pCurSurfScan,
                                         int max_iteration)
    {
        double parameters[7] = {0, 0, 0, 1, 0, 0, 0}; //orentation translation
        Eigen::Map<Eigen::Quaterniond> m_q_w_curr(parameters);
        Eigen::Map<Eigen::Vector3d> m_t_w_curr(parameters + 4);

        Eigen::Matrix4d Twl = pFrame->pose_lo;
        Eigen::Quaterniond q(Twl.topLeftCorner<3, 3>(0, 0));
        q.normalize();
        Eigen::Vector3d t = Twl.block<3, 1>(0, 3);
        
        std::cout << "Before optimization Twl: " << std::endl
                  << Twl << std::endl;

        //6DOF优化初值
        parameters[0] = q.x();
        parameters[1] = q.y();
        parameters[2] = q.z();
        parameters[3] = q.w();
        parameters[4] = t[0];
        parameters[5] = t[1];
        parameters[6] = t[2];

        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        Point_T pointOri, pointSel, pointProj, coeff;

        const int laserCloudCornerLastDSNum = pCurCornerScan->size();
        const int laserCloudSurfLastDSNum = pCurSurfScan->size();

        ceres::LinearSolverType slover_type = ceres::DENSE_SCHUR; // SPARSE_NORMAL_CHOLESKY | DENSE_QR | DENSE_SCHUR
        int m_para_cere_max_iterations = 100;
        int m_para_cere_prerun_times = 10;

        double m_inlier_ratio = 0.80;
        double m_inliner_dis = 0.02; // 0.02m
        double m_inliner_dis_visual = 5.0;
        double m_inlier_threshold_corner;
        double m_inlier_threshold_surf;
        double m_inlier_threshold_points;

        const double visual_weight = 0.06;
        // const double visual_weight = 1.0;

        double t_edge_feature_association = 0.0;
        double t_surf_feature_association = 0.0;

        // 6-DOF 优化
        if (pLocalCornerMap->size() > 10 && pLocalSurfMap->size() > 10)
        {
            pcl::KdTreeFLANN<Point_T> m_kdtree_corner_from_map;
            pcl::KdTreeFLANN<Point_T> m_kdtree_surf_from_map;
            m_kdtree_corner_from_map.setInputCloud(pLocalCornerMap);
            m_kdtree_surf_from_map.setInputCloud(pLocalSurfMap);

            for (int iterCount = 0; iterCount < max_iteration; iterCount++)
            {
                ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                ceres::LossFunction *visual_loss_function = new ceres::HuberLoss(sqrt(visual_weight * 5.991));
                ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
                ceres::Problem::Options problem_options;
                ceres::ResidualBlockId block_id;
                ceres::Problem problem(problem_options);
                std::vector<ceres::ResidualBlockId> mappoints_residual_block_ids;
                std::vector<ceres::ResidualBlockId> corner_residual_block_ids;
                std::vector<ceres::ResidualBlockId> surf_residual_block_ids;

                problem.AddParameterBlock(parameters, 4, q_parameterization);
                problem.AddParameterBlock(parameters + 4, 3);

                // Add Corner Points
                int corner_num = 0;
                int corner_num_rejected = 0;
                std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
                for (int i = 0; i < laserCloudCornerLastDSNum; i++)
                {
                    pointOri = pCurCornerScan->points[i];
                    pointAssociateToMap(&pointOri, m_q_w_curr, m_t_w_curr, &pointSel);
                    m_kdtree_corner_from_map.nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

                    if (pointSearchSqDis[4] < 1.0)
                    {
                        std::vector<Eigen::Vector3d> nearCorners;
                        Eigen::Vector3d center(0, 0, 0);
                        for (int j = 0; j < 5; j++)
                        {
                            Eigen::Vector3d tmp(pLocalCornerMap->points[pointSearchInd[j]].x,
                                                pLocalCornerMap->points[pointSearchInd[j]].y,
                                                pLocalCornerMap->points[pointSearchInd[j]].z);
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

                        // if is indeed line feature note Eigen library sort eigenvalues in increasing order
                        Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
                        Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
                        if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
                        {
                            Eigen::Vector3d point_on_line = center;
                            Eigen::Vector3d point_a, point_b;
                            point_a = 0.1 * unit_direction + point_on_line;
                            point_b = -0.1 * unit_direction + point_on_line;

                            double weight = 1.0;
                            ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, weight);
                            block_id = problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);

                            corner_residual_block_ids.push_back(block_id);
                            corner_num++;
                        }
                        else
                        {
                            corner_num_rejected++;
                        }
                    }
                }
                std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
                double t_edge = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
                t_edge_feature_association += t_edge;

                std::cout << "Add corner num: " << corner_num << std::endl;

                // Add Surf Points
                int surf_num = 0;
                int surf_rejected_num = 0;
                std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
                for (int i = 0; i < laserCloudSurfLastDSNum; i++)
                {
                    pointOri = pCurSurfScan->points[i];
                    pointAssociateToMap(&pointOri, m_q_w_curr, m_t_w_curr, &pointSel);
                    m_kdtree_surf_from_map.nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

                    Eigen::Matrix<double, 5, 3> matA0;
                    Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
                    if (pointSearchSqDis[4] < 1.0)
                    {
                        for (int j = 0; j < 5; j++)
                        {
                            matA0(j, 0) = pLocalSurfMap->points[pointSearchInd[j]].x;
                            matA0(j, 1) = pLocalSurfMap->points[pointSearchInd[j]].y;
                            matA0(j, 2) = pLocalSurfMap->points[pointSearchInd[j]].z;
                        }
                        // find the norm of plane
                        Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                        double negative_OA_dot_norm = 1 / norm.norm();
                        norm.normalize();

                        // Here n(pa, pb, pc) is unit norm of plane
                        bool planeValid = true;
                        for (int j = 0; j < 5; j++)
                        {
                            // if OX * n > 0.2, then plane is not fit well
                            if (fabs(norm(0) * pLocalSurfMap->points[pointSearchInd[j]].x +
                                     norm(1) * pLocalSurfMap->points[pointSearchInd[j]].y +
                                     norm(2) * pLocalSurfMap->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
                            {
                                planeValid = false;
                                break;
                            }
                        }
                        Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
                        if (planeValid)
                        {
                            double weight = 1.0;
                            ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(curr_point, norm, negative_OA_dot_norm, weight);
                            block_id = problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);

                            surf_residual_block_ids.push_back(block_id);
                            surf_num++;
                        }
                        else
                        {
                            surf_rejected_num++;
                        }
                    }
                }
                std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
                double t_surf = std::chrono::duration_cast<std::chrono::duration<double>>(t4 - t3).count();
                t_surf_feature_association += t_surf;
                std::cout << "Add surf num: " << surf_num << std::endl;

                // 第一次优化
                ceres::Solver::Options options;
                options.linear_solver_type = slover_type;
                options.max_num_iterations = m_para_cere_prerun_times;
                options.minimizer_progress_to_stdout = false;
                options.check_gradients = false;
                options.gradient_check_relative_precision = 1e-6;
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);

                // 计算边缘的残差,剔除错误的匹配
                ceres::Problem::EvaluateOptions surf_eval_options;
                surf_eval_options.residual_blocks = surf_residual_block_ids;
                double surf_total_cost = 0.0;
                std::vector<double> surf_residuals; //每个观测的误差
                problem.Evaluate(surf_eval_options, &surf_total_cost, &surf_residuals, nullptr, nullptr);

                // 计算平坦点的残差,剔除错误的匹配
                ceres::Problem::EvaluateOptions edge_eval_options;
                edge_eval_options.residual_blocks = corner_residual_block_ids;
                double corner_total_cost = 0.0;
                std::vector<double> corner_residuals; //每个观测的误差
                problem.Evaluate(edge_eval_options, &corner_total_cost, &corner_residuals, nullptr, nullptr);

                //边缘点
                std::vector<ceres::ResidualBlockId> corner_residual_block_ids_temp;
                corner_residual_block_ids_temp.reserve(corner_residual_block_ids.size());
                double m_inliner_ratio_threshold_corner = compute_inlier_residual_threshold_corner(corner_residuals, m_inlier_ratio);
                m_inlier_threshold_corner = std::max(m_inliner_dis, m_inliner_ratio_threshold_corner);

                //平坦点
                std::vector<ceres::ResidualBlockId> surf_residual_block_ids_temp;
                surf_residual_block_ids_temp.reserve(surf_residual_block_ids.size());
                double m_inliner_ratio_threshold_surf = compute_inlier_residual_threshold_surf(surf_residuals, m_inlier_ratio);
                m_inlier_threshold_surf = std::max(m_inliner_dis, m_inliner_ratio_threshold_surf);

                //剔除误差较大的边缘激光点
                for (unsigned int i = 0; i < corner_residual_block_ids.size(); i++)
                {
                    if ((fabs(corner_residuals[3 * i + 0]) + fabs(corner_residuals[3 * i + 1]) + fabs(corner_residuals[3 * i + 2])) > m_inlier_threshold_corner) // std::min( 1.0, 10 * avr_cost )
                    {
                        //screen_out << "Remove outliers, drop id = " << (void *)residual_block_ids[ i ] <<endl;
                        problem.RemoveResidualBlock(corner_residual_block_ids[i]);
                    }
                    else
                    {
                        corner_residual_block_ids_temp.push_back(corner_residual_block_ids[i]);
                    }
                }

                //剔除误差较大的平坦点激光点
                for (unsigned int i = 0; i < surf_residual_block_ids.size(); i++)
                {
                    if (fabs(surf_residuals[i]) > m_inlier_threshold_surf) // std::min( 1.0, 10 * avr_cost )
                    {
                        //screen_out << "Remove outliers, drop id = " << (void *)residual_block_ids[ i ] <<endl;
                        problem.RemoveResidualBlock(surf_residual_block_ids[i]);
                    }
                    else
                    {
                        surf_residual_block_ids_temp.push_back(surf_residual_block_ids[i]);
                    }
                }

                corner_residual_block_ids = corner_residual_block_ids_temp;
                surf_residual_block_ids = surf_residual_block_ids_temp;

                //重新进行优化
                options.linear_solver_type = slover_type;
                options.max_num_iterations = m_para_cere_max_iterations;
                options.minimizer_progress_to_stdout = false;
                options.check_gradients = false;
                options.gradient_check_relative_precision = 1e-10;

                // set_ceres_solver_bound( problem, m_para_buffer_incremental );
                ceres::Solve(options, &problem, &summary);
                std::cout << summary.BriefReport() << std::endl;
            }

            //更新当前帧的位姿
            Eigen::Matrix4d tcw = Eigen::Matrix4d::Identity();
            Eigen::Matrix3d r_cw = m_q_w_curr.toRotationMatrix().transpose();
            tcw.block<3, 3>(0, 0) = r_cw;
            tcw.block<3, 1>(0, 3) = -r_cw * m_t_w_curr;
            
            // pFrame->SetPose(Converter::toCvMat(tcw));

            std::cout << "time of edge feature association: " << t_edge_feature_association << std::endl
                      << "time of surf feature association: " << t_surf_feature_association << std::endl;
        }
    }
} // namespace ORB_SLAM2