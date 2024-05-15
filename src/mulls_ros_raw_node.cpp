//
// This file is for the general implements of Multi-Metrics Linear Least Square Lidar SLAM (MULLS)
// Compulsory Dependent 3rd Libs: PCL (>1.7), glog, gflags
// By Yue Pan

#include <ros/ros.h>
#include <thread>

// opencv
#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "dataio.hpp"
#include "cfilter.hpp"
#include "cregistration.hpp"
#include "utility.hpp"
#include "map_viewer.h"
#include "map_manager.h"
#include "odom_error_compute.h"
#include "common_nav.h"
#include "build_pose_graph.h"
#include "graph_optimizer.h"

#include "lidar_sparse_align/SparseLidarAlign.h"
#include "CeresOptimizer.h"
#include "VisualFeatureExtractor.h"
#include "LidarFeatureExtractor.h"
#include "ORBmatcher.h"

// 激光里程计
#include "odomEstimationClass.hpp"

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <tf/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Int64MultiArray.h>

// cv_bridge
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>

#include <visualization_msgs/MarkerArray.h>

#include <jsk_rviz_plugins/OverlayText.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
// #include <jsk_rviz_plugins/Plotter2D.h>

#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace lo;

// static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
// #define CUDA_CHECK(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)
// Parameter Lists: //TODO: delete those deprecated parameters (most of the parameters listed below can be used as default in practice)
//--------------------------------------------------------------------------------------------------------------------------------------------------------------
// GFLAG Template: DEFINE_TYPE(Flag_variable_name, default_value, "Comments")
// data path
DEFINE_string(point_cloud_folder, "", "folder containing the point cloud of each frame");
DEFINE_string(pc_format, ".pcd", "input point cloud format (select from .pcd, .ply, .txt, .las, .h5 ...");
DEFINE_string(output_adjacent_lo_pose_file_path, "", "the file for saving the adjacent transformation of lidar odometry");
DEFINE_string(gt_body_pose_file_path, "", "optional: the file containing the ground truth pose in body coor. sys. (as the format of transformation matrix)");
DEFINE_string(calib_file_path, "", "optional: the file containing the calibration matrix (as the format of transformation matrix)");
DEFINE_string(output_gt_lidar_pose_file_path, "", "optional: the file for saving the groundtruth pose in the lidar coor. sys.");
DEFINE_string(output_lo_body_pose_file_path, "", "optional: the file for saving the pose estimated by the lidar odometry in the body coor. sys.");
DEFINE_string(output_lo_lidar_pose_file_path, "", "optional: the file for saving the pose estimated by the lidar odometry in the lidar coor. sys.");
DEFINE_string(output_map_point_cloud_folder_path, "", "optional: the folder for saving the transformed point clouds in world coordinate system");
DEFINE_string(lo_lidar_pose_point_cloud, "", "optional: save the lidar odometry trajectory as point cloud");
DEFINE_string(gt_lidar_pose_point_cloud, "", "optional: save the ground truth trajectory as point cloud");
DEFINE_string(timing_report_file, "", "optional: the path of the file for recording the consuming time detail of the program");
DEFINE_bool(gt_in_lidar_frame, false, "whether the ground truth pose is provided in the lidar frame or the body/camera frame");
DEFINE_bool(gt_oxts_format, false, "is the ground truth pose in oxts (index ts tran quat) format or the transformation matrix format");
DEFINE_int32(dataset_type, 0, "dataset type, 0: kitti, 1: tum");

// used frame
DEFINE_int32(frame_num_begin, 0, "begin from this frame (file sequence in the folder)");
DEFINE_int32(frame_num_end, 99999, "end at this frame (file sequence in the folder)");
DEFINE_int32(frame_step, 1, "use one in ${frame_step} frames");
// map related
DEFINE_bool(write_out_map_on, false, "output map point clouds or not");
DEFINE_bool(write_out_gt_map_on, false, "output map point clouds generated from the gnssins pose or not");
DEFINE_bool(write_map_each_frame, false, "output each frame's point cloud in map coordinate system");
DEFINE_int32(map_downrate_output, 5, "downsampling rate for output map point cloud");
DEFINE_bool(map_filter_on, false, "clean the map point cloud before output");
DEFINE_bool(apply_dist_filter, false, "Use only the points inside a distance range or not");
DEFINE_double(min_dist_used, 1.0, "only the points whose distance to the scanner is larger than this value would be used for scan matching (m)");
DEFINE_double(max_dist_used, 120.0, "only the points whose distance to the scanner is smaller than this value would be used for scan matching (m)");
DEFINE_double(min_dist_mapping, 2.0, "only the points whose distance to the scanner is larger than this value would be used for map merging (m)");
DEFINE_double(max_dist_mapping, 60.0, "only the points whose distance to the scanner is smaller than this value would be used for map merging (m)");
// viusalization related
DEFINE_bool(real_time_viewer_on, false, "launch real time viewer or not");
DEFINE_int32(screen_width, 1920, "monitor horizontal resolution (pixel)");
DEFINE_int32(screen_height, 1080, "monitor vertical resolution (pixel)");
DEFINE_double(vis_intensity_scale, 256.0, "max intensity value of your data");
DEFINE_int32(vis_map_history_down_rate, 300, "downsampling rate of the map point cloud kept in the memory");
DEFINE_int32(vis_map_history_keep_frame_num, 150, "frame number of the dense map that would be kept in the memory");
DEFINE_int32(vis_initial_color_type, 0, "map viewer's rendering color style: (0: single color & semantic mask, 1: frame-wise, 2: height, 3: intensity)");
DEFINE_double(laser_vis_size, 0.5, "size of the laser on the map viewer");
DEFINE_bool(vis_pause_at_loop_closure, false, "the visualizer would pause when a new loop closure is cosntructed");
DEFINE_bool(show_range_image, false, "display the range image or not in realtime");
DEFINE_bool(show_bev_image, false, "display the bev image or not in realtime");
// lidar odometry related
DEFINE_bool(scan_to_scan_module_on, false, "apply scan-to-scan registration or just scan-to-localmap matching");
DEFINE_int32(initial_scan2scan_frame_num, 2, "only conduct scan to scan registration for the first　${initial_scan2scan_frame_num} frames");
DEFINE_int32(motion_compensation_method, 0, "method for motion compensation of lidar (0: disabled, 1: uniform motion model (from point-wise timestamp), 2: from azimuth, 3: from azimuth (rotation-only), 4: imu-assisted)");
DEFINE_bool(vertical_ang_calib_on, false, "apply vertical intrinsic angle correction for sepecific lidar");
DEFINE_double(vertical_ang_correction_deg, 0.0, "the intrinsic correction of the vertical angle of lidar");
DEFINE_bool(zupt_on_or_not, false, "enable zupt (zero velocity updating) or not");
DEFINE_bool(apply_scanner_filter, false, "enable scanner based distance filtering or not");
DEFINE_bool(semantic_assist_on, false, "apply semantic mask to assist the geometric feature points extraction");
DEFINE_double(cloud_down_res, 0.0, "voxel size(m) of downsample for target point cloud");
DEFINE_int32(dist_inverse_sampling_method, 2, "use distance inverse sampling or not (0: disabled, 1: linear weight, 2: quadratic weight)");
DEFINE_double(unit_dist, 15.0, "distance that correspoinding to unit weight in inverse distance downsampling");
DEFINE_bool(adaptive_parameters_on, false, "use self-adaptive parameters for different surroundings and road situation");
DEFINE_double(cloud_pca_neigh_r, 0.6, "pca neighborhood searching radius(m) for target point cloud");
DEFINE_int32(cloud_pca_neigh_k, 25, "use only the k nearest neighbor in the r-neighborhood to do PCA");
DEFINE_int32(cloud_pca_neigh_k_min, 8, "the min number of points in the neighborhood for doing PCA");
DEFINE_int32(pca_down_rate, 2, "Downsampling rate of the pca querying points and the points used to calculate pca and build the kd-tree");
DEFINE_bool(sharpen_with_nms_on, true, "using non-maximum supression to get the sharpen feature points from unsharpen points or not (use higher threshold)");
DEFINE_bool(fixed_num_downsampling_on, true, "enable/disable the fixed point number downsampling (processing time's standard deviation would br smaller)");
DEFINE_int32(ground_down_fixed_num, 300, "fixed number of the detected ground feature points (for source)");
DEFINE_int32(pillar_down_fixed_num, 100, "fixed number of the detected pillar feature points (for source)");
DEFINE_int32(facade_down_fixed_num, 400, "fixed number of the detected facade feature points (for source)");
DEFINE_int32(beam_down_fixed_num, 100, "fixed number of the detected beam feature points (for source)");
DEFINE_int32(roof_down_fixed_num, 0, "fixed number of the detected roof feature points (for source)");
DEFINE_int32(unground_down_fixed_num, 10000, "fixed number of the unground points used for PCA calculation");
DEFINE_double(gf_grid_size, 3.0, "grid size(m) of ground segmentation");
DEFINE_double(gf_in_grid_h_thre, 0.3, "height threshold(m) above the lowest point in a grid for ground segmentation");
DEFINE_double(gf_neigh_grid_h_thre, 1.5, "height threshold(m) among neighbor grids for ground segmentation");
DEFINE_double(gf_max_h, 5.0, "max height(m) allowed for ground point");
DEFINE_int32(ground_normal_method, 0, "method for estimating the ground points' normal vector ( 0: directly use (0,0,1), 1: estimate normal in fix radius neighborhood , 2: estimate normal in k nearest neighborhood, 3: use ransac to estimate plane coeffs in a grid)");
DEFINE_double(gf_normal_estimation_radius, 2.0, "neighborhood radius for local normal estimation of ground points (only enabled when ground_normal_method=1)");
DEFINE_int32(gf_ground_down_rate, 15, "downsampling decimation rate for target ground point cloud");
DEFINE_int32(gf_nonground_down_rate, 3, "downsampling decimation rate for non-ground point cloud");
DEFINE_double(intensity_thre_nonground, FLT_MAX, "Points whose intensity is larger than this value would be regarded as highly reflective object so that downsampling would not be applied.");
DEFINE_int32(gf_grid_min_pt_num, 10, "min number of points in a grid (if < this value, the grid would not be considered");
DEFINE_int32(gf_reliable_neighbor_grid_thre, 0, "min number of neighboring grid whose point number is larger than gf_grid_min_pt_num-1");
DEFINE_int32(gf_down_down_rate, 2, "downsampling rate based on the already downsampled ground point clouds used for source point cloud");
DEFINE_double(feature_pts_ratio_guess, 0.3, "A guess of the percent of the geometric feature points in the neighborhood");
DEFINE_double(linearity_thre, 0.65, "pca linearity threshold for target point cloud");
DEFINE_double(planarity_thre, 0.65, "pca planarity threshold for target point cloud");
DEFINE_double(linearity_thre_down, 0.75, "pca linearity threshold for source point cloud");
DEFINE_double(planarity_thre_down, 0.75, "pca planarity threshold for source point cloud");
DEFINE_double(curvature_thre, 0.12, "pca local curvature threshold");
DEFINE_int32(bsc_grid_num_per_side, 7, "numbder of grid per side in BSC feature");
DEFINE_double(beam_direction_ang, 25, "the verticle angle threshold for the direction vector of beam-type feature points");
DEFINE_double(pillar_direction_ang, 60, "the verticle angle threshold for the direction vector of pillar-type feature points");
DEFINE_double(facade_normal_ang, 30, "the verticle angle threshold for the normal vector of facade-type feature points");
DEFINE_double(roof_normal_ang, 75, "the verticle angle threshold for the normal vector roof-type feature points");
DEFINE_double(beam_max_height, FLT_MAX, "max bearable height for beam points");
DEFINE_int32(vertex_extraction_method, 2, "extraction method of vertex points (0: disabled, 1: maximum local curvature within stable points, 2: intersection points of pillar and beams)");
DEFINE_bool(detect_curb_or_not, false, "detect curb feature for urban scenarios or not");
DEFINE_bool(apply_roi_filter, false, "use the region of interest filter to remove dynamic objects or not");
DEFINE_double(roi_min_y, -FLT_MAX, "region of interest (delete part): min_y");
DEFINE_double(roi_max_y, FLT_MAX, "region of interest (delete part): max_y");
DEFINE_string(used_feature_type, "111100", "used_feature_type (1: on, 0: off, order: ground, pillar, beam, facade, roof, vetrex)");
DEFINE_bool(reg_intersection_filter_on, true, "filter the points outside the intersection aera of two point cloud during registration");
DEFINE_bool(normal_shooting_on, false, "using normal shooting instead of nearest neighbor searching when determing correspondences");
DEFINE_double(normal_bearing, 35.0, "the normal consistency checking angle threshold (unit: degree)");
DEFINE_double(corr_dis_thre_init, 1.5, "distance threshold between correspondence points at begining");
DEFINE_double(corr_dis_thre_min, 0.5, "minimum distance threshold between correspondence points at begining");
DEFINE_double(dis_thre_update_rate, 1.1, "update rate (divided by this value at each iteration) for distance threshold between correspondence points");
DEFINE_string(corr_weight_strategy, "1101", "weighting strategy for correspondences (1: on, 0: off, order: x,y,z balanced weight, residual weight, distance weight, intensity weight)");
DEFINE_double(z_xy_balance_ratio, 1.0, "the weight ratio of the error along z and x,y direction when balanced weight is enabled");
DEFINE_double(pt2pt_res_window, 0.1, "residual window size for the residual robust kernel function of point to point correspondence");
DEFINE_double(pt2pl_res_window, 0.1, "residual window size for the residual robust kernel function of point to plane correspondence");
DEFINE_double(pt2li_res_window, 0.1, "residual window size for the residual robust kernel function of point to line correspondence");
DEFINE_int32(reg_max_iter_num_s2s, 1, "max iteration number for icp-based registration (scan to scan)");
DEFINE_int32(reg_max_iter_num_s2m, 1, "max iteration number for icp-based registration (scan to map)");
DEFINE_int32(reg_max_iter_num_m2m, 3, "max iteration number for icp-based registration (map to map)");
DEFINE_double(converge_tran, 0.001, "convergence threshold for translation (in m)");
DEFINE_double(converge_rot_d, 0.01, "convergence threshold for rotation (in degree)");
DEFINE_double(post_sigma_thre, 0.35, "the maximum threshold for the posterior standard deviation of the least square adjustment during the registration.(unit:m)");
DEFINE_double(local_map_radius, 50.0, "the radius of the local map (regarded as a sphere aera)");
DEFINE_int32(local_map_max_pt_num, 8000, "max point number allowed for the local map");
DEFINE_int32(local_map_max_vertex_pt_num, 1000, "max vertex point number allowed for the local map");
DEFINE_double(append_frame_radius, 60.0, "the radius of the frame that used to append into the local map");
DEFINE_bool(apply_map_based_dynamic_removal, false, "use map based dynamic object removal or not");
DEFINE_double(map_min_dist_within_feature, 0.03, "if the expanded feature point is too close to already exsit map points, it would not be added to the map");
DEFINE_double(dynamic_removal_radius, 30.0, "the radius of the map based dynamic object removing");
DEFINE_double(dynamic_dist_thre_min, 0.3, "the distance threshold to judge if a point is dynamic or not");
DEFINE_int32(local_map_recalculation_frequency, 99999, "Recalculate the linear features in the local map each ${local_map_recalculation_frequency} frame");
DEFINE_int32(s2m_frequency, 1, "frequency of scan to map registration");
DEFINE_int32(initial_guess_mode, 2, "Use which kind of initial guess(0: no initial guess, 1: uniform motion(translation only), 2: uniform motion(translation+rotation), 3:imu based)");
// prior knowledge
DEFINE_double(approx_scanner_height, 1.5, "approximate height of the scanner (m)");
DEFINE_double(underground_height_thre, -6.0, "z-axis threshold for rejecting underground ghost points (lines)");
// loop closure and pose graph optimization related
DEFINE_bool(loop_closure_detection_on, false, "do loop closure detection and pose graph optimization or not");
DEFINE_double(submap_accu_tran, 15.0, "accumulated translation (m) for generating a new submap");
DEFINE_double(submap_accu_rot, 90.0, "accumulated rotation (deg) for generating a new submap");
DEFINE_int32(submap_accu_frame, 150, "accumulated frame number for generating a new submap");
DEFINE_double(map2map_reliable_sigma_thre, 0.04, "if the standard deviation of the map to map registration is smaller than this value, we would use it as odometry result");
DEFINE_bool(overall_loop_closure_searching_on, false, "searching loop clousre within a larger neighborhood");
DEFINE_double(min_iou_thre, 0.4, "min boundingbox iou for candidate registration edge");
DEFINE_double(min_iou_thre_global_reg, 0.5, "min boundingbox iou for global registration edge");
DEFINE_int32(min_submap_id_diff, 8, "min submap id difference between two submaps for a putable registration edge");
DEFINE_double(neighbor_search_dist, 50.0, "max distance for candidate registration edge");
DEFINE_double(map_to_map_min_cor_ratio, 0.15, "min feature point overlapping ratio for map to map registration");
DEFINE_int32(cooling_submap_num, 2, "waiting for several submaps (without loop closure detection) after applying a successful pgo");
DEFINE_double(adjacent_edge_weight_ratio, 1.0, "weight of adjacent edge compared with registration edge");
DEFINE_int32(num_frame_thre_large_drift, 1000, "the lidar odometry may have large drift after so many frames so we would use global registration instead of a local registration with lidar odom initial guess");
DEFINE_int32(max_used_reg_edge_per_optimization, 3, "only usee the first ${max_used_reg_edge_per_optimization} for pgo");
DEFINE_bool(equal_weight_on, false, "using equal weight for the information matrix in pose graph optimization");
DEFINE_bool(reciprocal_feature_match_on, false, "using reciprocal nn feature matching or not");
DEFINE_bool(best_n_feature_match_on, true, "select the n correspondence with min feature distance as the putatble matches");
DEFINE_int32(feature_corr_num, 1000, "number of the correspondence for global coarse registration");
DEFINE_bool(teaser_based_global_registration_on, true, "Using TEASER++ to do the global coarse registration or not");
DEFINE_int32(global_reg_min_inlier_count, 7, "min inlier correspondence for a successful feature based registration (for teaser or ransac)");
DEFINE_string(pose_graph_optimization_method, "ceres", "use which library to do pgo (select from g2o, ceres and gtsam)");
DEFINE_double(inter_submap_t_limit, 2.0, "the submap node's limit of translation variation, unit:m");
DEFINE_double(inter_submap_r_limit, 0.05, "the submap node's limit of rotation variation, unit quaternion");
DEFINE_double(inner_submap_t_limit, 0.1, "the inner submap frame node's limit of translation variation, unit:m");
DEFINE_double(inner_submap_r_limit, 0.01, "the inner submap frame node's limit of rotation variation, unit:m");
DEFINE_int32(max_iter_inter_submap, 100, "max iteration number for inter submap pgo");
DEFINE_int32(max_iter_inner_submap, 100, "max iteration number for inner submap pgo");
DEFINE_double(first_time_cov_update_ratio, 1.0, "edge covariance update (at first pgo)");
DEFINE_double(life_long_cov_update_ratio, 1.0, "edge covariance update (after first pgo)");
DEFINE_bool(diagonal_information_matrix_on, false, "use diagonal information matrix in pgo or not");
DEFINE_double(wrong_edge_tran_thre, 5.0, "translation threshold for judging if a edge is wrong or not");
DEFINE_double(wrong_edge_rot_thre_deg, 25.0, "rotation threshold for judging if a edge is wrong or not");
DEFINE_double(frame_estimated_error_tran, 1.0, "estimated max translation error of the lidar odometry per frame");
DEFINE_double(frame_estimated_error_rot_deg, 2.0, "estimated max rotation error of the lidar odometry per frame");
DEFINE_bool(robust_kernel_on, false, "turn on the robust kernel function in pgo");
DEFINE_bool(free_node_on, false, "enable the free node module or not");
DEFINE_bool(transfer_correct_reg_tran_on, true, "enable the registration tranformation transfer (only do global reg. once for each query submap)");
DEFINE_bool(framewise_pgo_on, false, "use frame-wise pgo or not");
// baseline method options
DEFINE_string(baseline_reg_method, "", "name of the baseline lidar odometery method (ndt, gicp, pclicp, etc.)");
DEFINE_double(reg_voxel_size, 1.0, "the grid size of ndt or vgicp");
DEFINE_bool(ndt_searching_method, true, "using direct searching or kdtree (0: kdtree, 1: direct7)");
DEFINE_bool(voxel_gicp_on, true, "using voxel based gicp (faster)");

// ORB feature extraction
DEFINE_int32(num_features, 1000, "number of features extracted from image");
DEFINE_double(scale_factor, 1.2, "scale factor");
DEFINE_int32(num_levels, 8, "number of pyramid levels");
DEFINE_int32(ini_th_fast, 20, "ini threshold fast");
DEFINE_int32(min_th_fast, 7, "min threshold fast");
DEFINE_string(bow_file, "", "bow file path");

// Camera params
DEFINE_int32(im_width, 1226, "number of pyramid levels");
DEFINE_int32(im_height, 370, "ini threshold fast");
DEFINE_double(cam_fx, 707.0912, "camera fx");
DEFINE_double(cam_fy, 707.0912, "camera fy");
DEFINE_double(cam_cx, 601.8873, "camera cx");
DEFINE_double(cam_cy, 183.1104, "camera cy");
DEFINE_double(cam_k1, 0.0, "scale factor");
DEFINE_double(cam_k2, 0.0, "scale factor");
DEFINE_double(cam_p1, 0.0, "scale factor");
DEFINE_double(cam_p2, 0.0, "scale factor");
DEFINE_double(cam_k3, 0.0, "scale factor");

// Lidar feature
DEFINE_string(lidar_feature_params, "", "patchwork params and sensor names");

// Direct lidar tracker
DEFINE_int32(tracker_levels, 3, "");
DEFINE_int32(tracker_min_level, 0, "");
DEFINE_int32(tracker_max_level, 2, "");
DEFINE_int32(tracker_max_iteration, 100, "");
DEFINE_string(tracker_scale_estimator, "TDistributionScale", "bow file path");
DEFINE_string(tracker_weight_function, "TDistributionWeight", "bow file path");
// Note: Set these parameters in the config file , or the default values are used.
//--------------------------------------------------------------------------------------------------------------------------------------------------------------

//---------------------------     global variables      ------------------------
ros::Publisher pub_lidar_raw_, pub_feature_ground_, pub_feature_facade_, pub_feature_roof_, pub_feature_pillar_, pub_feature_beam_, pub_feature_vertex_;
ros::Publisher pub_local_map_, pub_submap_lists_, pub_submap_boundingboxs_;
ros::Publisher pub_lidar_odom_path_, pub_floam_lidar_odom_path_;
ros::Publisher pub_img_, pub_submap_range_img_, pub_range_img_, pub_bev_img_;
// ros::Publisher pub_loop_node_, pub_loop_edge_;
ros::Publisher pub_pgo_markers_;
nav_msgs::Path lidar_odom_path_,floam_lidar_odom_path_;

OdomEstimationClass<Point_T> odomEstimationNode;

void LoadImagesKITTI(const string &strPathToSequence, vector<string> &vstrImageFilenames,
                     vector<string> &vstrLidarFilenames, vector<double> &vTimestamps);

void LoadImagesTUM(const string &strFile, vector<string> &vstrImageFilenames,
                   vector<string> &vstrLidarFilenames, vector<double> &vTimestamps);

void publishImage(const ros::Publisher &image_publisher_, cv::Mat &img);

void publishLidarFeatures(lo::cloudblock_Ptr cur_block, bool show_raw_lidar = false);

void PublishCurrentLocalMap_realtime(cloudblock_Ptr &local_map, bool show_with_color = false);

void PublishTF(const Eigen::Matrix4d &Twl);

void gray2color(const cv::Mat &imgGray, cv::Mat &imgColor, const float v_min, const float v_max);

void display_image(const cv::Mat &image, cv::Mat &ri_temp_cm, int color_scale);

void visualizeSubmapBlocks(const cloudblock_Ptrs &cblock_submaps);

// 可视化闭环信息
void visualizePoseGraph(ros::Publisher *pub_m, ros::Time timestamp, const constraints &cons);

int main(int argc, char **argv)
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging("Mylog_testlo");

    ros::init(argc, argv, "mulls_node");
    ros::NodeHandle nh, nh_private("~");

    pub_feature_ground_ = nh.advertise<sensor_msgs::PointCloud2>("ground_feature", 1);
    pub_feature_facade_ = nh.advertise<sensor_msgs::PointCloud2>("facade_feature", 1);
    pub_feature_roof_ = nh.advertise<sensor_msgs::PointCloud2>("roof_feature", 1);
    pub_feature_pillar_ = nh.advertise<sensor_msgs::PointCloud2>("pillar_feature", 1);
    pub_feature_beam_ = nh.advertise<sensor_msgs::PointCloud2>("beam_feature", 1);
    pub_feature_vertex_ = nh.advertise<sensor_msgs::PointCloud2>("vertex_feature", 1);
    pub_lidar_raw_ = nh.advertise<sensor_msgs::PointCloud2>("lidar_raw", 1);

    pub_local_map_ = nh.advertise<sensor_msgs::PointCloud2>("local_map", 1);
    pub_submap_lists_ = nh.advertise<sensor_msgs::PointCloud2>("sub_map_list", 1);
    pub_submap_boundingboxs_ = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("submap_boundingbox_list", 1);
    pub_lidar_odom_path_ = nh.advertise<nav_msgs::Path>("lo_path", 1);

    pub_floam_lidar_odom_path_= nh.advertise<nav_msgs::Path>("floam_lo_path", 1);

    pub_img_ = nh.advertise<sensor_msgs::Image>("image", 1);
    pub_submap_range_img_ = nh.advertise<sensor_msgs::Image>("submap_range_image", 1);
    pub_range_img_ = nh.advertise<sensor_msgs::Image>("range_image", 1);
    pub_bev_img_ = nh.advertise<sensor_msgs::Image>("bev_image", 1);

    pub_pgo_markers_ = nh.advertise<visualization_msgs::MarkerArray>("pgo_markers", 1);

    std::string FLAGS_dataset_folder("");
    std::string FLAGS_output_adjacent_lo_pose_file_path("");
    std::string FLAGS_output_lo_lidar_pose_file_path("");
    std::string FLAGS_output_lo_body_pose_file_path("");
    std::string FLAGS_gt_body_pose_file_path("");
    bool need_inverse = false;
    std::string FLAGS_calib_file_path("");
    std::string FLAGS_output_map_point_cloud_folder_path("");
    std::string FLAGS_lo_lidar_pose_point_cloud("");
    nh_private.param("FLAGS_dataset_folder", FLAGS_dataset_folder, FLAGS_dataset_folder);
    nh_private.param("FLAGS_output_adjacent_lo_pose_file_path", FLAGS_output_adjacent_lo_pose_file_path, FLAGS_output_adjacent_lo_pose_file_path);
    nh_private.param("FLAGS_output_lo_lidar_pose_file_path", FLAGS_output_lo_lidar_pose_file_path, FLAGS_output_lo_lidar_pose_file_path);
    nh_private.param("FLAGS_output_lo_body_pose_file_path", FLAGS_output_lo_body_pose_file_path, FLAGS_output_lo_body_pose_file_path);
    nh_private.param("FLAGS_gt_body_pose_file_path", FLAGS_gt_body_pose_file_path, FLAGS_gt_body_pose_file_path);
    nh_private.param("FLAGS_calib_file_path", FLAGS_calib_file_path, FLAGS_calib_file_path);
    nh_private.param("need_inverse", need_inverse, need_inverse);
    nh_private.param("FLAGS_output_map_point_cloud_folder_path", FLAGS_output_map_point_cloud_folder_path, FLAGS_output_map_point_cloud_folder_path);
    nh_private.param("FLAGS_lo_lidar_pose_point_cloud", FLAGS_lo_lidar_pose_point_cloud, FLAGS_lo_lidar_pose_point_cloud);

    int FLAGS_frame_num_begin = 0;
    int FLAGS_frame_num_end = 9999;
    int FLAGS_frame_step = 1;
    nh_private.param("FLAGS_frame_num_begin", FLAGS_frame_num_begin, FLAGS_frame_num_begin);
    nh_private.param("FLAGS_frame_num_end", FLAGS_frame_num_end, FLAGS_frame_num_end);
    nh_private.param("FLAGS_frame_step", FLAGS_frame_step, FLAGS_frame_step);

    bool FLAGS_real_time_viewer_on = false;
    nh_private.param("FLAGS_real_time_viewer_on", FLAGS_real_time_viewer_on, FLAGS_real_time_viewer_on);

    LOG(INFO) << "Launch the program!";
    LOG(INFO) << "Logging is written to " << FLAGS_log_dir;

    ROS_INFO("FLAGS_dataset_folder: %s", FLAGS_dataset_folder.c_str());

    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS); // Ban pcl warnings
    CHECK(FLAGS_dataset_folder != "") << "Need to specify the dataset's folder.";
    CHECK(FLAGS_output_lo_lidar_pose_file_path != "") << "Need to specify the output lidar odometery pose (in lidar coor. sys.) file's path.";

    // Import configuration
    // Data path (now only the *.pcd format is available)
    std::string pc_folder = FLAGS_dataset_folder + "/velodyne";
    std::string pc_format = FLAGS_pc_format;
    std::string gt_body_pose_file = FLAGS_gt_body_pose_file_path;
    std::string calib_file = FLAGS_calib_file_path;
    std::string output_adjacent_lo_pose_file = FLAGS_output_adjacent_lo_pose_file_path;
    std::string output_lo_body_pose_file = FLAGS_output_lo_body_pose_file_path;
    std::string output_lo_lidar_pose_file = FLAGS_output_lo_lidar_pose_file_path;
    std::string output_gt_lidar_pose_file = FLAGS_output_gt_lidar_pose_file_path;
    std::string output_pc_folder = FLAGS_output_map_point_cloud_folder_path;
    std::string gt_lidar_pose_point_cloud_file = FLAGS_gt_lidar_pose_point_cloud;
    std::string lo_lidar_pose_point_cloud_file = FLAGS_lo_lidar_pose_point_cloud;
    // visualization settings
    int downsamping_rate_scan_vis = 5;
    int display_time_ms = 1;
    // parameters (mainly for experiment)
    float gf_grid_resolution = FLAGS_gf_grid_size;
    float gf_max_grid_height_diff = FLAGS_gf_in_grid_h_thre;
    float gf_neighbor_height_diff = FLAGS_gf_neigh_grid_h_thre;
    int ground_down_rate = FLAGS_gf_ground_down_rate;
    int nonground_down_rate = FLAGS_gf_nonground_down_rate;
    int gf_grid_min_pt_num = FLAGS_gf_grid_min_pt_num;
    int gf_reliable_neighbor_grid_thre = FLAGS_gf_reliable_neighbor_grid_thre;
    float pca_neigh_r = FLAGS_cloud_pca_neigh_r;
    int pca_neigh_k = FLAGS_cloud_pca_neigh_k;
    float feature_neighbor_radius = 2.0 * pca_neigh_r;
    float pca_linearity_thre = FLAGS_linearity_thre;
    float pca_planarity_thre = FLAGS_planarity_thre;
    float beam_direction_sin = std::sin(FLAGS_beam_direction_ang / 180.0 * M_PI);
    float pillar_direction_sin = std::sin(FLAGS_pillar_direction_ang / 180.0 * M_PI);
    float facade_normal_sin = std::sin(FLAGS_facade_normal_ang / 180.0 * M_PI);
    float roof_normal_sin = std::sin(FLAGS_roof_normal_ang / 180.0 * M_PI);
    float pca_curvature_thre = FLAGS_curvature_thre;
    float reg_corr_dis_thre_init = FLAGS_corr_dis_thre_init;
    float reg_corr_dis_thre_min = FLAGS_corr_dis_thre_min;
    float dis_thre_update_rate = FLAGS_dis_thre_update_rate;
    float z_xy_balance_ratio = FLAGS_z_xy_balance_ratio;
    float converge_tran = FLAGS_converge_tran;
    float converge_rot_d = FLAGS_converge_rot_d;
    float pt2pt_residual_window = FLAGS_pt2pt_res_window;
    float pt2pl_residual_window = FLAGS_pt2pl_res_window;
    float pt2li_residual_window = FLAGS_pt2li_res_window;
    int max_iteration_num_s2s = FLAGS_reg_max_iter_num_s2s;
    int max_iteration_num_s2m = FLAGS_reg_max_iter_num_s2m;
    int max_iteration_num_m2m = FLAGS_reg_max_iter_num_m2m;
    int initial_guess_mode = FLAGS_initial_guess_mode;
    float local_map_radius = FLAGS_local_map_radius;
    float append_frame_radius = FLAGS_append_frame_radius;
    int local_map_max_pt_num = FLAGS_local_map_max_pt_num;
    int vertex_keeping_num = FLAGS_local_map_max_vertex_pt_num;
    float dynamic_removal_radius = FLAGS_dynamic_removal_radius;
    float dynamic_dist_thre_min = FLAGS_dynamic_dist_thre_min;
    bool loop_closure_detection_on = FLAGS_loop_closure_detection_on;

    DataIo<Point_T> dataio;
    std::thread submap_viewer_thread_; // 子图显示线程
    CFilter<Point_T> cfilter;
    CRegistration<Point_T> creg;
    MapManager mmanager;
    Navigation nav;
    Constraint_Finder confinder;
    GlobalOptimize pgoptimizer;
    SubmapFeatureExtractor submap_feature_extractor;                       // 子图激光特征提取
    ORB_SLAM2::VisualFeatureExtractor *mpVisualFeatureExtractor = nullptr; // bev视觉特征提取
    SparseLidarAlign *mpSparseVisualLidarAlign_ = nullptr;                 // 视觉激光直接法跟踪

    ROS_INFO("FLAGS_lidar_feature_params: %s", FLAGS_lidar_feature_params.c_str());
    ORB_SLAM2::LidarFeatureExtractor *mpLidarFeatureExtractor = nullptr;
    mpLidarFeatureExtractor = new ORB_SLAM2::LidarFeatureExtractor(FLAGS_lidar_feature_params);

    // 激光里程计
    odomEstimationNode.init(0.3);

    // 相机模型初始化
    int imWidth = FLAGS_im_width;
    int imHeight = FLAGS_im_height;
    float cam_fx = FLAGS_cam_fx;
    float cam_fy = FLAGS_cam_fy;
    float cam_cx = FLAGS_cam_cx;
    float cam_cy = FLAGS_cam_cy;
    float cam_k1 = FLAGS_cam_k1;
    float cam_k2 = FLAGS_cam_k2;
    float cam_p1 = FLAGS_cam_p1;
    float cam_p2 = FLAGS_cam_p2;
    float cam_k3 = FLAGS_cam_k3;

    vk::PinholeCamera *pinhole_model_ = nullptr;
    // pinhole_model_ = new vk::PinholeCamera(imWidth, imHeight,
    //                                        cam_fx, cam_fy, cam_cx, cam_cy);

    pinhole_model_ = new vk::PinholeCamera(imWidth, imHeight,
                                           cam_fx, cam_fy, cam_cx, cam_cy,
                                           cam_k1, cam_k2, cam_p1, cam_p2, cam_k3);

    // 直接法跟踪
    tracker_t tracker_params;
    tracker_params.levels = FLAGS_tracker_levels;
    tracker_params.min_level = FLAGS_tracker_min_level;
    tracker_params.max_level = FLAGS_tracker_max_level;
    tracker_params.max_iteration = FLAGS_tracker_max_iteration;
    tracker_params.scale_estimator = FLAGS_tracker_scale_estimator;
    tracker_params.weight_function = FLAGS_tracker_weight_function;
    tracker_params.set_scale_estimator_type();
    tracker_params.set_weight_function_type();
    mpSparseVisualLidarAlign_ = new SparseLidarAlign(pinhole_model_, tracker_params);

    cv::Mat imGray;
    double time_count = 0.0, visual_feature_extraction_count = 0.0, lidar_feature_extraction_count = 0.0, lidar_registration_count = 0.0;

    // set pgo options
    pgoptimizer.set_covariance_updating_ratio(FLAGS_first_time_cov_update_ratio, FLAGS_life_long_cov_update_ratio);
    pgoptimizer.set_wrong_edge_check_threshold(FLAGS_wrong_edge_tran_thre, FLAGS_wrong_edge_rot_thre_deg);

    Matrix4ds poses_gt_body_cs;  // in vehicle body (gnssins) coordinate system
    Matrix4ds poses_lo_body_cs;  // in vehicle body (gnssins) coordinate system
    Matrix4ds poses_gt_lidar_cs; // in lidar coordinate system
    Matrix4ds poses_lo_lidar_cs; // in lidar coordinate system
    Matrix4ds poses_lo_adjacent;
    Eigen::Matrix4d calib_mat; // the calib_mat is the transformation from lidar frame to body/camera frame (Tb_l)
    Eigen::Matrix4d init_poses_gt_lidar_cs;
    if (FLAGS_gt_in_lidar_frame)
    {
        poses_gt_lidar_cs = dataio.load_poses_from_transform_matrix(gt_body_pose_file, FLAGS_frame_num_begin, FLAGS_frame_num_end, FLAGS_frame_step);
        init_poses_gt_lidar_cs = poses_gt_lidar_cs[0];
    }
    else
    {
        if (FLAGS_gt_oxts_format)
            poses_gt_body_cs = dataio.load_poses_from_pose_quat(gt_body_pose_file, FLAGS_frame_num_begin, FLAGS_frame_num_end, FLAGS_frame_step);
        else
            poses_gt_body_cs = dataio.load_poses_from_transform_matrix(gt_body_pose_file, FLAGS_frame_num_begin, FLAGS_frame_num_end, FLAGS_frame_step);
    }
    dataio.load_calib_mat(calib_file, calib_mat, need_inverse);

    // 视觉特征预处理
    lo::orb_params_t orb_params;
    orb_params.n_features = FLAGS_num_features;
    orb_params.nLevels = FLAGS_num_levels;
    orb_params.scale_factor = FLAGS_scale_factor;
    orb_params.iniThFast = FLAGS_ini_th_fast;
    orb_params.minThFast = FLAGS_min_th_fast;
    Eigen::Matrix4d T_cam_lidar = calib_mat;
    mpVisualFeatureExtractor = new ORB_SLAM2::VisualFeatureExtractor(
        orb_params, tracker_params, pinhole_model_, &cfilter, T_cam_lidar, FLAGS_bow_file);

    ORB_SLAM2::ORBmatcher mOrbMatcher;

    // 读取数据集
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesLidar;
    vector<double> vTimestamps;
    if (0 == FLAGS_dataset_type)
        LoadImagesKITTI(FLAGS_dataset_folder, vstrImageFilenamesRGB, vstrImageFilenamesLidar, vTimestamps);
    else if (1 == FLAGS_dataset_type)
        LoadImagesTUM(FLAGS_dataset_folder, vstrImageFilenamesRGB, vstrImageFilenamesLidar, vTimestamps);
    std::vector<std::string> filenames = vstrImageFilenamesLidar;
    int frame_num = filenames.size();
    ROS_INFO("filenames size: %d", filenames.size());

    std::vector<std::vector<float>> timing_array(frame_num); // unit: s
    cloudblock_Ptr cblock_target(new cloudblock_t());
    cloudblock_Ptr cblock_source(new cloudblock_t());
    cloudblock_Ptr cblock_history(new cloudblock_t());
    cloudblock_Ptr cblock_local_map(new cloudblock_t());
    cloudblock_Ptrs cblock_submaps;
    cloudblock_Ptrs cblock_frames;
    constraint_t scan2scan_reg_con;        //相邻帧运动估计结果
    constraint_t scan2map_reg_con;         //帧到地图运动估计结果
    constraint_t scan2ground_reg_con;
    constraint_t scan2history_reg_con;
    constraints pgo_edges;
    Eigen::Matrix4d initial_guess_tran = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d adjacent_pose_out = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d first_frame_body = Eigen::Matrix4d::Identity();
    ROS_INFO("[ %d ] threads availiable in total", omp_get_max_threads());

    pcTPtr cblock_submap_raw(new pcT()); // 局部地图的原始点云拼接地图

    // 显示子图线程
    submap_viewer_thread_ = std::thread(visualizeSubmapBlocks, std::ref(cblock_submaps));

    bool seg_new_submap = false, local_map_recalculate_feature_on = false, motion_com_while_reg_on = false, apply_roi_filter = false, lo_status_healthy = true;
    int submap_count = 0, cooling_index = 0, accu_frame = 0, accu_frame_count_wo_opt = 0;
    float accu_tran = 0.0, accu_rot_deg = 0.0, current_linear_velocity = 0.0, current_angular_velocity = 0.0, add_length = 0.0, roi_min_y = 0.0, roi_max_y = 0.0;
    float non_max_suppresssion_radius = 0.25 * pca_neigh_r;

    ROS_INFO("load lidar points from file: %s", filenames[0].c_str());
    if (FLAGS_motion_compensation_method > 0)
        motion_com_while_reg_on = true;
    cblock_target->filename = filenames[0];
    cblock_target->unique_id = FLAGS_frame_num_begin;
    cblock_target->is_single_scanline = false; // multi-scanline lidar
    cblock_target->pose_lo.setIdentity();
    cblock_target->pose_gt.setIdentity();
    // dataio.check_overwrite_exsiting_file_or_not(output_adjacent_lo_pose_file);
    dataio.write_lo_pose_overwrite(cblock_target->pose_lo, output_lo_lidar_pose_file);
    dataio.write_lo_pose_overwrite(cblock_target->pose_gt, output_gt_lidar_pose_file);
    dataio.write_lo_pose_overwrite(first_frame_body, output_lo_body_pose_file);
    poses_lo_body_cs.push_back(first_frame_body);
    poses_lo_lidar_cs.push_back(cblock_target->pose_lo);
    if (FLAGS_gt_in_lidar_frame)
        poses_gt_lidar_cs[0] = cblock_target->pose_gt;
    else
        poses_gt_lidar_cs.push_back(cblock_target->pose_gt);
    dataio.read_pc_cloud_block(cblock_target);

    imGray = cv::imread(vstrImageFilenamesRGB[0], CV_LOAD_IMAGE_GRAYSCALE);
    mpVisualFeatureExtractor->ImagePreprocessing(imGray, cblock_target);
    {
        cv::Mat sam_img = imGray.clone();
        mpVisualFeatureExtractor->ShowLidarSamplePoints(sam_img, cblock_target, 1);
    }
    std::chrono::steady_clock::time_point tic_feature_extraction_init = std::chrono::steady_clock::now();
    if (FLAGS_apply_dist_filter)
    {
        cfilter.dist_filter(cblock_target->pc_raw, FLAGS_min_dist_used, FLAGS_max_dist_used);
    }

    // 激光点云内参标定
    if (FLAGS_vertical_ang_calib_on)
    {
        // intrinsic angle correction
        cfilter.vertical_intrinsic_calibration(cblock_target->pc_raw, FLAGS_vertical_ang_correction_deg);
    }

    
    // 激光点云特征提取
    cfilter.extract_semantic_pts(cblock_target, FLAGS_cloud_down_res, gf_grid_resolution, gf_max_grid_height_diff,
                                 gf_neighbor_height_diff, FLAGS_gf_max_h, ground_down_rate,
                                 nonground_down_rate, pca_neigh_r, pca_neigh_k,
                                 pca_linearity_thre, pca_planarity_thre, pca_curvature_thre,
                                 FLAGS_linearity_thre_down, FLAGS_planarity_thre_down, true,
                                 FLAGS_dist_inverse_sampling_method, FLAGS_unit_dist,
                                 FLAGS_ground_normal_method, FLAGS_gf_normal_estimation_radius,
                                 FLAGS_adaptive_parameters_on, FLAGS_apply_scanner_filter, FLAGS_detect_curb_or_not,
                                 FLAGS_vertex_extraction_method, gf_grid_min_pt_num, gf_reliable_neighbor_grid_thre,
                                 FLAGS_gf_down_down_rate, FLAGS_cloud_pca_neigh_k_min, FLAGS_pca_down_rate, FLAGS_intensity_thre_nonground,
                                 pillar_direction_sin, beam_direction_sin, roof_normal_sin, facade_normal_sin,
                                 FLAGS_sharpen_with_nms_on, FLAGS_fixed_num_downsampling_on, FLAGS_ground_down_fixed_num, FLAGS_pillar_down_fixed_num,
                                 FLAGS_facade_down_fixed_num, FLAGS_beam_down_fixed_num, FLAGS_roof_down_fixed_num, FLAGS_unground_down_fixed_num,
                                 FLAGS_beam_max_height, FLAGS_approx_scanner_height + 0.5, FLAGS_approx_scanner_height, FLAGS_underground_height_thre,
                                 FLAGS_feature_pts_ratio_guess, FLAGS_semantic_assist_on, apply_roi_filter, roi_min_y, roi_max_y);

    // 提取LOAM系列的特征点
    {
        cblock_target->cornerPointsSharp.reset(new pcT());
        cblock_target->cornerPointsLessSharp.reset(new pcT());
        cblock_target->surfPointsFlat.reset(new pcT());
        cblock_target->surfPointsLessFlat.reset(new pcT());
        double timestamp = ros::Time::now().toSec();
        mpLidarFeatureExtractor->ExtractLoamFeatures(*cblock_target->pc_unground, timestamp,
                                                     cblock_target->cornerPointsSharp, cblock_target->cornerPointsLessSharp,
                                                     cblock_target->surfPointsFlat, cblock_target->surfPointsLessFlat);

        odomEstimationNode.initMapWithPoints(cblock_target->cornerPointsLessSharp, cblock_target->surfPointsLessFlat);
    }

    std::chrono::steady_clock::time_point toc_feature_extraction_init = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used_init_feature_extraction = std::chrono::duration_cast<std::chrono::duration<double>>(toc_feature_extraction_init - tic_feature_extraction_init);
    timing_array[0].push_back(time_used_init_feature_extraction.count());
    for (int k = 0; k < 3; k++) // for the first frame, we only extract its feature points
        timing_array[0].push_back(0.0);

    initial_guess_tran(0, 3) = 0.5; // initialization

    int i = 1;
    ros::Rate rate(1000);
    while (ros::ok())
    {
        if (i < frame_num)
        {
            std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
            accu_frame_count_wo_opt++;

            // for the first scan matching (no initial guess avaliable --> larger corr distance)
            if (i == 1)
                add_length = 1.0;

            // 读取激光雷达数据
            cblock_source->filename = filenames[i];
            cblock_source->unique_id = i * FLAGS_frame_step + FLAGS_frame_num_begin;
            cblock_source->is_single_scanline = false;
            if (poses_gt_body_cs.size() > 0)
            {                                                                                                                     // check if gt pose is availiable
                cblock_source->pose_gt = calib_mat.inverse() * (poses_gt_body_cs[0].inverse() * poses_gt_body_cs[i]) * calib_mat; // set ground truth pose (from body to lidar coordinate system)
                if (FLAGS_gt_in_lidar_frame)
                    cblock_source->pose_gt = init_poses_gt_lidar_cs.inverse() * poses_gt_lidar_cs[i];
                poses_gt_body_cs[i] = poses_gt_body_cs[0].inverse() * poses_gt_body_cs[i]; // according to the first frame
            }
            dataio.read_pc_cloud_block(cblock_source);

            // 读取图像
            imGray = cv::imread(vstrImageFilenamesRGB[i], CV_LOAD_IMAGE_GRAYSCALE);
            publishImage(pub_img_, imGray);
            std::chrono::steady_clock::time_point toc_import_pc = std::chrono::steady_clock::now();

            // 视觉特征提取
            mpVisualFeatureExtractor->ImagePreprocessing(imGray, cblock_source);
            {
                cv::Mat sam_img = imGray.clone();
                mpVisualFeatureExtractor->ShowLidarSamplePoints(sam_img, cblock_source, 1);
            }

            std::chrono::steady_clock::time_point toc_visual_feature_extraction = std::chrono::steady_clock::now();

            // 距离滤波
            if (FLAGS_apply_dist_filter)
                cfilter.dist_filter(cblock_source->pc_raw, FLAGS_min_dist_used, FLAGS_max_dist_used);
            if (FLAGS_vertical_ang_calib_on) // intrinsic angle correction
                cfilter.vertical_intrinsic_calibration(cblock_source->pc_raw, FLAGS_vertical_ang_correction_deg);

            // motion compensation [first step: using the last frame's transformation]
            if (FLAGS_motion_compensation_method == 1) // calculate from time-stamp
                cfilter.get_pts_timestamp_ratio_in_frame(cblock_source->pc_raw, true);
            else if (FLAGS_motion_compensation_method == 2)                                   // calculate from azimuth
                cfilter.get_pts_timestamp_ratio_in_frame(cblock_source->pc_raw, false, 90.0); // HESAI Lidar: 90.0 (y+ axis, clockwise), Velodyne Lidar: 180.0

            // 提取特征
            cfilter.extract_semantic_pts(cblock_source, FLAGS_cloud_down_res, gf_grid_resolution, gf_max_grid_height_diff,
                                         gf_neighbor_height_diff, FLAGS_gf_max_h, ground_down_rate,
                                         nonground_down_rate, pca_neigh_r, pca_neigh_k, pca_linearity_thre, pca_planarity_thre, pca_curvature_thre,
                                         FLAGS_linearity_thre_down, FLAGS_planarity_thre_down, true, FLAGS_dist_inverse_sampling_method, FLAGS_unit_dist,
                                         FLAGS_ground_normal_method, FLAGS_gf_normal_estimation_radius, FLAGS_adaptive_parameters_on, FLAGS_apply_scanner_filter, FLAGS_detect_curb_or_not,
                                         FLAGS_vertex_extraction_method, gf_grid_min_pt_num, gf_reliable_neighbor_grid_thre,
                                         FLAGS_gf_down_down_rate, FLAGS_cloud_pca_neigh_k_min, FLAGS_pca_down_rate, FLAGS_intensity_thre_nonground,
                                         pillar_direction_sin, beam_direction_sin, roof_normal_sin, facade_normal_sin, FLAGS_sharpen_with_nms_on, FLAGS_fixed_num_downsampling_on, FLAGS_ground_down_fixed_num,
                                         FLAGS_pillar_down_fixed_num, FLAGS_facade_down_fixed_num, FLAGS_beam_down_fixed_num, FLAGS_roof_down_fixed_num, FLAGS_unground_down_fixed_num,
                                         FLAGS_beam_max_height, FLAGS_approx_scanner_height + 0.5, FLAGS_approx_scanner_height, FLAGS_underground_height_thre, FLAGS_feature_pts_ratio_guess, FLAGS_semantic_assist_on,
                                         apply_roi_filter, roi_min_y, roi_max_y);

            // 提取LOAM系列的特征点
            {
                cblock_source->cornerPointsSharp.reset(new pcT());
                cblock_source->cornerPointsLessSharp.reset(new pcT());
                cblock_source->surfPointsFlat.reset(new pcT());
                cblock_source->surfPointsLessFlat.reset(new pcT());
                double timestamp = ros::Time::now().toSec();
                mpLidarFeatureExtractor->ExtractLoamFeatures(*cblock_source->pc_unground, timestamp,
                                                             cblock_source->cornerPointsSharp, cblock_source->cornerPointsLessSharp,
                                                             cblock_source->surfPointsFlat, cblock_source->surfPointsLessFlat);

                // 使用LOAM的方式进行运动估计
                odomEstimationNode.updatePointsToMap(cblock_source->cornerPointsLessSharp, cblock_source->surfPointsLessFlat);
            }

            std::chrono::steady_clock::time_point toc_lidar_feature_extraction = std::chrono::steady_clock::now();

            // update local map(利用上一帧数据更新局部地图)
            if (i % FLAGS_local_map_recalculation_frequency == 0)
                local_map_recalculate_feature_on = true;
            else
                local_map_recalculate_feature_on = false;

            if (i > FLAGS_initial_scan2scan_frame_num + 1)
                mmanager.update_local_map(cblock_local_map, cblock_target, local_map_radius, local_map_max_pt_num, vertex_keeping_num, append_frame_radius,
                                          FLAGS_apply_map_based_dynamic_removal, FLAGS_used_feature_type, dynamic_removal_radius, dynamic_dist_thre_min, current_linear_velocity * 0.15,
                                          FLAGS_map_min_dist_within_feature, local_map_recalculate_feature_on); // 1.5 * 0.1 * velocity (set as the max distance threshold for dynamic obejcts)
            else
                mmanager.update_local_map(cblock_local_map, cblock_target, local_map_radius, local_map_max_pt_num, vertex_keeping_num, append_frame_radius, false, FLAGS_used_feature_type);

            // 局部地图拼接
            pcTPtr pc_raw_w_tmp(new pcT());
            pcTPtr pc_raw_w_tmp_filter(new pcT());
            pcl::transformPointCloudWithNormals(*cblock_target->pc_unground, *pc_raw_w_tmp, cblock_target->pose_lo);
            // pcl::transformPointCloudWithNormals(*cblock_target->pc_raw, *pc_raw_w_tmp, cblock_target->pose_lo);
            cfilter.voxel_downsample(pc_raw_w_tmp, pc_raw_w_tmp_filter, 0.1);
            *cblock_submap_raw += *pc_raw_w_tmp_filter;

            int temp_accu_frame = accu_frame;
            if (loop_closure_detection_on) // determine if we can add a new submap
                seg_new_submap = mmanager.judge_new_submap(accu_tran, accu_rot_deg, accu_frame, FLAGS_submap_accu_tran, FLAGS_submap_accu_rot, FLAGS_submap_accu_frame);
            else
                seg_new_submap = false;
            std::chrono::steady_clock::time_point toc_update_map = std::chrono::steady_clock::now();

            if (seg_new_submap) // add nodes and edges in pose graph for pose graph optimization (pgo)
            {
                ROS_WARN("Create new submap [ %d ]", submap_count);
                cloudblock_Ptr current_cblock_local_map(new cloudblock_t(*cblock_local_map, true, false));
                current_cblock_local_map->strip_id = 0;
                current_cblock_local_map->id_in_strip = submap_count;
                current_cblock_local_map->last_frame_index = i - 1; // save the last frame index
                current_cblock_local_map->unique_id = cblock_target->unique_id;
                current_cblock_local_map->pose_init = current_cblock_local_map->pose_lo;                                            // init guess for pgo
                current_cblock_local_map->information_matrix_to_next = 1.0 / temp_accu_frame * scan2map_reg_con.information_matrix; // not used now //Use the correct function by deducing Jacbobian of transformation

                char submap_filename[128] = {0};
                pcTPtr pc_raw_w_local(new pcT());
                Eigen::Matrix4d Tlw = cblock_target->pose_lo.inverse();
                pcl::transformPointCloudWithNormals(*cblock_submap_raw, *pc_raw_w_local, Tlw);
                cblock_submap_raw.reset(new pcT());

                cv::Mat ri_temp, ri_temp_cm;
                cfilter.pointcloud_to_rangeimage(pc_raw_w_local, ri_temp);
                display_image(ri_temp, ri_temp_cm, 1);
                publishImage(pub_range_img_, ri_temp_cm);

                cv::Mat bev_temp, bev_temp_cm;
                bev_temp = cfilter.generate_2d_map(pc_raw_w_local, 4, 400, 400, 2, 50, -5.0, 15.0);
                display_image(bev_temp, ri_temp_cm, 0);
                publishImage(pub_bev_img_, ri_temp_cm);

                // char file_name[128] = {0};
                // sprintf(file_name, "/home/kinggreat24/pc/MULLS/%d_bv_image.png", current_cblock_local_map->unique_id);
                // cv::imwrite(file_name, bev_temp);

                // 对BEV图像进行特征提取
                mpVisualFeatureExtractor->ExtractORB(bev_temp, current_cblock_local_map);

                // encode features in the submap (this is already done in each frame)
                cfilter.non_max_suppress(current_cblock_local_map->pc_vertex, non_max_suppresssion_radius, false, current_cblock_local_map->tree_vertex);
                if (submap_count == 0)
                    current_cblock_local_map->pose_fixed = true;    // fixed the first submap
                cblock_submaps.push_back(current_cblock_local_map); // add new node
                submap_count++;
                cooling_index--;
                if (submap_count > 1)
                {
                    // (1) delete the past registration edges (which are not reliable)
                    confinder.cancel_registration_constraint(pgo_edges);

                    // (2) add adjacent edge between current submap and the last submap
                    confinder.add_adjacent_constraint(cblock_submaps, pgo_edges, submap_count);

                    // (3) do adjacent map to map registration
                    int current_edge_index = pgo_edges.size() - 1;

                    // 相邻子图之间的位姿估计
                    int registration_status_map2map =
                        creg.mm_lls_icp(pgo_edges[current_edge_index], max_iteration_num_m2m, 1.5 * reg_corr_dis_thre_init,
                                        converge_tran, converge_rot_d, 1.5 * reg_corr_dis_thre_min, dis_thre_update_rate,
                                        FLAGS_used_feature_type, "1101", z_xy_balance_ratio,
                                        2 * pt2pt_residual_window, 2 * pt2pl_residual_window, 2 * pt2li_residual_window,
                                        pgo_edges[current_edge_index].Trans1_2, FLAGS_reg_intersection_filter_on, false,
                                        FLAGS_normal_shooting_on, 1.5 * FLAGS_normal_bearing, true, true, FLAGS_post_sigma_thre); // use its information matrix for pgo

                    if (registration_status_map2map <= 0) // candidate wrong registration
                    {
                    }
                    else
                    {
                        ROS_INFO("map to map registration done\nsubmap [ %d ] - [ %d ]",
                                 pgo_edges[current_edge_index].block1->id_in_strip,
                                 pgo_edges[current_edge_index].block2->id_in_strip);
                    }

                    // 位姿图
                    if (pgo_edges[current_edge_index].sigma > FLAGS_map2map_reliable_sigma_thre) // if the estimated posterior standard deviation of map to map registration is a bit large
                    {
                        pgo_edges[current_edge_index].Trans1_2 = pgo_edges[current_edge_index].block1->pose_lo.inverse() *
                                                                 pgo_edges[current_edge_index].block2->pose_lo; // still use the scan-to-map odometry's prediction (but we will keep the information matrix)
                    }
                    else
                    {
                        ROS_WARN("We would trust the map to map registration, update current pose");                                        // the edge's transformation and information are already calculted via the last map-to-map registration
                        cblock_local_map->pose_lo = pgo_edges[current_edge_index].block1->pose_lo * pgo_edges[current_edge_index].Trans1_2; // update current local map's pose
                        cblock_target->pose_lo = cblock_local_map->pose_lo;                                                                 // update target frame
                        cblock_submaps[submap_count - 1]->pose_lo = cblock_local_map->pose_lo;                                              // update the pgo node (submap)
                        cblock_submaps[submap_count - 1]->pose_init = cblock_local_map->pose_lo;                                            // update the initial guess of pgo node (submap)
                    }

                    pgo_edges[current_edge_index].information_matrix = FLAGS_adjacent_edge_weight_ratio * pgo_edges[current_edge_index].information_matrix; // TODO: fix (change the weight of the weight of adjacent edges)
                    constraints current_registration_edges;
                    if (cooling_index < 0) // find registration edges and then do pgo
                    {
                        bool overall_loop_searching_on = false;
                        int reg_edge_count = 0;
                        if (accu_frame_count_wo_opt > FLAGS_num_frame_thre_large_drift && FLAGS_overall_loop_closure_searching_on) // expand the loop closure searching area
                        {
                            overall_loop_searching_on = true;
                            reg_edge_count = confinder.find_overlap_registration_constraint(cblock_submaps, current_registration_edges, 1.5 * FLAGS_neighbor_search_dist, 0.0, FLAGS_min_submap_id_diff, true, 20);
                        }
                        else // standard loop closure searching
                        {
                            reg_edge_count = confinder.find_overlap_registration_constraint(cblock_submaps, current_registration_edges, FLAGS_neighbor_search_dist, FLAGS_min_iou_thre, FLAGS_min_submap_id_diff, true);
                        }

                        int reg_edge_successful_count = 0;
                        bool stable_reg_found = false;
                        // suppose node 3 is the current submap, node 1 and node 2 are two history submaps with loop constraints with node 3,
                        // suppose node 1 and node 3 's transformation is already known (a stable registration)
                        Eigen::Matrix4d reference_pose_mat;      // the pose of the reference node (node 1), Tw1
                        Eigen::Matrix4d reference_loop_tran_mat; // the loop transformation of the reference node , T13
                        ROS_INFO("registration edges: %d", reg_edge_count);
                        for (int j = 0; j < reg_edge_count; j++)
                        {
                            // we do not need too many registration edges (for example, more than 3 reg edges)
                            if (reg_edge_successful_count >= FLAGS_max_used_reg_edge_per_optimization)
                                break;

                            pcTPtr cur_map_origin(new pcT()), cur_map_guess(new pcT()), cur_map_tran(new pcT()), hist_map(new pcT()), kp_guess(new pcT());
                            pcTPtr target_cor(new pcT()), source_cor(new pcT());
                            current_registration_edges[j].block2->merge_feature_points(cur_map_origin, false);
                            current_registration_edges[j].block1->merge_feature_points(hist_map, false);
                            Eigen::Matrix4d init_mat = current_registration_edges[j].Trans1_2;
                            if (stable_reg_found)                                                                                                  // infer the init guess according to a already successfully registered loop edge for current submap
                                init_mat = current_registration_edges[j].block1->pose_lo.inverse() * reference_pose_mat * reference_loop_tran_mat; // T23 = T21 * T13 = T2w * Tw1 * T13

                            // global (coarse) registration by teaser or ransac (using ncc, bsc or fpfh as feature)
                            bool global_reg_on = false;
                            if (!stable_reg_found && (current_registration_edges[j].overlapping_ratio > FLAGS_min_iou_thre_global_reg || overall_loop_searching_on))
                            {
                                // with higher overlapping ratio, try to TEASER
                                creg.find_feature_correspondence_ncc(current_registration_edges[j].block1->pc_vertex, current_registration_edges[j].block2->pc_vertex,
                                                                     target_cor, source_cor, FLAGS_best_n_feature_match_on, FLAGS_feature_corr_num, FLAGS_reciprocal_feature_match_on);

                                int global_reg_status = -1;
                                if (FLAGS_teaser_based_global_registration_on)
                                    global_reg_status = creg.coarse_reg_teaser(target_cor, source_cor, init_mat, pca_neigh_r, FLAGS_global_reg_min_inlier_count);
                                else // using ransac, a bit slower than teaser
                                    global_reg_status = creg.coarse_reg_ransac(target_cor, source_cor, init_mat, pca_neigh_r, FLAGS_global_reg_min_inlier_count);

                                if (global_reg_status == 0) // double check
                                {
                                    if (overall_loop_searching_on)
                                        global_reg_on = confinder.double_check_tran(init_mat, current_registration_edges[j].Trans1_2, init_mat, 10.0 * FLAGS_wrong_edge_tran_thre, 6.0 * FLAGS_wrong_edge_rot_thre_deg); // the difference tolerance can be a bit larger
                                    else
                                        global_reg_on = confinder.double_check_tran(init_mat, current_registration_edges[j].Trans1_2, init_mat, 3.0 * FLAGS_wrong_edge_tran_thre, 3.0 * FLAGS_wrong_edge_rot_thre_deg); // if the difference of the transformation of teaser and lo initial guess is too large, we will trust lidar odometry
                                }
                                else if (global_reg_status == 1) // reliable
                                    global_reg_on = true;
                            }

                            // TODO: the tolerance should be determine using pose covariance (reference: overlapnet)
                            // without the global registration and since the lidar odometry may drift a lot, the inital guess may not be reliable, just skip
                            if (!global_reg_on && !stable_reg_found && accu_frame_count_wo_opt > FLAGS_num_frame_thre_large_drift)
                                continue;

                            int registration_status_map2map =
                                creg.mm_lls_icp(current_registration_edges[j], max_iteration_num_m2m, 3.5 * reg_corr_dis_thre_init,
                                                converge_tran, converge_rot_d, 2.0 * reg_corr_dis_thre_min, dis_thre_update_rate,
                                                FLAGS_used_feature_type, "1101", z_xy_balance_ratio,
                                                2 * pt2pt_residual_window, 2 * pt2pl_residual_window, 2 * pt2li_residual_window,
                                                init_mat, FLAGS_reg_intersection_filter_on, false,
                                                FLAGS_normal_shooting_on, 1.5 * FLAGS_normal_bearing,
                                                true, true, FLAGS_post_sigma_thre, FLAGS_map_to_map_min_cor_ratio);

                            ROS_INFO("registration: %d, results: %d", j, registration_status_map2map);
                            if (registration_status_map2map > 0)
                            {
                                ROS_WARN("map to map registration done, submap [ %d ] - [ %d ]!",
                                         current_registration_edges[j].block1->id_in_strip,
                                         current_registration_edges[j].block2->id_in_strip);

                                pgo_edges.push_back(current_registration_edges[j]);
                                reg_edge_successful_count++; // putable correctly registered registration edge

                                if (!stable_reg_found && FLAGS_transfer_correct_reg_tran_on)
                                {
                                    reference_pose_mat = current_registration_edges[j].block1->pose_lo; // the correct registered loop closure history node's pose
                                    reference_loop_tran_mat = current_registration_edges[j].Trans1_2;   // the correct registered loop closure's transformation
                                    stable_reg_found = true;                                            // first time global registration, then local registration
                                }
                            }

                            std::cout << "map to map registration done\nsubmap [" << current_registration_edges[j].block1->id_in_strip << "] - [" << current_registration_edges[j].block2->id_in_strip << "]:\n"
                                      << current_registration_edges[j].Trans1_2 << std::endl;

                            pcT().swap(*cur_map_origin);
                            pcT().swap(*cur_map_guess);
                            pcT().swap(*cur_map_tran);
                            pcT().swap(*hist_map);
                        }

                        // apply pose graph optimization (pgo) only when there's correctly registered registration edge
                        if (reg_edge_successful_count > 0)
                        {
                            ROS_ERROR("pose graph optimization: edge_successful");
                            pgoptimizer.set_robust_function(FLAGS_robust_kernel_on);
                            pgoptimizer.set_equal_weight(FLAGS_equal_weight_on);
                            pgoptimizer.set_max_iter_num(FLAGS_max_iter_inter_submap);
                            pgoptimizer.set_diagonal_information_matrix(FLAGS_diagonal_information_matrix_on);
                            pgoptimizer.set_free_node(FLAGS_free_node_on);
                            pgoptimizer.set_problem_size(false);
                            bool pgo_successful;
                            if (!strcmp(FLAGS_pose_graph_optimization_method.c_str(), "g2o"))
                                pgo_successful = pgoptimizer.optimize_pose_graph_g2o(cblock_submaps, pgo_edges);
                            else if (!strcmp(FLAGS_pose_graph_optimization_method.c_str(), "ceres"))
                                pgo_successful = pgoptimizer.optimize_pose_graph_ceres(cblock_submaps, pgo_edges, FLAGS_inter_submap_t_limit, FLAGS_inter_submap_r_limit);
                            else if (!strcmp(FLAGS_pose_graph_optimization_method.c_str(), "gtsam"))
                                pgo_successful = pgoptimizer.optimize_pose_graph_gtsam(cblock_submaps, pgo_edges); // TODO: you'd better use gtsam instead (just like lego-loam)
                            else                                                                                   // default: ceres
                                pgo_successful = pgoptimizer.optimize_pose_graph_ceres(cblock_submaps, pgo_edges, FLAGS_inter_submap_t_limit, FLAGS_inter_submap_r_limit);
                            if (pgo_successful)
                            {
                                pgoptimizer.update_optimized_nodes(cblock_submaps, true, true);        // let pose_lo = = pose_init = pose_optimized && update bbx at the same time
                                cblock_local_map->pose_lo = cblock_submaps[submap_count - 1]->pose_lo; // update current local map's pose
                                cblock_target->pose_lo = cblock_local_map->pose_lo;                    // update target frame
                                cooling_index = FLAGS_cooling_submap_num;                              // wait for several submap (without loop closure detection)
                                for (int k = 0; k < cblock_submaps.size(); k++)
                                    cblock_submaps[k]->pose_stable = true;
                                accu_frame_count_wo_opt = 0; // clear
                            }
                        }
                    }
                    constraints().swap(current_registration_edges);
                }
            } // end of seg_new_submap
            std::chrono::steady_clock::time_point toc_loop_closure = std::chrono::steady_clock::now();

            /*********************************************************************************/
            /**************************      scan 2 scan registration    ***********************/
            /*********************************************************************************/
            // scan to scan registration
            if (FLAGS_scan_to_scan_module_on || i <= FLAGS_initial_scan2scan_frame_num)
            {
                // ROS_INFO("scan 2 scan registration: %d", i);
                creg.assign_source_target_cloud(cblock_target, cblock_source, scan2scan_reg_con);

                int registration_status_scan2scan =
                    creg.mm_lls_icp(scan2scan_reg_con, max_iteration_num_s2s, reg_corr_dis_thre_init + add_length,
                                    converge_tran, converge_rot_d, reg_corr_dis_thre_min, dis_thre_update_rate,
                                    FLAGS_used_feature_type, FLAGS_corr_weight_strategy, z_xy_balance_ratio,
                                    pt2pt_residual_window, pt2pl_residual_window, pt2li_residual_window,
                                    initial_guess_tran, FLAGS_reg_intersection_filter_on,
                                    motion_com_while_reg_on, FLAGS_normal_shooting_on, FLAGS_normal_bearing,
                                    false, false, FLAGS_post_sigma_thre);

                if (registration_status_scan2scan < 0) // candidate wrong registration --> use baseline method instead to avoid the crash of the system
                {
                    add_length = 0.8;
                    lo_status_healthy = false;
                }
                else
                    add_length = 1.0;

                if (FLAGS_zupt_on_or_not)
                    nav.zupt_simple(scan2scan_reg_con.Trans1_2);
                cblock_source->pose_lo = cblock_target->pose_lo * scan2scan_reg_con.Trans1_2;
                // std::cout << "scan to scan registration done\nframe [" << i - 1 << "] - [" << i << "]:\n"
                //           << scan2scan_reg_con.Trans1_2 << std::endl;
                initial_guess_tran = scan2scan_reg_con.Trans1_2;
            }

#if (USE_DVT)
            // 直接法测试
            {
                Eigen::Matrix4d T_adjacent_pose_cam = Eigen::Matrix4d::Identity();
                relative_pose_transform(adjacent_pose_out, T_cam_lidar, T_adjacent_pose_cam);

                Sophus::SE3 Tcl = lo::toSophusSE3(T_adjacent_pose_cam);
                mpSparseVisualLidarAlign_->tracking(cblock_target, cblock_source, Tcl);

                // Eigen::Matrix4d T_adjacent_pose_lidar = Eigen::Matrix4d::Identity();
                // relative_pose_transform(Tcl.matrix(), T_cam_lidar.inverse(), T_adjacent_pose_lidar);

                // add_length = 1.0;
                // cblock_source->pose_lo = cblock_target->pose_lo * T_adjacent_pose_lidar.inverse();
                // initial_guess_tran = T_adjacent_pose_lidar.inverse();
            }
#endif // USE_DVT

            /*********************************************************************************/
            /**************************      scan to map registration    ***********************/
            /*********************************************************************************/
            if (i % FLAGS_s2m_frequency == 0 && i > FLAGS_initial_scan2scan_frame_num)
            {
                // ROS_INFO("scan 2 map registration: %d", i);
                creg.assign_source_target_cloud(cblock_local_map, cblock_source, scan2map_reg_con);

                int registration_status_scan2map = creg.mm_lls_icp(scan2map_reg_con, max_iteration_num_s2m, reg_corr_dis_thre_init + add_length,
                                                                   converge_tran, converge_rot_d, reg_corr_dis_thre_min, dis_thre_update_rate,
                                                                   FLAGS_used_feature_type, FLAGS_corr_weight_strategy, z_xy_balance_ratio,
                                                                   pt2pt_residual_window, pt2pl_residual_window, pt2li_residual_window,
                                                                   initial_guess_tran, FLAGS_reg_intersection_filter_on,
                                                                   motion_com_while_reg_on, FLAGS_normal_shooting_on, FLAGS_normal_bearing,
                                                                   false, false, FLAGS_post_sigma_thre);
                if (registration_status_scan2map < 0)
                {
                    // candidate wrong registration
                    add_length = 1.0;
                    lo_status_healthy = false;
                }
                else
                    add_length = 0.0;

                if (FLAGS_zupt_on_or_not)
                    nav.zupt_simple(scan2map_reg_con.Trans1_2);
                cblock_source->pose_lo = cblock_local_map->pose_lo * scan2map_reg_con.Trans1_2;
                LOG(INFO) << "scan to map registration done\nframe [" << i << "]:\n"
                          << scan2map_reg_con.Trans1_2;
            }
            // adjacent_pose_out is the transformation from k to k+1 frame (T2_1)
            adjacent_pose_out = cblock_source->pose_lo.inverse() * cblock_target->pose_lo;

            std::chrono::steady_clock::time_point toc_registration = std::chrono::steady_clock::now();

            // visualization
            if (FLAGS_real_time_viewer_on)
            {
                // 发布位姿信息
                PublishTF(cblock_source->pose_lo);

                // 发布激光点云特征
                publishLidarFeatures(cblock_source, true);
            }

            // 运动畸变补偿
            if (motion_com_while_reg_on)
            {
                std::chrono::steady_clock::time_point tic_undistortion = std::chrono::steady_clock::now();
                cfilter.apply_motion_compensation(cblock_source->pc_raw, adjacent_pose_out);
                cfilter.batch_apply_motion_compensation(cblock_source->pc_ground, cblock_source->pc_pillar, cblock_source->pc_facade,
                                                        cblock_source->pc_beam, cblock_source->pc_roof, cblock_source->pc_vertex, adjacent_pose_out);
                cfilter.batch_apply_motion_compensation(cblock_source->pc_ground_down, cblock_source->pc_pillar_down, cblock_source->pc_facade_down,
                                                        cblock_source->pc_beam_down, cblock_source->pc_roof_down, cblock_source->pc_vertex, adjacent_pose_out); // TODO: we do not need vertex feature (the curvature of vertex stores the descriptor instead of the timestamp)
                std::chrono::steady_clock::time_point toc_undistortion = std::chrono::steady_clock::now();
                std::chrono::duration<double> time_used_undistortion = std::chrono::duration_cast<std::chrono::duration<double>>(toc_undistortion - tic_undistortion);
                LOG(INFO) << "map motion compensation done in [" << 1000.0 * time_used_undistortion.count() << "] ms.\n";
            }
            std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();

            if (FLAGS_real_time_viewer_on) // visualization
            {
                std::chrono::steady_clock::time_point tic_vis = std::chrono::steady_clock::now();
                if (FLAGS_show_range_image)
                {
                    // pc_raw
                    cv::Mat ri_temp;
                    cfilter.pointcloud_to_rangeimage(cblock_target->pc_raw, ri_temp);

                    cv::Mat ri_temp_cm;
                    display_image(ri_temp, ri_temp_cm, 1);
                    publishImage(pub_range_img_, ri_temp_cm);
                }
                if (FLAGS_show_bev_image)
                {
                    cv::Mat bev_temp;
                    bev_temp = cfilter.generate_2d_map(cblock_target->pc_unground, 4, 400, 400, 2, 50, -5.0, 15.0);

                    cv::Mat bev_temp_cm;
                    display_image(bev_temp, bev_temp_cm, 0);
                    publishImage(pub_bev_img_, bev_temp_cm);
                }

                // 发布局部地图
                PublishCurrentLocalMap_realtime(cblock_local_map, false);

                std::chrono::steady_clock::time_point toc_vis = std::chrono::steady_clock::now();
                std::chrono::duration<double> vis_time_used_per_frame = std::chrono::duration_cast<std::chrono::duration<double>>(toc_vis - tic_vis);
                LOG(INFO) << "Render frame [" << i << "] in [" << 1000.0 * vis_time_used_per_frame.count() << "] ms.\n";
            }

            // 发布局部位姿图
            visualizePoseGraph(&pub_pgo_markers_, ros::Time::now(), pgo_edges);

            std::chrono::steady_clock::time_point tic_output = std::chrono::steady_clock::now();

            // write out pose
            if (!output_adjacent_lo_pose_file.empty())
            {
                if (i == 1)
                {
                    dataio.write_lo_pose_overwrite(adjacent_pose_out, output_adjacent_lo_pose_file);
                }
                else
                    dataio.write_lo_pose_append(adjacent_pose_out, output_adjacent_lo_pose_file);
            }

            Eigen::Matrix4d pose_lo_body_frame;
            pose_lo_body_frame = calib_mat * cblock_source->pose_lo * calib_mat.inverse();
            dataio.write_lo_pose_append(cblock_source->pose_lo, output_lo_lidar_pose_file); // lo pose in lidar frame
            dataio.write_lo_pose_append(cblock_source->pose_gt, output_gt_lidar_pose_file); // gt pose in lidar frame
            dataio.write_lo_pose_append(pose_lo_body_frame, output_lo_body_pose_file);      // lo pose in body frame (requried by KITTI)
            poses_lo_lidar_cs.push_back(cblock_source->pose_lo);
            if (FLAGS_gt_in_lidar_frame)
                poses_gt_lidar_cs[i] = cblock_source->pose_gt;
            else
                poses_gt_lidar_cs.push_back(cblock_source->pose_gt);
            poses_lo_body_cs.push_back(pose_lo_body_frame);
            poses_lo_adjacent.push_back(adjacent_pose_out); // poses_lo_adjacent is the container of adjacent_pose_out

            // update initial guess
            initial_guess_tran.setIdentity();
            if (initial_guess_mode == 1 && lo_status_healthy)
                initial_guess_tran.block<3, 1>(0, 3) = adjacent_pose_out.inverse().block<3, 1>(0, 3); // uniform motion model
            else if (initial_guess_mode == 2 && lo_status_healthy)
                initial_guess_tran = adjacent_pose_out.inverse();

            // save current frame (only metadata)
            cloudblock_Ptr current_cblock_frame(new cloudblock_t(*cblock_target));
            current_cblock_frame->pose_optimized = current_cblock_frame->pose_lo;
            if (i % FLAGS_s2m_frequency == 0 && i > FLAGS_initial_scan2scan_frame_num) // scan-to-map reg on
                current_cblock_frame->information_matrix_to_next = scan2map_reg_con.information_matrix;
            else // scan-to-map reg off (only scan-to-scan)
                current_cblock_frame->information_matrix_to_next = scan2scan_reg_con.information_matrix;
            cblock_frames.push_back(current_cblock_frame);
            // use this frame as the next iter's target frame
            cblock_target.swap(cblock_source);
            // std::cout << "cblock_target point 0 size: " << cblock_target->pc_raw->size() << std::endl;
            cblock_source->free_all();
            lo_status_healthy = true;

            // cv::imshow("target_image",cblock_target->mvImgPyramid[0]);
            // cv::waitKey(1);

            // update accumulated information
            accu_tran += nav.cal_translation_from_tranmat(adjacent_pose_out);
            accu_rot_deg += nav.cal_rotation_deg_from_tranmat(adjacent_pose_out);
            accu_frame += FLAGS_frame_step;
            current_linear_velocity = nav.cal_velocity(poses_lo_adjacent);

            // report timing
            std::chrono::steady_clock::time_point toc_output = std::chrono::steady_clock::now();
            std::chrono::duration<double> time_use_visual_feature_extraction = std::chrono::duration_cast<std::chrono::duration<double>>(toc_visual_feature_extraction - toc_import_pc);
            std::chrono::duration<double> time_use_lidar_feature_extraction = std::chrono::duration_cast<std::chrono::duration<double>>(toc_lidar_feature_extraction - toc_visual_feature_extraction);
            std::chrono::duration<double> time_used_per_frame_lo_1 = std::chrono::duration_cast<std::chrono::duration<double>>(toc_update_map - toc_visual_feature_extraction);
            std::chrono::duration<double> time_used_per_frame_lo_2 = std::chrono::duration_cast<std::chrono::duration<double>>(toc_registration - toc_loop_closure);
            std::chrono::duration<double> time_used_per_frame_1 = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);
            std::chrono::duration<double> time_used_per_frame_2 = std::chrono::duration_cast<std::chrono::duration<double>>(toc_output - tic_output);
            time_count += (time_used_per_frame_lo_1.count() + time_used_per_frame_lo_2.count());

            visual_feature_extraction_count += time_use_visual_feature_extraction.count();
            lidar_feature_extraction_count += time_use_lidar_feature_extraction.count();
            lidar_registration_count += time_used_per_frame_lo_2.count();
            LOG(INFO) << "Consuming time of lidar odometry for current frame is [" << 1000.0 * (time_used_per_frame_lo_1.count() + time_used_per_frame_lo_2.count()) << "] ms.\n";
            LOG(INFO) << "Process frame (including data IO) [" << i << "] in [" << 1000.0 * (time_used_per_frame_1.count() + time_used_per_frame_2.count()) << "] ms.\n";

            // record timing
            std::chrono::duration<double> time_used_per_frame_feature_extraction = std::chrono::duration_cast<std::chrono::duration<double>>(toc_lidar_feature_extraction - toc_visual_feature_extraction);
            std::chrono::duration<double> time_used_per_frame_map_updating = std::chrono::duration_cast<std::chrono::duration<double>>(toc_update_map - toc_lidar_feature_extraction);
            std::chrono::duration<double> time_used_per_frame_loop_closure = std::chrono::duration_cast<std::chrono::duration<double>>(toc_loop_closure - toc_update_map);
            std::chrono::duration<double> time_used_per_frame_registration = std::chrono::duration_cast<std::chrono::duration<double>>(toc_registration - toc_loop_closure);
            timing_array[i].push_back(time_used_per_frame_feature_extraction.count());
            timing_array[i].push_back(time_used_per_frame_map_updating.count());
            timing_array[i].push_back(time_used_per_frame_registration.count());
            timing_array[i].push_back(time_used_per_frame_loop_closure.count());

            /**********************************************/
            i++;
            rate.sleep();
            ros::spinOnce();
        } //  end of frame-to-frame tracking
        else
        {
            //
            static bool flag = false;
            if (!flag)
            {
                flag = true;

                cloudblock_Ptr current_cblock_frame(new cloudblock_t(*cblock_target));
                current_cblock_frame->pose_optimized = current_cblock_frame->pose_lo;
                cblock_frames.push_back(current_cblock_frame);

                ROS_INFO("Lidar Odometry done. Average processing time per frame is [ %f ] ms over [ %d ] frames, avg visual feature extraction time: [ %f ] ,avg lidar feature extraction time: [ %f ], avg lidar registration time: [ %f ]",
                         1000.0 * time_count / frame_num, frame_num, 1000.0 * visual_feature_extraction_count / frame_num, 1000.0 * lidar_feature_extraction_count / frame_num, 1000.0 * lidar_registration_count / frame_num);

                if (loop_closure_detection_on)
                {
                    std::chrono::steady_clock::time_point tic_inner_submap_refine = std::chrono::steady_clock::now();
                    if (FLAGS_framewise_pgo_on)
                    {
                        // method 1: pgo of all the frame nodes

                        pgoptimizer.set_robust_function(FLAGS_robust_kernel_on);
                        pgoptimizer.set_equal_weight(FLAGS_equal_weight_on);
                        pgoptimizer.set_max_iter_num(FLAGS_max_iter_inner_submap);
                        pgoptimizer.set_diagonal_information_matrix(FLAGS_diagonal_information_matrix_on);
                        pgoptimizer.set_free_node(FLAGS_free_node_on);
                        pgoptimizer.set_problem_size(false); // large size problem
                        constraints framewise_pgo_edges;
                        cblock_frames[0]->pose_fixed = true; // fix the first frame
                        for (int i = 0; i < cblock_frames.size(); i++)
                        {
                            cblock_frames[i]->id_in_strip = i;
                            cblock_frames[i]->pose_init = cblock_frames[i]->pose_lo;
                            if (i < cblock_frames.size() - 1)
                            {
                                Eigen::Matrix4d tran_mat_12 = poses_lo_adjacent[i].inverse();
                                confinder.add_adjacent_constraint(cblock_frames, framewise_pgo_edges, tran_mat_12, i + 2);
                            }
                        }
                        for (int i = 0; i < pgo_edges.size(); i++)
                        {
                            if (pgo_edges[i].con_type == REGISTRATION)
                            {
                                int frame_idx_1 = cblock_submaps[pgo_edges[i].block1->id_in_strip]->last_frame_index;
                                int frame_idx_2 = cblock_submaps[pgo_edges[i].block2->id_in_strip]->last_frame_index;
                                pgo_edges[i].block1 = cblock_frames[frame_idx_1];
                                pgo_edges[i].block2 = cblock_frames[frame_idx_2];
                                framewise_pgo_edges.push_back(pgo_edges[i]);
                            }
                        }
                        bool inner_submap_optimization_status = false;
                        inner_submap_optimization_status = pgoptimizer.optimize_pose_graph_ceres(cblock_frames, framewise_pgo_edges, FLAGS_inner_submap_t_limit, FLAGS_inner_submap_r_limit, false); // set the limit better
                        // inner_submap_optimization_status = pgoptimizer.optimize_pose_graph_g2o(cblock_frames, framewise_pgo_edges, false);
                        for (int i = 0; i < cblock_frames.size(); i++)
                        {
                            if (!inner_submap_optimization_status)
                                cblock_frames[i]->pose_optimized = cblock_frames[i]->pose_init;
                        }
                        constraints().swap(framewise_pgo_edges);
                    }
                    else
                    {
                        ROS_INFO("Pose graph optimization");
                        // method 2: inner-submap pgo (post processing : refine pose within the submap and output final map, update pose of each frame in each submap using pgo)
                        pgoptimizer.set_robust_function(false);
                        pgoptimizer.set_equal_weight(FLAGS_equal_weight_on);
                        pgoptimizer.set_max_iter_num(FLAGS_max_iter_inner_submap);
                        pgoptimizer.set_diagonal_information_matrix(FLAGS_diagonal_information_matrix_on);
                        pgoptimizer.set_free_node(FLAGS_free_node_on);
                        pgoptimizer.set_problem_size(true); // small size problem --> dense schur
                        for (int i = 1; i < cblock_submaps.size(); i++)
                        {
                            cloudblock_Ptrs cblock_frames_in_submap;
                            constraints inner_submap_edges;
                            cblock_frames[cblock_submaps[i - 1]->last_frame_index]->pose_init = cblock_submaps[i - 1]->pose_lo;
                            cblock_frames[cblock_submaps[i - 1]->last_frame_index]->strip_id = cblock_submaps[i]->id_in_strip;
                            cblock_frames_in_submap.push_back(cblock_frames[cblock_submaps[i - 1]->last_frame_index]); // end frame of the last submap
                            cblock_frames_in_submap[0]->id_in_strip = 0;
                            cblock_frames_in_submap[0]->pose_fixed = true; // fix the first frame
                            int node_count = 1;

                            // last submap's end frame to this submap's end frame (index j) [last_frame_index store the index of the last frame of the submap]
                            for (int j = cblock_submaps[i - 1]->last_frame_index; j < cblock_submaps[i]->last_frame_index; j++)
                            {
                                Eigen::Matrix4d tran_mat_12 = poses_lo_adjacent[j].inverse();
                                cblock_frames[j + 1]->id_in_strip = node_count;
                                cblock_frames[j + 1]->strip_id = cblock_submaps[i]->id_in_strip;
                                cblock_frames[j + 1]->pose_init = cblock_frames[j]->pose_init * tran_mat_12;
                                cblock_frames_in_submap.push_back(cblock_frames[j + 1]);
                                node_count++;
                                confinder.add_adjacent_constraint(cblock_frames_in_submap, inner_submap_edges, tran_mat_12, node_count);
                            }
                            cblock_frames_in_submap[node_count - 1]->pose_fixed = true; // fix the last frame
                            cblock_frames_in_submap[node_count - 1]->pose_init = cblock_submaps[i]->pose_lo;
                            bool inner_submap_optimization_status = false;
                            if (!strcmp(FLAGS_pose_graph_optimization_method.c_str(), "g2o"))
                                inner_submap_optimization_status = pgoptimizer.optimize_pose_graph_g2o(cblock_frames_in_submap, inner_submap_edges, false);
                            else if (!strcmp(FLAGS_pose_graph_optimization_method.c_str(), "ceres"))
                                inner_submap_optimization_status = pgoptimizer.optimize_pose_graph_ceres(cblock_frames_in_submap, inner_submap_edges, FLAGS_inner_submap_t_limit, FLAGS_inner_submap_r_limit, false); // set the limit better
                            else if (!strcmp(FLAGS_pose_graph_optimization_method.c_str(), "gtsam"))
                                inner_submap_optimization_status = pgoptimizer.optimize_pose_graph_gtsam(cblock_frames_in_submap, inner_submap_edges);
                            else // default: ceres
                                inner_submap_optimization_status = pgoptimizer.optimize_pose_graph_ceres(cblock_frames_in_submap, inner_submap_edges, FLAGS_inner_submap_t_limit, FLAGS_inner_submap_r_limit, false);

                            for (int j = cblock_submaps[i - 1]->last_frame_index + 1; j <= cblock_submaps[i]->last_frame_index; j++)
                            {
                                if (inner_submap_optimization_status)
                                    cblock_frames[j]->pose_optimized = cblock_frames_in_submap[cblock_frames[j]->id_in_strip]->pose_optimized;
                                else
                                    cblock_frames[j]->pose_optimized = cblock_frames[j]->pose_init;
                            }
                            constraints().swap(inner_submap_edges);
                            cloudblock_Ptrs().swap(cblock_frames_in_submap);
                            ROS_INFO("Inner-submap pose refining done for submap [ %d ]", i);
                        }
                    }
                    std::chrono::steady_clock::time_point toc_inner_submap_refine = std::chrono::steady_clock::now();
                    std::chrono::duration<double> time_used_inner_submap_refine = std::chrono::duration_cast<std::chrono::duration<double>>(toc_inner_submap_refine - tic_inner_submap_refine);
                    ROS_INFO("Inner-submap pose refinement done in [ %f ] ms.", 1000 * time_used_inner_submap_refine.count());
                }

                if (loop_closure_detection_on)
                {
                    for (int i = 0; i < frame_num; i++) // update poses
                    {
                        poses_lo_lidar_cs[i] = cblock_frames[i]->pose_optimized;
                        if (i == 0) // write optimized pose out (overwrite)
                        {
                            dataio.write_lo_pose_overwrite(poses_lo_lidar_cs[0], output_lo_lidar_pose_file);
                            dataio.write_lo_pose_overwrite(poses_lo_body_cs[0], output_lo_body_pose_file); // required by KITTI
                        }
                        else
                        {
                            poses_lo_body_cs[i] = calib_mat * poses_lo_lidar_cs[i] * calib_mat.inverse(); // what we want actually is the Tb0_bi
                            dataio.write_lo_pose_append(poses_lo_lidar_cs[i], output_lo_lidar_pose_file);
                            dataio.write_lo_pose_append(poses_lo_body_cs[i], output_lo_body_pose_file); // required by KITTI
                        }
                    }

                    // free submaps' point cloud
                    for (int i = 0; i < cblock_submaps.size(); i++)
                        cblock_submaps[i]->free_all();

                    visualizePoseGraph(&pub_pgo_markers_, ros::Time::now(), pgo_edges);

                    constraints().swap(pgo_edges);
                }

                if (FLAGS_write_out_map_on || FLAGS_write_out_gt_map_on) // export map point cloud //TODO: be careful of the memory problem here!!! //FIX memory leakage while outputing map point cloud
                {
                    ROS_WARN("Begin to output the generated map");
                    pcTPtr pc_map_merged(new pcT), pc_map_gt_merged(new pcT);
                    for (int i = 0; i < frame_num; i++) // output merged map (with dist_filter, intrinsic correction and motion distortion
                    {
                        dataio.read_pc_cloud_block(cblock_frames[i]);
                        if (FLAGS_vertical_ang_calib_on) // intrinsic angle correction
                            cfilter.vertical_intrinsic_calibration(cblock_frames[i]->pc_raw, FLAGS_vertical_ang_correction_deg);
                        if (FLAGS_apply_dist_filter)
                            cfilter.dist_filter(cblock_frames[i]->pc_raw, FLAGS_min_dist_mapping, FLAGS_max_dist_mapping);
                        cfilter.random_downsample(cblock_frames[i]->pc_raw, FLAGS_map_downrate_output);
                        if (FLAGS_motion_compensation_method == 1) // calculate from time-stamp
                            cfilter.get_pts_timestamp_ratio_in_frame(cblock_frames[i]->pc_raw, true);
                        else if (FLAGS_motion_compensation_method == 2)                                      // calculate from azimuth
                            cfilter.get_pts_timestamp_ratio_in_frame(cblock_frames[i]->pc_raw, false, 90.0); // HESAI Lidar: 90.0 (y+ axis, clockwise), Velodyne Lidar: 180.0
                        if (FLAGS_write_out_map_on)
                        {
                            if (FLAGS_motion_compensation_method > 0 && i > 0)
                            {
                                Eigen::Matrix4d adjacent_tran = cblock_frames[i]->pose_optimized.inverse() * cblock_frames[i - 1]->pose_optimized;
                                cfilter.apply_motion_compensation(cblock_frames[i]->pc_raw, adjacent_tran);
                            }
                            pcl::transformPointCloud(*cblock_frames[i]->pc_raw, *cblock_frames[i]->pc_raw_w, cblock_frames[i]->pose_optimized);
                            if (FLAGS_write_map_each_frame)
                            {
                                std::string filename_without_path = cblock_frames[i]->filename.substr(cblock_frames[i]->filename.rfind('/') + 1);
                                std::string filename_without_extension = filename_without_path.substr(0, filename_without_path.rfind('.'));
                                std::string output_pc_file = output_pc_folder + "/" + filename_without_extension + ".pcd";
                                ROS_INFO("Save pointcloud to: %s",output_pc_file.c_str());
                                dataio.write_cloud_file(output_pc_file, cblock_frames[i]->pc_raw_w);
                            }
                            pc_map_merged->points.insert(pc_map_merged->points.end(), cblock_frames[i]->pc_raw_w->points.begin(), cblock_frames[i]->pc_raw_w->points.end());
                            cfilter.random_downsample(cblock_frames[i]->pc_raw_w, 2);
                            // mviewer.display_dense_map_realtime(cblock_frames[i], map_viewer, frame_num, display_time_ms);
                        }
                        if (FLAGS_write_out_gt_map_on)
                        {
                            if (FLAGS_motion_compensation_method > 0 && i > 0)
                            {
                                Eigen::Matrix4d adjacent_tran = cblock_frames[i]->pose_gt.inverse() * cblock_frames[i - 1]->pose_gt;
                                cfilter.apply_motion_compensation(cblock_frames[i]->pc_raw, adjacent_tran);
                            }
                            pcl::transformPointCloud(*cblock_frames[i]->pc_raw, *cblock_frames[i]->pc_sketch, cblock_frames[i]->pose_gt);
                            pc_map_gt_merged->points.insert(pc_map_gt_merged->points.end(), cblock_frames[i]->pc_sketch->points.begin(), cblock_frames[i]->pc_sketch->points.end());
                        }
                        cblock_frames[i]->free_all();
                    }

                    // TODO: add more map based operation //1.generate 2D geo-referenced image //2.intensity generalization
                    if (FLAGS_map_filter_on)
                        cfilter.sor_filter(pc_map_merged, 20, 2.0); // sor filtering before output
                    std::string output_merged_map_file = output_pc_folder + "/" + "merged_map.pcd";
                    ROS_INFO("output_merged_map_file: %s",output_merged_map_file.c_str());
                    if (FLAGS_write_out_map_on)
                        dataio.write_pcd_file(output_merged_map_file, pc_map_merged); // write merged map point cloud
                    std::string output_merged_gt_map_file = output_pc_folder + "/" + "merged_gt_map.pcd";
                    if (FLAGS_write_out_gt_map_on)
                        dataio.write_pcd_file(output_merged_gt_map_file, pc_map_gt_merged); // write merged map point cloud

                    if (FLAGS_write_out_map_on)
                    {
                        cv::Mat map_2d;
                        map_2d = cfilter.generate_2d_map(pc_map_merged, 1, FLAGS_screen_width, FLAGS_screen_height, 20, 1000, -FLT_MAX, FLT_MAX, true);
                        if (FLAGS_show_bev_image)
                        {
                            // mviewer.display_image(map_2d, "2D Map");
                        }
                        std::string output_merged_map_image = output_pc_folder + "/" + "merged_map_2d.png";
                        cv::imwrite(output_merged_map_image, map_2d);
                    }

                    pcT().swap(*pc_map_merged);
                    pcT().swap(*pc_map_gt_merged);
                }
                dataio.report_consuming_time(FLAGS_timing_report_file, timing_array);
                dataio.write_pose_point_cloud(lo_lidar_pose_point_cloud_file, poses_lo_lidar_cs);

                if (!gt_lidar_pose_point_cloud_file.empty())
                    dataio.write_pose_point_cloud(gt_lidar_pose_point_cloud_file, poses_gt_lidar_cs);

                OdomErrorCompute ec;
                std::vector<odom_errors_t> odom_errs, slam_errors;
                if (poses_gt_body_cs.size() > 0) // gt pose is availiable
                {
                    poses_gt_body_cs[0].setIdentity();
                    if (FLAGS_gt_in_lidar_frame)
                        odom_errs = ec.compute_error(poses_gt_lidar_cs, poses_lo_lidar_cs);
                    else
                        odom_errs = ec.compute_error(poses_gt_body_cs, poses_lo_body_cs);
                    ec.print_error(odom_errs);
                    if (loop_closure_detection_on)
                    {
                        if (FLAGS_gt_in_lidar_frame)
                            slam_errors = ec.compute_error(poses_gt_lidar_cs, poses_lo_lidar_cs, true); // longer segments, better for the evaluation of global localization performance
                        else
                            slam_errors = ec.compute_error(poses_gt_body_cs, poses_lo_body_cs, true); // longer segments
                        ec.print_error(slam_errors, true);
                    }
                }
            }

            rate.sleep();
            ros::spinOnce();
        }
    }

    submap_viewer_thread_.join();

    return 0;
}

// TODO LIST (potential future work)
// TODO: add ros support
// TODO: use deep learning based methods to do global registration (for loop closure)
// TODO: add the on-fly calibration of intrinsic parameters (intrinsic angle displacement)
// TODO: code refactoring from scratch

void LoadImagesKITTI(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<string> &vstrLidarFilenames, vector<double> &vTimestamps)
{
    std::ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while (!fTimes.eof())
    {
        string s;
        getline(fTimes, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixLidar = strPathToSequence + "/velodyne/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);
    vstrLidarFilenames.resize(nTimes);

    for (int i = 0; i < nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
        // vstrLidarFilenames[i] = strPrefixLidar + ss.str() + ".bin";
        vstrLidarFilenames[i] = strPrefixLidar + ss.str() + "." + FLAGS_pc_format;
    }
}

void LoadImagesTUM(const string &strFile, vector<string> &vstrImageFilenames,
                   vector<string> &vstrLidarFilenames, vector<double> &vTimestamps)
{
    std::ifstream f;
    f.open((strFile + "/associate_vl.txt").c_str());

    while (!f.eof())
    {
        string s;
        getline(f, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sL;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(strFile + "/" + sRGB);

            ss >> t;
            ss >> sL;
            vstrLidarFilenames.push_back(strFile + "/" + sL);
        }
    }
}

void publishLidarFeatures(lo::cloudblock_Ptr cur_block, bool show_raw_lidar)
{
    sensor_msgs::PointCloud2 ground_msg;
    pcl::toROSMsg(*cur_block->pc_ground, ground_msg);
    ground_msg.header.stamp = ros::Time::now();
    ground_msg.header.frame_id = "/base_link";
    pub_feature_ground_.publish(ground_msg);

    sensor_msgs::PointCloud2 facade_msg;
    pcl::toROSMsg(*cur_block->pc_facade, facade_msg);
    facade_msg.header.stamp = ros::Time::now();
    facade_msg.header.frame_id = "/base_link";
    pub_feature_facade_.publish(facade_msg);

    sensor_msgs::PointCloud2 roof_msg;
    pcl::toROSMsg(*cur_block->pc_roof, roof_msg);
    roof_msg.header.stamp = ros::Time::now();
    roof_msg.header.frame_id = "/base_link";
    pub_feature_roof_.publish(roof_msg);

    sensor_msgs::PointCloud2 pillar_msg;
    pcl::toROSMsg(*cur_block->pc_pillar, pillar_msg);
    pillar_msg.header.stamp = ros::Time::now();
    pillar_msg.header.frame_id = "/base_link";
    pub_feature_pillar_.publish(pillar_msg);

    sensor_msgs::PointCloud2 beam_msg;
    pcl::toROSMsg(*cur_block->pc_beam, beam_msg);
    beam_msg.header.stamp = ros::Time::now();
    beam_msg.header.frame_id = "/base_link";
    pub_feature_beam_.publish(beam_msg);

    sensor_msgs::PointCloud2 vertex_msg;
    pcl::toROSMsg(*cur_block->pc_vertex, vertex_msg);
    vertex_msg.header.stamp = ros::Time::now();
    vertex_msg.header.frame_id = "/base_link";
    pub_feature_vertex_.publish(vertex_msg);

    if (show_raw_lidar)
    {
        sensor_msgs::PointCloud2 raw_pc_msg;
        cur_block->pc_raw->height = 1;
        cur_block->pc_raw->width = cur_block->pc_raw->size();
        pcl::toROSMsg(*cur_block->pc_raw, raw_pc_msg);
        raw_pc_msg.header.stamp = ros::Time::now();
        raw_pc_msg.header.frame_id = "/base_link";
        pub_lidar_raw_.publish(raw_pc_msg);
    }
}

void PublishCurrentLocalMap_realtime(cloudblock_Ptr &local_map, bool show_with_color)
{
    if (show_with_color)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr feature_map(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<Point_T>::Ptr ground_pts(new pcl::PointCloud<Point_T>);
        pcl::PointCloud<Point_T>::Ptr facade_pts(new pcl::PointCloud<Point_T>);
        pcl::PointCloud<Point_T>::Ptr roof_pts(new pcl::PointCloud<Point_T>);
        pcl::PointCloud<Point_T>::Ptr pillar_pts(new pcl::PointCloud<Point_T>);
        pcl::PointCloud<Point_T>::Ptr beam_pts(new pcl::PointCloud<Point_T>);
        pcl::PointCloud<Point_T>::Ptr vertex_pts(new pcl::PointCloud<Point_T>);

        // transform to world coordinate system
        pcl::transformPointCloudWithNormals(*local_map->pc_ground, *ground_pts, local_map->pose_lo);
        pcl::transformPointCloudWithNormals(*local_map->pc_facade, *facade_pts, local_map->pose_lo);
        pcl::transformPointCloudWithNormals(*local_map->pc_roof, *roof_pts, local_map->pose_lo);
        pcl::transformPointCloudWithNormals(*local_map->pc_pillar, *pillar_pts, local_map->pose_lo);
        pcl::transformPointCloudWithNormals(*local_map->pc_beam, *beam_pts, local_map->pose_lo);
        pcl::transformPointCloudWithNormals(*local_map->pc_vertex, *vertex_pts, local_map->pose_lo);

        // change color map to semantic kitti format
        for (int i = 0; i < ground_pts->points.size(); ++i)
        {
            pcl::PointXYZRGB pt;
            pt.x = ground_pts->points[i].x;
            pt.y = ground_pts->points[i].y;
            pt.z = ground_pts->points[i].z;
            pt.r = 128;
            pt.g = 128;
            pt.b = 128;
            // Ground - Gray
            feature_map->points.push_back(pt);
        }
        for (int i = 0; i < pillar_pts->points.size(); ++i)
        {
            pcl::PointXYZRGB pt;
            pt.x = pillar_pts->points[i].x;
            pt.y = pillar_pts->points[i].y;
            pt.z = pillar_pts->points[i].z;
            pt.r = 0;
            pt.g = 255;
            pt.b = 0;
            // Pillar - Green
            feature_map->points.push_back(pt);
        }
        for (int i = 0; i < beam_pts->points.size(); ++i)
        {
            pcl::PointXYZRGB pt;
            pt.x = beam_pts->points[i].x;
            pt.y = beam_pts->points[i].y;
            pt.z = beam_pts->points[i].z;
            pt.r = 255;
            pt.g = 255;
            pt.b = 0;
            // Beam - Yellow
            feature_map->points.push_back(pt);
        }
        for (int i = 0; i < facade_pts->points.size(); ++i)
        {
            pcl::PointXYZRGB pt;
            pt.x = facade_pts->points[i].x;
            pt.y = facade_pts->points[i].y;
            pt.z = facade_pts->points[i].z;
            pt.r = 0;
            pt.g = 0;
            pt.b = 255;
            // Facade - Blue
            feature_map->points.push_back(pt);
        }
        for (int i = 0; i < roof_pts->points.size(); ++i)
        {
            pcl::PointXYZRGB pt;
            pt.x = roof_pts->points[i].x;
            pt.y = roof_pts->points[i].y;
            pt.z = roof_pts->points[i].z;
            pt.r = 255;
            pt.g = 0;
            pt.b = 0;
            // Roof - Red
            feature_map->points.push_back(pt);
        }

        for (int i = 0; i < vertex_pts->points.size(); ++i)
        {
            pcl::PointXYZRGB pt;
            pt.x = vertex_pts->points[i].x;
            pt.y = vertex_pts->points[i].y;
            pt.z = vertex_pts->points[i].z;
            pt.r = 255;
            pt.g = 0;
            pt.b = 255;
            // Vertex - Purple
            feature_map->points.push_back(pt);
        }

        sensor_msgs::PointCloud2 local_map_msg;
        pcl::toROSMsg(*feature_map, local_map_msg);
        local_map_msg.header.stamp = ros::Time::now();
        local_map_msg.header.frame_id = "/odom";
        pub_local_map_.publish(local_map_msg);

        // release memory
        pcl::PointCloud<pcl::PointXYZRGB>().swap(*feature_map);
        pcl::PointCloud<Point_T>().swap(*facade_pts);
        pcl::PointCloud<Point_T>().swap(*pillar_pts);
        pcl::PointCloud<Point_T>().swap(*ground_pts);
        pcl::PointCloud<Point_T>().swap(*vertex_pts);
        pcl::PointCloud<Point_T>().swap(*roof_pts);
        pcl::PointCloud<Point_T>().swap(*beam_pts);
    }
    else
    {
        pcl::PointCloud<Point_T>::Ptr feature_map(new pcl::PointCloud<Point_T>);
        pcl::PointCloud<Point_T>::Ptr feature_map_world(new pcl::PointCloud<Point_T>);
        local_map->merge_feature_points(feature_map, false);
        pcl::transformPointCloudWithNormals(*feature_map, *feature_map_world, local_map->pose_lo);

        sensor_msgs::PointCloud2 local_map_msg;
        feature_map_world->height = 1;
        feature_map_world->width = feature_map_world->size();
        pcl::toROSMsg(*feature_map_world, local_map_msg);
        local_map_msg.header.stamp = ros::Time::now();
        local_map_msg.header.frame_id = "/odom";
        pub_local_map_.publish(local_map_msg);
    }
}

void PublishTF(const Eigen::Matrix4d &Twl)
{
    Eigen::Vector3d translation = Twl.block<3, 1>(0, 3);
    Eigen::Matrix3d rotation = Twl.block<3, 3>(0, 0);

    // Eigen::Vector3f euler_angles = rotation.eulerAngles(1, 0, 2); //ZXY顺序,相机坐标系中
    Eigen::Quaterniond qf = Eigen::Quaterniond(rotation);

    static tf::TransformBroadcaster odometery_tf_publisher;
    geometry_msgs::TransformStamped odom_trans;
    odom_trans.header.stamp = ros::Time::now();
    odom_trans.header.frame_id = "/odom";
    odom_trans.child_frame_id = "/base_link";
    odom_trans.transform.translation.x = translation[0];
    odom_trans.transform.translation.y = translation[1];
    odom_trans.transform.translation.z = translation[2];
    // odom_trans.transform.rotation = tf::createQuaternionMsgFromRollPitchYaw(euler_angles[2], euler_angles[1], euler_angles[0]);
    odom_trans.transform.rotation.x = qf.x();
    odom_trans.transform.rotation.y = qf.y();
    odom_trans.transform.rotation.z = qf.z();
    odom_trans.transform.rotation.w = qf.w();
    odometery_tf_publisher.sendTransform(odom_trans);

    // 发布激光雷达的轨迹
    geometry_msgs::PoseStamped lidar_odometry_posestamped;
    lidar_odometry_posestamped.pose.position.x = translation(0);
    lidar_odometry_posestamped.pose.position.y = translation(1);
    lidar_odometry_posestamped.pose.position.z = translation(2);
    lidar_odometry_posestamped.pose.orientation.x = qf.x();
    lidar_odometry_posestamped.pose.orientation.y = qf.y();
    lidar_odometry_posestamped.pose.orientation.z = qf.z();
    lidar_odometry_posestamped.pose.orientation.w = qf.w();
    lidar_odometry_posestamped.header.stamp = ros::Time::now();
    lidar_odometry_posestamped.header.frame_id = "odom";
    lidar_odom_path_.header.frame_id = "odom";
    lidar_odom_path_.header.stamp = ros::Time::now();
    lidar_odom_path_.poses.push_back(lidar_odometry_posestamped);

    pub_lidar_odom_path_.publish(lidar_odom_path_);

    // 发布floam的优化结果
    geometry_msgs::PoseStamped floam_lidar_odometry_posestamped;
    floam_lidar_odometry_posestamped.header.frame_id = "odom";
    floam_lidar_odometry_posestamped.header.stamp = ros::Time::now();
    Eigen::Quaterniond qf_floam = Eigen::Quaterniond(odomEstimationNode.odom.rotation());
    Eigen::Vector3d t_floam = odomEstimationNode.odom.translation();
    floam_lidar_odometry_posestamped.pose.orientation.x = qf_floam.x();
    floam_lidar_odometry_posestamped.pose.orientation.y = qf_floam.y();
    floam_lidar_odometry_posestamped.pose.orientation.z = qf_floam.z();
    floam_lidar_odometry_posestamped.pose.orientation.w = qf_floam.w();
    floam_lidar_odometry_posestamped.pose.position.x = t_floam.x();
    floam_lidar_odometry_posestamped.pose.position.y = t_floam.y();
    floam_lidar_odometry_posestamped.pose.position.z = t_floam.z();
    floam_lidar_odom_path_.header.frame_id = "odom";
    floam_lidar_odom_path_.header.stamp = ros::Time::now();
    floam_lidar_odom_path_.poses.push_back(floam_lidar_odometry_posestamped);
    pub_floam_lidar_odom_path_.publish(floam_lidar_odom_path_);
}

void publishImage(const ros::Publisher &image_publisher_, cv::Mat &img)
{
    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = "image";
    if (img.channels() == 1)
    {
        const sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(header, "mono8", img).toImageMsg();
        image_publisher_.publish(image_msg);
    }
    else if (img.channels() == 3)
    {
        const sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(header, "bgr8", img).toImageMsg();
        image_publisher_.publish(image_msg);
    }
}

// 显示当前位姿图信息
void visualizePoseGraph(ros::Publisher *pub_m, ros::Time timestamp, const constraints &cons)
{
    int edge_count = cons.size();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr nodes(new pcl::PointCloud<pcl::PointXYZRGB>());

    visualization_msgs::MarkerArray markerArray;

    // 1. add lines to viewer
    // 1.1 adjacent edges
    visualization_msgs::Marker markerAdjacentEdge;
    markerAdjacentEdge.header.frame_id = "odom";
    markerAdjacentEdge.header.stamp = timestamp;
    markerAdjacentEdge.action = visualization_msgs::Marker::ADD;
    markerAdjacentEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerAdjacentEdge.ns = "adjacent_edges";
    markerAdjacentEdge.id = 1;
    markerAdjacentEdge.pose.orientation.w = 1;
    markerAdjacentEdge.scale.x = 0.1;
    markerAdjacentEdge.color.r = 0.0;
    markerAdjacentEdge.color.g = 1.0;
    markerAdjacentEdge.color.b = 1.0;
    markerAdjacentEdge.color.a = 1;

    // 1.2 loop edges
    visualization_msgs::Marker markerLoopEdge;
    markerLoopEdge.header.frame_id = "odom";
    markerLoopEdge.header.stamp = timestamp;
    markerLoopEdge.action = visualization_msgs::Marker::ADD;
    markerLoopEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerLoopEdge.ns = "loop_edges";
    markerLoopEdge.id = 2;
    markerLoopEdge.pose.orientation.w = 1;
    markerLoopEdge.scale.x = 0.1;
    markerLoopEdge.color.r = 1.0;
    markerLoopEdge.color.g = 0.0;
    markerLoopEdge.color.b = 1.0;
    markerLoopEdge.color.a = 1;

    // 1.3 history edges
    visualization_msgs::Marker markerHistoryEdge;
    markerHistoryEdge.header.frame_id = "odom";
    markerHistoryEdge.header.stamp = timestamp;
    markerHistoryEdge.action = visualization_msgs::Marker::ADD;
    markerHistoryEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerHistoryEdge.ns = "history_edges";
    markerHistoryEdge.id = 3;
    markerHistoryEdge.pose.orientation.w = 1;
    markerHistoryEdge.scale.x = 0.1;
    markerHistoryEdge.color.r = 1.0;
    markerHistoryEdge.color.g = 1.0;
    markerHistoryEdge.color.b = 0.0;
    markerHistoryEdge.color.a = 1;

    // 1.4 none/wrong edges
    visualization_msgs::Marker markerNoneEdge;
    markerNoneEdge.header.frame_id = "odom";
    markerNoneEdge.header.stamp = timestamp;
    markerNoneEdge.action = visualization_msgs::Marker::ADD;
    markerNoneEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerNoneEdge.ns = "none_wrong_edges";
    markerNoneEdge.id = 4;
    markerNoneEdge.pose.orientation.w = 1;
    markerNoneEdge.scale.x = 0.1;
    markerNoneEdge.color.r = 1.0;
    markerNoneEdge.color.g = 0.0;
    markerNoneEdge.color.b = 0.0;
    markerNoneEdge.color.a = 1;

    // 2. submap node list
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = "odom";
    markerNode.header.stamp = timestamp;
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "submap_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.8;
    markerNode.scale.y = 0.8;
    markerNode.scale.z = 0.8;

    for (int i = 0; i < edge_count; i++)
    {
        float r = 0.0, g = 0.0, b = 0.0;

        // target block
        geometry_msgs::Point ptc1;
        ptc1.x = cons[i].block1->pose_lo(0, 3);
        ptc1.y = cons[i].block1->pose_lo(1, 3);
        ptc1.z = cons[i].block1->pose_lo(2, 3);

        std_msgs::ColorRGBA ptc1_color;
        random_color(r, g, b, 255);
        ptc1_color.r = r / 255.0;
        ptc1_color.g = g / 255.0;
        ptc1_color.b = b / 255.0;
        ptc1_color.a = 1.0;

        pcl::PointXYZRGB ptc1_rgb;
        ptc1_rgb.x = ptc1.x;
        ptc1_rgb.y = ptc1.y;
        ptc1_rgb.z = ptc1.z;
        ptc1_rgb.r = r;
        ptc1_rgb.g = g;
        ptc1_rgb.b = b;
        nodes->points.push_back(ptc1_rgb);

        // source block
        geometry_msgs::Point ptc2;
        ptc2.x = cons[i].block2->pose_lo(0, 3);
        ptc2.y = cons[i].block2->pose_lo(1, 3);
        ptc2.z = cons[i].block2->pose_lo(2, 3);

        std_msgs::ColorRGBA ptc2_color;
        random_color(r, g, b, 255);
        ptc2_color.r = r / 255.0;
        ptc2_color.g = g / 255.0;
        ptc2_color.b = b / 255.0;
        ptc2_color.a = 1.0;

        pcl::PointXYZRGB ptc2_rgb;
        ptc2_rgb.x = ptc2.x;
        ptc2_rgb.y = ptc2.y;
        ptc2_rgb.z = ptc2.z;
        ptc2_rgb.r = r;
        ptc2_rgb.g = g;
        ptc2_rgb.b = b;
        nodes->points.push_back(ptc2_rgb);

        markerNode.points.push_back(ptc1);
        markerNode.colors.push_back(ptc1_color);
        markerNode.points.push_back(ptc2);
        markerNode.colors.push_back(ptc2_color);

        switch (cons[i].con_type)
        {
        case ADJACENT: // 相邻边
            markerAdjacentEdge.points.push_back(ptc1);
            markerAdjacentEdge.points.push_back(ptc2);
            break;
        case REGISTRATION: //
            markerLoopEdge.points.push_back(ptc1);
            markerLoopEdge.points.push_back(ptc2);
            break;
        case HISTORY: // edges in the long-term memory (history)
            markerHistoryEdge.points.push_back(ptc1);
            markerHistoryEdge.points.push_back(ptc2);
            break;
        case NONE: // failed/wrong edge
            markerNoneEdge.points.push_back(ptc1);
            markerNoneEdge.points.push_back(ptc2);
            break;
        default:
            break;
        }
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerAdjacentEdge);
    markerArray.markers.push_back(markerLoopEdge);
    markerArray.markers.push_back(markerHistoryEdge);
    markerArray.markers.push_back(markerNoneEdge);
    pub_m->publish(markerArray);
}

void visualizeSubmapBlocks(const cloudblock_Ptrs &cblock_submaps)
{
    while (ros::ok())
    {
        // ROS_INFO("publish submaps, size: %d", cblock_submaps.size());
        pcl::PointCloud<Point_T>::Ptr feature_map_world(new pcl::PointCloud<Point_T>);

        jsk_recognition_msgs::BoundingBoxArray submap_boundingbox_list_;
        submap_boundingbox_list_.header.stamp = ros::Time::now();
        submap_boundingbox_list_.header.frame_id = "odom";

        for (int i = 0; i < cblock_submaps.size(); i++)
        {
            cloudblock_Ptr cblock_submap = cblock_submaps.at(i);

            pcl::PointCloud<Point_T>::Ptr feature_map(new pcl::PointCloud<Point_T>);
            pcl::PointCloud<Point_T>::Ptr feature_map_tmp(new pcl::PointCloud<Point_T>);

            cblock_submap->merge_feature_points(feature_map, false);
            pcl::transformPointCloudWithNormals(*feature_map, *feature_map_tmp, cblock_submap->pose_lo);
            *feature_map_world += *feature_map_tmp;

            // 计算子地图的包围盒
            bounds_t submap_bound = cblock_submap->bound;
            jsk_recognition_msgs::BoundingBox submap_boundingbox;
            submap_boundingbox.header.frame_id = "odom";
            submap_boundingbox.header.stamp = ros::Time::now();
            submap_boundingbox.label = i;
            submap_boundingbox.pose.position.x = (submap_bound.min_x + submap_bound.max_x) * 0.5;
            submap_boundingbox.pose.position.y = (submap_bound.min_y + submap_bound.max_y) * 0.5;
            submap_boundingbox.pose.position.z = (submap_bound.min_z + submap_bound.max_z) * 0.5;
            submap_boundingbox.pose.orientation.x = 0.0;
            submap_boundingbox.pose.orientation.y = 0.0;
            submap_boundingbox.pose.orientation.z = 0.0;
            submap_boundingbox.pose.orientation.w = 1.0;
            submap_boundingbox.dimensions.x = submap_bound.max_x - submap_bound.min_x;
            submap_boundingbox.dimensions.y = submap_bound.max_y - submap_bound.min_y;
            submap_boundingbox.dimensions.z = 0.1;
            submap_boundingbox_list_.boxes.push_back(submap_boundingbox);

            feature_map.reset(new pcT());
            feature_map_tmp.reset(new pcT());
        }

        sensor_msgs::PointCloud2 submap_lists_msg;
        feature_map_world->height = 1;
        feature_map_world->width = feature_map_world->size();
        pcl::toROSMsg(*feature_map_world, submap_lists_msg);
        submap_lists_msg.header.stamp = ros::Time::now();
        submap_lists_msg.header.frame_id = "/odom";
        pub_submap_lists_.publish(submap_lists_msg);

        pub_submap_boundingboxs_.publish(submap_boundingbox_list_);

        ros::Duration(1.0).sleep();
    }
    // bound
}

void gray2color(const cv::Mat &imgGray, cv::Mat &imgColor, const float v_min, const float v_max)
{
    imgColor = cv::Mat(imgGray.rows, imgGray.cols, CV_8UC3);
    float dv = v_max - v_min;
    for (int i = 0; i < imgGray.rows; i++)
    {
        for (int j = 0; j < imgGray.cols; j++)
        {
            float v = imgGray.at<float>(i, j);
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

            imgColor.at<cv::Vec3b>(i, j)[0] = 255 * r;
            imgColor.at<cv::Vec3b>(i, j)[1] = 255 * g;
            imgColor.at<cv::Vec3b>(i, j)[2] = 255 * b;
        }
    }
}

void display_image(const cv::Mat &image, cv::Mat &ri_temp_cm, int color_scale)
{
    if (color_scale == 0) // gray image
        ri_temp_cm = image;
    else if (color_scale == 1) // COLORMAP_JET
        cv::applyColorMap(image, ri_temp_cm, cv::COLORMAP_JET);
    else if (color_scale == 2) // COLORMAP_AUTUMN
        cv::applyColorMap(image, ri_temp_cm, cv::COLORMAP_AUTUMN);
    else if (color_scale == 3) // COLORMAP_HOT
        cv::applyColorMap(image, ri_temp_cm, cv::COLORMAP_HOT);
    else // default: gray
        ri_temp_cm = image;
    // cv::imshow(image_viewer_name, ri_temp_cm);
    // cv::waitKey(time_delay_ms);
}