//
// This file is a General Definition and Tool for Point Cloud Processing based on PCL.
// Dependent 3rd Libs: PCL (>1.7)
// By Yue Pan et al.
//
#ifndef _INCLUDE_UTILITY_HPP_
#define _INCLUDE_UTILITY_HPP_

//PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/point_representation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/io/pcd_io.h>

// OpenCV
#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>

//Eigen
#include <Eigen/Core>

#include <sophus/se3.h>

#include <vector>
#include <list>
#include <chrono>
#include <limits>
#include <time.h>

#include <glog/logging.h>

#include "lidar_sparse_align/WeightFunction.h"
#include "ThirdParty/DBoW3/src/DBoW3.h"

#define max_(a, b) (((a) > (b)) ? (a) : (b))
#define min_(a, b) (((a) < (b)) ? (a) : (b))

using namespace std;

//TypeDef

//Select from these two (with/without intensity)
//typedef pcl::PointNormal Point_T;
typedef pcl::PointXYZINormal Point_T; //mind that 'curvature' here is used as ring number for spining scanner
//typedef ccn::PointXYZINTRL Point_T; //TODO : with time stamp, label and ring number property (a better way is to use the customed point type without template class)
//typedef pcl::PointSurfel Point_T;

/**
//pcl::PointXYZINormal member variables
//x,y,z: 3d coordinates
//intensity: relective intensity
//normal_x, normal_y, normal_z: used as normal or direction vector
//curvature: used as timestamp or descriptor 
//data[3]: used as neighborhood curvature
//normal[3]: used as the height above ground
**/

typedef pcl::PointCloud<Point_T>::Ptr pcTPtr;
typedef pcl::PointCloud<Point_T> pcT;

typedef pcl::search::KdTree<Point_T>::Ptr pcTreePtr;
typedef pcl::search::KdTree<Point_T> pcTree;

typedef pcl::PointCloud<pcl::PointXYZI>::Ptr pcXYZIPtr;
typedef pcl::PointCloud<pcl::PointXYZI> pcXYZI;

typedef pcl::PointCloud<pcl::PointXYZ>::Ptr pcXYZPtr;
typedef pcl::PointCloud<pcl::PointXYZ> pcXYZ;

typedef pcl::PointCloud<pcl::PointXY>::Ptr pcXYPtr;
typedef pcl::PointCloud<pcl::PointXY> pcXY;

typedef pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcXYZRGBPtr;
typedef pcl::PointCloud<pcl::PointXYZRGB> pcXYZRGB;

typedef pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pcXYZRGBAPtr;
typedef pcl::PointCloud<pcl::PointXYZRGBA> pcXYZRGBA;

typedef pcl::PointCloud<pcl::PointNormal>::Ptr pcXYZNPtr;
typedef pcl::PointCloud<pcl::PointNormal> pcXYZN;

typedef pcl::PointCloud<pcl::PointXYZINormal>::Ptr pcXYZINPtr;
typedef pcl::PointCloud<pcl::PointXYZINormal> pcXYZIN;

typedef pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhPtr;
typedef pcl::PointCloud<pcl::FPFHSignature33> fpfh;

// typedef Eigen::Matrix<double, 6, 1> Vector6d;
// typedef Eigen::Matrix<double, 6, 6> Matrix6d;

typedef std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> Matrix4ds;

typedef Eigen::Matrix<double, 4, 1> Vector4d;
typedef Eigen::Matrix<float, 6, 1> Vector6f;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<float, 2, 6> Matrix2x6;
typedef Eigen::Matrix<double, 4, 4> Matrix4d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

namespace lo
{

	struct centerpoint_t
	{
		double x;
		double y;
		double z;
		centerpoint_t(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
	};

	//regular bounding box whose edges are parallel to x,y,z axises
	struct bounds_t
	{
		double min_x;
		double min_y;
		double min_z;
		double max_x;
		double max_y;
		double max_z;
		int type;

		bounds_t()
		{
			min_x = min_y = min_z = max_x = max_y = max_z = 0.0;
		}
		void inf_x()
		{
			min_x = -DBL_MAX;
			max_x = DBL_MAX;
		}
		void inf_y()
		{
			min_y = -DBL_MAX;
			max_y = DBL_MAX;
		}
		void inf_z()
		{
			min_z = -DBL_MAX;
			max_z = DBL_MAX;
		}
		void inf_xyz()
		{
			inf_x();
			inf_y();
			inf_z();
		}
	};

	// ORB特征提取参数
	typedef struct _orb_params_t orb_params_t;
	struct _orb_params_t
	{
		/* data */
		int n_features;
		float scale_factor;
		int nLevels;
		int iniThFast;
		int minThFast;
	};

	//the point cloud's collecting template (node's type)
	enum DataType
	{
		ALS,
		TLS,
		MLS,
		BPLS,
		RGBD,
		SLAM
	};

	//the edge (constraint)'s type
	enum ConstraintType
	{
		REGISTRATION,
		ADJACENT,
		HISTORY,
		SMOOTH,
		NONE
	};

	typedef struct SphericalPoint
	{
		float az; // azimuth
		float el; // elevation
		float r;  // radius
	} SphericalPoint;

	inline std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors)
	{
		std::vector<cv::Mat> vDesc;
		vDesc.reserve(Descriptors.rows);
		for (int j = 0; j < Descriptors.rows; j++)
			vDesc.push_back(Descriptors.row(j));

		return vDesc;
	}

	inline Sophus::SE3 toSophusSE3(const Eigen::Matrix4d &trans)
	{
		return Sophus::SE3(trans.topLeftCorner(3, 3), trans.topRightCorner(3, 1));
	}

	// 已知两个link之间的外参，以及其中一个link的运动，计算另一个link的运动
	inline bool relative_pose_transform(const Eigen::Matrix4d &trans_soure, const Eigen::Matrix4d &Extrinsics, Eigen::Matrix4d &trans_target)
	{
		trans_target = Extrinsics * trans_soure * Extrinsics.inverse();
	}

	inline float rad2deg(double radians)
	{
		return radians * 180.0 / M_PI;
	}

	inline float deg2rad(double degrees)
	{
		return degrees * M_PI / 180.0;
	}

	inline void random_color(float &r, float &g, float &b, float range_max) //range_max example 1,255...
	{
		r = range_max * (rand() / (1.0 + RAND_MAX));
		g = range_max * (rand() / (1.0 + RAND_MAX));
		b = range_max * (rand() / (1.0 + RAND_MAX));
	}

	typedef struct _tracker_t tracker_t;
	struct _tracker_t
	{
		int levels;
		int min_level;
		int max_level;
		int max_iteration;

		bool use_weight_scale = true;
		string scale_estimator;
		string weight_function;

		ScaleEstimatorType scale_estimator_type;
		WeightFunctionType weight_function_type;

		void set_scale_estimator_type()
		{
			if (!scale_estimator.compare("None"))
				use_weight_scale = false;
			if (!scale_estimator.compare("TDistributionScale"))
				scale_estimator_type = ScaleEstimatorType::TDistributionScale;

			// cerr << "ScaleType : " << static_cast<int>(scale_estimator_type);
		}

		void set_weight_function_type()
		{
			if (!weight_function.compare("TDistributionWeight"))
				weight_function_type = WeightFunctionType::TDistributionWeight;
		}
	};

	struct SubmapFeatureExtractor
	{
		std::pair<float, float> fov_;				 /* e.g., [vfov = 50 (upper 25, lower 25), hfov = 360] */
		std::pair<int, int> rimg_size_;				 // 距离图的尺寸
		float range_img_resolution_[2] = {0.4, 0.8}; //remove_resolution_list: [2.5, 1.5]

		SubmapFeatureExtractor()
		{
			fov_.first = 50;   //垂直视场角
			fov_.second = 360; //水平视场角
			rimg_size_.first = fov_.first / range_img_resolution_[0];
			rimg_size_.second = fov_.second / range_img_resolution_[0];
		}

		SubmapFeatureExtractor(const std::pair<float, float> &fov, const std::pair<int, int> &rimg_size, float *range_img_resolution)
			: fov_(fov), rimg_size_(rimg_size)
		{
			range_img_resolution_[0] = range_img_resolution[0];
			range_img_resolution_[1] = range_img_resolution[1];
		}

		SphericalPoint cart2sph(const Point_T &_cp)
		{ // _cp means cartesian point

			SphericalPoint sph_point{
				std::atan2(_cp.y, _cp.x),
				std::atan2(_cp.z, std::sqrt(_cp.x * _cp.x + _cp.y * _cp.y)),
				std::sqrt(_cp.x * _cp.x + _cp.y * _cp.y + _cp.z * _cp.z)};
			return sph_point;
		}

		void map2RangeImg(const pcl::PointCloud<Point_T>::Ptr &_scan,
						  std::pair<cv::Mat, cv::Mat> &rangeImg_pair)
		{
			map2RangeImg(_scan, fov_, rimg_size_, rangeImg_pair);
		}

		void map2RangeImg(const pcl::PointCloud<Point_T>::Ptr &_scan,
						  const std::pair<float, float> _fov, /* e.g., [vfov = 50 (upper 25, lower 25), hfov = 360] */
						  const std::pair<int, int> _rimg_size,
						  std::pair<cv::Mat, cv::Mat> &rangeImg_pair)
		{
			const float kVFOV = _fov.first;
			const float kHFOV = _fov.second;

			const int kNumRimgRow = _rimg_size.first;
			const int kNumRimgCol = _rimg_size.second;

			// @ range image initizliation
			const float kFlagNoPOINT = 10000.0;														   // no point constant, 10000 has no meaning, but must be larger than the maximum scan range (e.g., 200 meters)
			cv::Mat rimg = cv::Mat(kNumRimgRow, kNumRimgCol, CV_32FC1, cv::Scalar::all(kFlagNoPOINT)); // float matrix, save range value
			cv::Mat rimg_ptidx = cv::Mat(kNumRimgRow, kNumRimgCol, CV_32SC1, cv::Scalar::all(0));	   // int matrix, save point (of global map) index

			// @ points to range img
			int num_points = _scan->points.size();

#pragma omp parallel for num_threads(4)
			for (int pt_idx = 0; pt_idx < num_points; ++pt_idx)
			{
				Point_T this_point = _scan->points[pt_idx];
				SphericalPoint sph_point = cart2sph(this_point);

				// @ note about vfov: e.g., (+ V_FOV/2) to adjust [-15, 15] to [0, 30]
				// @ min and max is just for the easier (naive) boundary checks.
				int lower_bound_row_idx{0};
				int lower_bound_col_idx{0};
				int upper_bound_row_idx{kNumRimgRow - 1};
				int upper_bound_col_idx{kNumRimgCol - 1};
				int pixel_idx_row = int(std::min(std::max(std::round(kNumRimgRow * (1 - (rad2deg(sph_point.el) + (kVFOV / float(2.0))) / (kVFOV - float(0.0)))), float(lower_bound_row_idx)), float(upper_bound_row_idx)));
				int pixel_idx_col = int(std::min(std::max(std::round(kNumRimgCol * ((rad2deg(sph_point.az) + (kHFOV / float(2.0))) / (kHFOV - float(0.0)))), float(lower_bound_col_idx)), float(upper_bound_col_idx)));

				float curr_range = sph_point.r;

				// @ Theoretically, this if-block would have race condition (i.e., this is a critical section),
				// @ But, the resulting range image is acceptable (watching via Rviz),
				// @      so I just naively applied omp pragma for this whole for-block (2020.10.28)
				// @ Reason: because this for loop is splited by the omp, points in a single splited for range do not race among them,
				// @         also, a point A and B lied in different for-segments do not tend to correspond to the same pixel,
				// #               so we can assume practically there are few race conditions.
				// @ P.S. some explicit mutexing directive makes the code even slower ref: https://stackoverflow.com/questions/2396430/how-to-use-lock-in-openmp
				if (curr_range < rimg.at<float>(pixel_idx_row, pixel_idx_col))
				{
					rimg.at<float>(pixel_idx_row, pixel_idx_col) = curr_range;
					rimg_ptidx.at<int>(pixel_idx_row, pixel_idx_col) = pt_idx;
				}
			}

			rangeImg_pair.first = rimg.clone();
			rangeImg_pair.second = rimg_ptidx.clone();
		}
	};

	//pose as the format of translation vector and unit quaterniond, used in pose graph optimization
	struct pose_qua_t //copyright: Jingwei Li
	{
		Eigen::Vector3d trans;
		Eigen::Quaterniond quat;

		pose_qua_t()
		{
			trans << 0, 0, 0;
			quat = Eigen::Quaterniond(Eigen::Matrix3d::Identity());
			quat.normalize();
		}
		pose_qua_t(const pose_qua_t &pose)
		{
			this->copyFrom(pose);
		}
		pose_qua_t(Eigen::Quaterniond quat,
				   Eigen::Vector3d trans) : trans(trans), quat(quat)
		{
			quat.normalize();
		}
		pose_qua_t operator*(const pose_qua_t &pose) const
		{
			return pose_qua_t(this->quat.normalized() * pose.quat.normalized(), this->quat.normalized() * pose.trans + this->trans);
		}
		bool operator==(const pose_qua_t &pose) const
		{
			if (this->quat.x() == pose.quat.x() &&
				this->quat.y() == pose.quat.y() &&
				this->quat.z() == pose.quat.z() &&
				this->quat.w() == pose.quat.w() &&
				this->trans == pose.trans)
			{
				return true;
			}
			else
			{
				return false;
			}
		}
		Eigen::Matrix4d GetMatrix() const
		{
			//CHECK(quat.norm() == 1) << "NO EQUAL";
			Eigen::Matrix4d transformation_matrix;
			transformation_matrix.block<3, 3>(0, 0) = quat.normalized().toRotationMatrix(); //You need to gurantee the quat is normalized
			transformation_matrix.block<3, 1>(0, 3) = trans;
			transformation_matrix.block<1, 4>(3, 0) << 0, 0, 0, 1;
			return transformation_matrix;
		}
		void SetPose(Eigen::Matrix4d transformation)
		{
			quat = Eigen::Quaterniond(transformation.block<3, 3>(0, 0)).normalized();
			trans << transformation(0, 3), transformation(1, 3), transformation(2, 3);
		}
		void copyFrom(const pose_qua_t &pose)
		{
			trans = pose.trans;
			//  trans << pose.trans[0], pose.trans[1], pose.trans[2];
			quat = Eigen::Quaterniond(pose.quat);
		}
		// inverse and return
		pose_qua_t inverse()
		{
			Eigen::Matrix4d transformation_matrix = GetMatrix();
			SetPose(transformation_matrix.inverse());
			return *this;
		}

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};

	typedef std::vector<pose_qua_t, Eigen::aligned_allocator<pose_qua_t>> pose_quas;

	//Basic processing unit(node)
	struct cloudblock_t
	{
		//Strip (transaction) should be the container of the cloudblock while cloudblock can act as either a frame or submap (local map)
		int unique_id;		  //Unique ID
		int strip_id;		  //Strip ID
		int id_in_strip;	  //ID in the strip
		int last_frame_index; //last_frame_id is the frame index (not unique_id) of the last frame of the submap
		//ID means the number may not be continous and begining from 0 (like 3, 7, 11, ...),
		//but index should begin from 0 and its step (interval) should be 1 (like 0, 1, 2, 3, ...)

		DataType data_type; //Datatype

		bounds_t bound;				  //Bounding Box in geo-coordinate system
		centerpoint_t cp;			  //Center Point in geo-coordinate system
		centerpoint_t station;		  //Station position in geo-coordinate system
		Eigen::Matrix4d station_pose; //Station pose in geo-coordinate system

		bounds_t local_bound;				//Bounding Box in local coordinate system
		centerpoint_t local_cp;				//Center Point in local coordinate system
		centerpoint_t local_station;		//Station position in local coordinate system
		Eigen::Matrix4d local_station_pose; //Station pose in local coordinate system

		bool station_position_available; //If the approximate position of the station is provided
		bool station_pose_available;	 //If the approximate pose of the station is provided
		bool is_single_scanline;		 //If the scanner is a single scanline sensor, determining if adjacent cloud blocks in a strip would have overlap

		bool pose_fixed = false;  //the pose is fixed or not
		bool pose_stable = false; //the pose is stable or not after the optimization

		//poses
		Eigen::Matrix4d pose_lo;		//used for lidar odometry
		Eigen::Matrix4d pose_gt;		//used for lidar odometry (ground turth)
		Eigen::Matrix4d pose_optimized; //optimized pose
		Eigen::Matrix4d pose_init;		//used for the init guess for pgo

		Matrix6d information_matrix_to_next;

		std::string filename;			//full path of the original point cloud file
		std::string filenmae_processed; //full path of the processed point cloud file

		// struct SubmapFeatureExtractor submapFeatureExtractor;
		std::pair<cv::Mat, cv::Mat> rangeImg_pair;

		// direct tracking
		std::vector<cv::Mat> mvImgPyramid;
		pcTPtr mpMagLidarPointCloud;

		//image features
		int NP;
		std::vector<cv::KeyPoint> mvKeys;
		std::vector<cv::KeyPoint> mvKeysUn;
		cv::Mat mDescriptors;

		// Scale pyramid info.
		int mnScaleLevels;
		float mfScaleFactor;
		float mfLogScaleFactor;
		vector<float> mvScaleFactors;
		vector<float> mvInvScaleFactors;
		vector<float> mvLevelSigma2;
		vector<float> mvInvLevelSigma2;

		// Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
		std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

		// Bag of Words Vector structures.
		DBoW3::BowVector mBowVec;
		DBoW3::FeatureVector mFeatVec;

		// 用于保存局部submap的原始数据
		pcTPtr pc_local_map_raw_w_down;
		pcTPtr pc_local_map_raw_w;

		//Raw point cloud
		pcTPtr pc_raw;

		//Downsampled point cloud
		pcTPtr pc_down;
		pcTPtr pc_sketch; //very sparse point cloud

		pcTPtr pc_raw_w; //in world coordinate system (for lidar odometry)

		//unground point cloud
		pcTPtr pc_unground;

		// All kinds of geometric feature points (loam)
		pcTPtr cornerPointsSharp;
		pcTPtr cornerPointsLessSharp;
		pcTPtr surfPointsFlat;
		pcTPtr surfPointsLessFlat;

		// All kinds of geometric feature points (in target scan)
		pcTPtr pc_ground;
		pcTPtr pc_facade;
		pcTPtr pc_roof;
		pcTPtr pc_pillar;
		pcTPtr pc_beam;
		pcTPtr pc_vertex;

		//downsampled feature points (in source scan)
		pcTPtr pc_ground_down;
		pcTPtr pc_facade_down;
		pcTPtr pc_roof_down;
		pcTPtr pc_pillar_down;
		pcTPtr pc_beam_down;

		//Kdtree of the feature points (denser ones)
		pcTreePtr tree_ground;
		pcTreePtr tree_pillar;
		pcTreePtr tree_beam;
		pcTreePtr tree_facade;
		pcTreePtr tree_roof;
		pcTreePtr tree_vertex;

		//actually, it's better to save the indices of feature_points_down instead of saving another feature point cloud
		int down_feature_point_num;
		int feature_point_num;

		cloudblock_t()
		{
			init();
			//default value
			station_position_available = false;
			station_pose_available = false;
			is_single_scanline = true;
			pose_lo.setIdentity();
			pose_gt.setIdentity();
			pose_optimized.setIdentity();
			pose_init.setIdentity();
			information_matrix_to_next.setIdentity();
		}

		cloudblock_t(const cloudblock_t &in_block, bool clone_feature = false, bool clone_raw = false)
		{
			init();
			clone_metadata(in_block);

			// clone_visualrelated_metadata(in_block);

			if (clone_feature)
			{
				//clone point cloud (instead of pointer)
				*pc_ground = *(in_block.pc_ground);
				*pc_pillar = *(in_block.pc_pillar);
				*pc_facade = *(in_block.pc_facade);
				*pc_beam = *(in_block.pc_beam);
				*pc_roof = *(in_block.pc_roof);
				*pc_vertex = *(in_block.pc_vertex);
				// keypoint_bsc = in_block.keypoint_bsc;
			}
			if (clone_raw)
				*pc_raw = *(in_block.pc_raw);
		}

		void init()
		{
			pc_raw = boost::make_shared<pcT>();
			pc_down = boost::make_shared<pcT>();
			pc_raw_w = boost::make_shared<pcT>();
			pc_sketch = boost::make_shared<pcT>();
			pc_unground = boost::make_shared<pcT>();

			pc_ground = boost::make_shared<pcT>();
			pc_facade = boost::make_shared<pcT>();
			pc_roof = boost::make_shared<pcT>();
			pc_pillar = boost::make_shared<pcT>();
			pc_beam = boost::make_shared<pcT>();
			pc_vertex = boost::make_shared<pcT>();

			pc_ground_down = boost::make_shared<pcT>();
			pc_facade_down = boost::make_shared<pcT>();
			pc_roof_down = boost::make_shared<pcT>();
			pc_pillar_down = boost::make_shared<pcT>();
			pc_beam_down = boost::make_shared<pcT>();

			init_tree();

			//doubleVectorSBF().swap(keypoint_bsc);

			down_feature_point_num = 0;
			feature_point_num = 0;

			// kinggreat24
			mpMagLidarPointCloud = boost::make_shared<pcT>();
			NP = 0;
		}

		void init_tree()
		{
			tree_ground = boost::make_shared<pcTree>();
			tree_facade = boost::make_shared<pcTree>();
			tree_pillar = boost::make_shared<pcTree>();
			tree_beam = boost::make_shared<pcTree>();
			tree_roof = boost::make_shared<pcTree>();
			tree_vertex = boost::make_shared<pcTree>();
		}

		void free_raw_cloud()
		{
			pc_raw.reset(new pcT());
			pc_down.reset(new pcT());
			pc_unground.reset(new pcT());
		}

		void free_tree()
		{
			tree_ground.reset(new pcTree());
			tree_facade.reset(new pcTree());
			tree_pillar.reset(new pcTree());
			tree_beam.reset(new pcTree());
			tree_roof.reset(new pcTree());
			tree_vertex.reset(new pcTree());
		}

		void free_visual_data()
		{
			mvImgPyramid.clear();
			mpMagLidarPointCloud.reset(new pcT());

			//image features
			NP = 0;
			mvKeys.clear();
			mvKeysUn.clear();
			mDescriptors = cv::Mat();

			// Bag of Words Vector structures.
			mBowVec.clear();
			mFeatVec.clear();
		}

		void free_all()
		{
			free_raw_cloud();
			free_tree();
			pc_ground.reset(new pcT());
			pc_facade.reset(new pcT());
			pc_pillar.reset(new pcT());
			pc_beam.reset(new pcT());
			pc_roof.reset(new pcT());
			pc_vertex.reset(new pcT());
			pc_ground_down.reset(new pcT());
			pc_facade_down.reset(new pcT());
			pc_pillar_down.reset(new pcT());
			pc_beam_down.reset(new pcT());
			pc_roof_down.reset(new pcT());
			pc_sketch.reset(new pcT());
			pc_raw_w.reset(new pcT());
			//doubleVectorSBF().swap(keypoint_bsc);
		}

		void clone_metadata(const cloudblock_t &in_cblock)
		{
			feature_point_num = in_cblock.feature_point_num;
			bound = in_cblock.bound;
			local_bound = in_cblock.local_bound;
			local_cp = in_cblock.local_cp;
			pose_lo = in_cblock.pose_lo;
			pose_gt = in_cblock.pose_gt;
			pose_init = in_cblock.pose_init;
			pose_optimized = in_cblock.pose_optimized;
			unique_id = in_cblock.unique_id;
			id_in_strip = in_cblock.id_in_strip;
			filename = in_cblock.filename;
		}

		void clone_visualrelated_metadata(const cloudblock_t &in_cblock)
		{
			mvImgPyramid = in_cblock.mvImgPyramid;
			*mpMagLidarPointCloud = *in_cblock.mpMagLidarPointCloud;

			//image features
			NP = in_cblock.NP;
			mvKeys = in_cblock.mvKeys;
			mvKeysUn = in_cblock.mvKeysUn;
			mDescriptors = in_cblock.mDescriptors;

			// Scale pyramid info.
			mnScaleLevels = in_cblock.mnScaleLevels;
			mfScaleFactor = in_cblock.mfScaleFactor;
			mfLogScaleFactor = in_cblock.mfLogScaleFactor;
			mvScaleFactors = in_cblock.mvScaleFactors;
			mvInvScaleFactors = in_cblock.mvInvScaleFactors;
			mvLevelSigma2 = in_cblock.mvLevelSigma2;
			mvInvLevelSigma2 = in_cblock.mvInvLevelSigma2;

			// Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
			// mGrid = in_cblock.mGrid;
			for (int i = 0; i < FRAME_GRID_COLS; i++)
				for (int j = 0; j < FRAME_GRID_ROWS; j++)
					mGrid[i][j] = in_cblock.mGrid[i][j];

			// Bag of Words Vector structures.
			mBowVec = in_cblock.mBowVec;
			mFeatVec = in_cblock.mFeatVec;
		}

		void append_feature(const cloudblock_t &in_cblock, bool append_down, std::string used_feature_type)
		{
			//pc_raw->points.insert(pc_raw->points.end(), in_cblock.pc_raw->points.begin(), in_cblock.pc_raw->points.end());
			if (!append_down)
			{
				if (used_feature_type[0] == '1')
					pc_ground->points.insert(pc_ground->points.end(), in_cblock.pc_ground->points.begin(), in_cblock.pc_ground->points.end());
				if (used_feature_type[1] == '1')
					pc_pillar->points.insert(pc_pillar->points.end(), in_cblock.pc_pillar->points.begin(), in_cblock.pc_pillar->points.end());
				if (used_feature_type[2] == '1')
					pc_facade->points.insert(pc_facade->points.end(), in_cblock.pc_facade->points.begin(), in_cblock.pc_facade->points.end());
				if (used_feature_type[3] == '1')
					pc_beam->points.insert(pc_beam->points.end(), in_cblock.pc_beam->points.begin(), in_cblock.pc_beam->points.end());
				if (used_feature_type[4] == '1')
					pc_roof->points.insert(pc_roof->points.end(), in_cblock.pc_roof->points.begin(), in_cblock.pc_roof->points.end());
				pc_vertex->points.insert(pc_vertex->points.end(), in_cblock.pc_vertex->points.begin(), in_cblock.pc_vertex->points.end());
			}
			else
			{
				if (used_feature_type[0] == '1')
					pc_ground->points.insert(pc_ground->points.end(), in_cblock.pc_ground_down->points.begin(), in_cblock.pc_ground_down->points.end());
				if (used_feature_type[1] == '1')
					pc_pillar->points.insert(pc_pillar->points.end(), in_cblock.pc_pillar_down->points.begin(), in_cblock.pc_pillar_down->points.end());
				if (used_feature_type[2] == '1')
					pc_facade->points.insert(pc_facade->points.end(), in_cblock.pc_facade_down->points.begin(), in_cblock.pc_facade_down->points.end());
				if (used_feature_type[3] == '1')
					pc_beam->points.insert(pc_beam->points.end(), in_cblock.pc_beam_down->points.begin(), in_cblock.pc_beam_down->points.end());
				if (used_feature_type[4] == '1')
					pc_roof->points.insert(pc_roof->points.end(), in_cblock.pc_roof_down->points.begin(), in_cblock.pc_roof_down->points.end());
				pc_vertex->points.insert(pc_vertex->points.end(), in_cblock.pc_vertex->points.begin(), in_cblock.pc_vertex->points.end());
			}
		}

		void merge_feature_points(pcTPtr &pc_out, bool merge_down, bool with_out_ground = false)
		{
			if (!merge_down)
			{
				if (!with_out_ground)
					pc_out->points.insert(pc_out->points.end(), pc_ground->points.begin(), pc_ground->points.end());
				pc_out->points.insert(pc_out->points.end(), pc_facade->points.begin(), pc_facade->points.end());
				pc_out->points.insert(pc_out->points.end(), pc_pillar->points.begin(), pc_pillar->points.end());
				pc_out->points.insert(pc_out->points.end(), pc_beam->points.begin(), pc_beam->points.end());
				pc_out->points.insert(pc_out->points.end(), pc_roof->points.begin(), pc_roof->points.end());
				pc_out->points.insert(pc_out->points.end(), pc_vertex->points.begin(), pc_vertex->points.end());
			}
			else
			{
				if (!with_out_ground)
					pc_out->points.insert(pc_out->points.end(), pc_ground_down->points.begin(), pc_ground_down->points.end());
				pc_out->points.insert(pc_out->points.end(), pc_facade_down->points.begin(), pc_facade_down->points.end());
				pc_out->points.insert(pc_out->points.end(), pc_pillar_down->points.begin(), pc_pillar_down->points.end());
				pc_out->points.insert(pc_out->points.end(), pc_beam_down->points.begin(), pc_beam_down->points.end());
				pc_out->points.insert(pc_out->points.end(), pc_roof_down->points.begin(), pc_roof_down->points.end());
				pc_out->points.insert(pc_out->points.end(), pc_vertex->points.begin(), pc_vertex->points.end());
			}
		}

		void transform_feature(const Eigen::Matrix4d &trans_mat, bool transform_down = true, bool transform_undown = true)
		{
			if (transform_undown)
			{
				pcl::transformPointCloudWithNormals(*pc_ground, *pc_ground, trans_mat);
				pcl::transformPointCloudWithNormals(*pc_pillar, *pc_pillar, trans_mat);
				pcl::transformPointCloudWithNormals(*pc_beam, *pc_beam, trans_mat);
				pcl::transformPointCloudWithNormals(*pc_facade, *pc_facade, trans_mat);
				pcl::transformPointCloudWithNormals(*pc_roof, *pc_roof, trans_mat);
				pcl::transformPointCloudWithNormals(*pc_vertex, *pc_vertex, trans_mat);
			}
			if (transform_down)
			{
				pcl::transformPointCloudWithNormals(*pc_ground_down, *pc_ground_down, trans_mat);
				pcl::transformPointCloudWithNormals(*pc_pillar_down, *pc_pillar_down, trans_mat);
				pcl::transformPointCloudWithNormals(*pc_beam_down, *pc_beam_down, trans_mat);
				pcl::transformPointCloudWithNormals(*pc_facade_down, *pc_facade_down, trans_mat);
				pcl::transformPointCloudWithNormals(*pc_roof_down, *pc_roof_down, trans_mat);
			}
		}

		void transform_raw_pointcloud(pcTPtr &pc_out, const Eigen::Matrix4d &trans_mat)
		{
			pcl::transformPointCloudWithNormals(*pc_raw, *pc_out, trans_mat);
		}

		void clone_cloud(pcTPtr &pc_out, bool get_pc_done)
		{
			if (get_pc_done)
				pc_out->points.insert(pc_out->points.end(), pc_down->points.begin(), pc_down->points.end());
			else
				pc_out->points.insert(pc_out->points.end(), pc_raw->points.begin(), pc_raw->points.end());
		}

		void clone_feature(pcTPtr &pc_ground_out,
						   pcTPtr &pc_pillar_out,
						   pcTPtr &pc_beam_out,
						   pcTPtr &pc_facade_out,
						   pcTPtr &pc_roof_out,
						   pcTPtr &pc_vertex_out, bool get_feature_down)

		{
			if (get_feature_down)
			{
				*pc_ground_out = *pc_ground_down;
				*pc_pillar_out = *pc_pillar_down;
				*pc_beam_out = *pc_beam_down;
				*pc_facade_out = *pc_facade_down;
				*pc_roof_out = *pc_roof_down;
				*pc_vertex_out = *pc_vertex;
			}
			else
			{
				*pc_ground_out = *pc_ground;
				*pc_pillar_out = *pc_pillar;
				*pc_beam_out = *pc_beam;
				*pc_facade_out = *pc_facade;
				*pc_roof_out = *pc_roof;
				*pc_vertex_out = *pc_vertex;
			}
		}

		void save_feature_clouds(const std::string &path_dir)
		{
			char file_name[128] = {0};
			if (pc_ground->size() > 0)
			{
				sprintf(file_name, "%s/ground.pcd", path_dir.c_str());
				pc_ground->height = 1;
				pc_ground->width = pc_ground->size();
				pcl::io::savePCDFileASCII(file_name, *pc_ground); //将点云保存到PCD文件中
			}

			if (pc_facade->size() > 0)
			{
				sprintf(file_name, "%s/facade.pcd", path_dir.c_str());
				pc_facade->height = 1;
				pc_facade->width = pc_facade->size();
				pcl::io::savePCDFileASCII(file_name, *pc_facade); //将点云保存到PCD文件中
			}

			if (pc_roof->size() > 0)
			{
				sprintf(file_name, "%s/roof.pcd", path_dir.c_str());
				pc_roof->height = 1;
				pc_roof->width = pc_roof->size();
				pcl::io::savePCDFileASCII(file_name, *pc_roof); //将点云保存到PCD文件中
			}

			if (pc_pillar->size() > 0)
			{
				sprintf(file_name, "%s/pillar.pcd", path_dir.c_str());
				pc_pillar->height = 1;
				pc_pillar->width = pc_pillar->size();
				pcl::io::savePCDFileASCII(file_name, *pc_pillar); //将点云保存到PCD文件中
			}

			if (pc_beam->size() > 0)
			{
				sprintf(file_name, "%s/beam.pcd", path_dir.c_str());
				pc_beam->height = 1;
				pc_beam->width = pc_beam->size();
				pcl::io::savePCDFileASCII(file_name, *pc_beam); //将点云保存到PCD文件中
			}

			if (pc_vertex->size() > 0)
			{
				sprintf(file_name, "%s/vertex.pcd", path_dir.c_str());
				pc_vertex->height = 1;
				pc_vertex->width = pc_vertex->size();
				pcl::io::savePCDFileASCII(file_name, *pc_vertex); //将点云保存到PCD文件中
			}
		}

		//激光直接法
		pcl::PointCloud<Point_T> &pointcloud()
		{
			return (*mpMagLidarPointCloud);
		}

		//空间点对姿态的导数
		inline static void jacobian_xyz2uv(const Eigen::Vector3f &xyz_in_f, Matrix2x6 &J)
		{
			const float x = xyz_in_f[0];
			const float y = xyz_in_f[1];
			const float z_inv = 1. / xyz_in_f[2];
			const float z_inv_2 = z_inv * z_inv;

			J(0, 0) = -z_inv;				// -1/z
			J(0, 1) = 0.0;					// 0
			J(0, 2) = x * z_inv_2;			// x/z^2
			J(0, 3) = y * J(0, 2);			// x*y/z^2
			J(0, 4) = -(1.0 + x * J(0, 2)); // -(1.0 + x^2/z^2)
			J(0, 5) = y * z_inv;			// y/z

			J(1, 0) = 0.0;				 // 0
			J(1, 1) = -z_inv;			 // -1/z
			J(1, 2) = y * z_inv_2;		 // y/z^2
			J(1, 3) = 1.0 + y * J(1, 2); // 1.0 + y^2/z^2
			J(1, 4) = -J(0, 3);			 // -x*y/z^2
			J(1, 5) = -x * z_inv;		 // x/z
		}

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};

	typedef std::vector<cloudblock_t, Eigen::aligned_allocator<cloudblock_t>> strip;
	typedef std::vector<strip> strips;
	typedef boost::shared_ptr<cloudblock_t> cloudblock_Ptr;
	typedef std::vector<cloudblock_Ptr> cloudblock_Ptrs;

	//the edge of pose(factor) graph
	struct constraint_t
	{
		int unique_id;				   //Unique ID
		cloudblock_Ptr block1, block2; //Two block  //Target: block1,  Source: block2
		ConstraintType con_type;	   //ConstraintType
		Eigen::Matrix4d Trans1_2;	   //transformation from 2 to 1 (in global shifted map coordinate system)
		Matrix6d information_matrix;
		float overlapping_ratio; //overlapping ratio (not bbx IOU) of two cloud blocks
		float confidence;
		float sigma;			  //standard deviation of the edge
		bool cov_updated = false; //has the information_matrix already updated

		constraint_t()
		{
			block1 = cloudblock_Ptr(new cloudblock_t);
			block2 = cloudblock_Ptr(new cloudblock_t);
			Trans1_2.setIdentity();
			information_matrix.setIdentity();
			sigma = FLT_MAX;
			cov_updated = false;
		}

		void free_cloud()
		{
			block1->free_all();
			block2->free_all();
		}

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};

	typedef std::vector<constraint_t, Eigen::aligned_allocator<constraint_t>> constraints;

	//2d ring map(image) for the point cloud collected by mechanical(spining) multiple-scanline LiDAR
	struct ring_map_t
	{
		int width;
		int height;

		float f_up;
		float f_down;

		std::vector<std::vector<unsigned int>> ring_array;

		void init_hdl64() //check it later
		{
			width = 1800;
			height = 64;
			f_up = 15;
			f_down = 15;
			init_ring_map();
		}

		void init_pandar64()
		{
			width = 1800;
			height = 64;
			f_up = 15;
			f_down = 25;
			init_ring_map();
		}

		void init_ring_map()
		{
			ring_array.resize(height);
			for (int i = 0; i < height; i++)
				ring_array[i].resize(width);
		}

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};

	//structs for gnssins navigation and localization
	struct gnss_info_t //OXTS format
	{
		double lon;			 // longitude (deg)
		double lat;			 // latitude (deg)
		double alt;			 // altitude (m)
		double roll;		 // roll angle (rad),    0 = level, positive = left side up,      range: -pi   .. +pi
		double pitch;		 // pitch angle (rad),   0 = level, positive = front down,        range: -pi/2 .. +pi/2
		double yaw;			 // heading (rad),       0 = east,  positive = counter clockwise, range: -pi   .. +pi
		double vn;			 // velocity towards north (m/s)
		double ve;			 // velocity towards east (m/s)
		double vf;			 // forward velocity, i.e. parallel to earth-surface (m/s)
		double vl;			 // leftward velocity, i.e. parallel to earth-surface (m/s)
		double vd;			 // upward velocity, i.e. perpendicular to earth-surface (m/s)
		double ax;			 // acceleration in x, i.e. in direction of vehicle front (m/s^2)
		double ay;			 // acceleration in y, i.e. in direction of vehicle left (m/s^2)
		double az;			 // acceleration in z, i.e. in direction of vehicle top (m/s^2)
		double af;			 // forward acceleration (m/s^2)
		double al;			 // leftward acceleration (m/s^2)
		double ad;			 // upward acceleration (m/s^2)
		double wx;			 // angular rate around x (rad/s)
		double wy;			 // angular rate around y (rad/s)
		double wz;			 // angular rate around z (rad/s)
		double wf;			 // angular rate around forward axis (rad/s)
		double wl;			 // angular rate around leftward axis (rad/s)
		double wd;			 // angular rate around upward axis (rad/s)
		double pos_accuracy; // velocity accuracy (north/east in m)
		double vel_accuracy; // velocity accuracy (north/east in m/s)
		char navstat;		 // navigation status (see navstat_to_string)
		int numsats;		 // number of satellites tracked by primary GPS receiver
	};

	struct imu_info_t
	{
		//accelerometr
		double ax;
		double ay;
		double az;
		//bias
		double bax;
		double bay;
		double baz;
		//gyro
		double wx; //unit:degree
		double wy; //unit:degree
		double wz; //unit:degree
				   //bias
		double bwx;
		double bwy;
		double bwz;
		timeval time_stamp;

		bool set_imu_info(double ax_i, double ay_i, double az_i,
						  double wx_i, double wy_i, double wz_i,
						  timeval time_stamp_i)
		{
			ax = ax_i;
			ay = ay_i;
			az = az_i;
			wx = wx_i;
			wy = wy_i;
			wz = wz_i;
			time_stamp = time_stamp_i;
		}
	};

	struct imu_infos_t
	{
		static const int frequncy = 10; // one imu_infos_t includes 10 imu_info_t
		std::vector<imu_info_t> imu_infos;
		timeval time_stamp; // time_stamp is the sensor time of the last ImuInfo
	};

	//parameter lists for lidar odometry (not used)
	struct constraint_param_t
	{
		std::string constraint_output_file;
		std::string registration_output_file;

		int find_constraint_knn;
		double find_constraint_overlap_ratio;
		int cloud_block_capacity;

		double vf_downsample_resolution_als;
		double vf_downsample_resolution_tls;
		double vf_downsample_resolution_mls;
		double vf_downsample_resolution_bpls;

		double gf_grid_resolution;
		double gf_max_grid_height_diff;
		double gf_neighbor_height_diff;
		int ground_downsample_rate;
		int nonground_downsample_rate;

		bool normal_downsampling_on;
		int normal_down_ratio;

		double pca_neigh_r;
		int pca_neigh_k;
		double pca_linearity_thre;
		double pca_planarity_thre;
		double pca_stablity_thre;

		double reg_corr_dis_thre;
		int reg_max_iteration_num;
		double converge_tran;
		double converge_rot_d;
	};

	//parameters and options for pose graph optimization (pgo)
	struct pgo_param_t
	{
		std::string block_tran_file = "";

		//TODO: maybe change the optimization methods here
		std::string trust_region_strategy = "levenberg_marquardt";
		//std::string trust_region_strategy = "dogleg";

		std::string linear_solver = "dense_schur";
		//std::string linear_solver = "sparse_schur";

		std::string sparse_linear_algebra_library = "suite_sparse";
		std::string dense_linear_algebra_library = "eigen";

		std::string robust_kernel_strategy = "huber";

		std::string ordering;  //marginalization
		bool robustify = true; //robust kernel function
		bool use_equal_weight = false;
		bool only_limit_translation = false;
		bool use_diagonal_information_matrix = false;
		bool free_all_nodes = false;
		bool is_small_size_problem = true;
		float robust_delta = 1.0;
		bool verbose = false; //show the detailed logs or not

		int num_threads = omp_get_max_threads(); //default = max
		//int num_threads = 4;
		int num_iterations = 50;

		float quat_tran_ratio = 1000.0; //the information (weight) ratio of rotation quaternion elements over translation elements

		//thresholds
		float wrong_edge_translation_thre = 5.0;
		float wrong_edge_rotation_thre = 20.0;
		float wrong_edge_ratio_thre = 0.1;
		//if wrong_edge_ratio_thre *100 % edges are judged as wrong edge, then we think this optimization is wrong

		//covariance (information matrix) updationg ratio
		float first_time_updating_ratio = 1.0;
		float life_long_updating_ratio = 1.0;

		//error estimation of each cloud block (for assigning the adjacent edge's information matrix [vcm^-1])
		float tx_std = 0.01;
		float ty_std = 0.01;
		float tz_std = 0.01;
		float roll_std = 0.05;
		float pitch_std = 0.05;
		float yaw_std = 0.05;
	};

	//basic common functions of point cloud
	template <typename PointT>
	class CloudUtility
	{
	public:
		//Get Center of a Point Cloud
		void get_cloud_cpt(const typename pcl::PointCloud<PointT>::Ptr &cloud, centerpoint_t &cp)
		{
			double cx = 0, cy = 0, cz = 0;
			int point_num = cloud->points.size();

			for (int i = 0; i < point_num; i++)
			{
				cx += cloud->points[i].x / point_num;
				cy += cloud->points[i].y / point_num;
				cz += cloud->points[i].z / point_num;
			}
			cp.x = cx;
			cp.y = cy;
			cp.z = cz;
		}

		//Get Bound of a Point Cloud
		void get_cloud_bbx(const typename pcl::PointCloud<PointT>::Ptr &cloud, bounds_t &bound)
		{
			double min_x = DBL_MAX;
			double min_y = DBL_MAX;
			double min_z = DBL_MAX;
			double max_x = -DBL_MAX;
			double max_y = -DBL_MAX;
			double max_z = -DBL_MAX;

			for (int i = 0; i < cloud->points.size(); i++)
			{
				if (min_x > cloud->points[i].x)
					min_x = cloud->points[i].x;
				if (min_y > cloud->points[i].y)
					min_y = cloud->points[i].y;
				if (min_z > cloud->points[i].z)
					min_z = cloud->points[i].z;
				if (max_x < cloud->points[i].x)
					max_x = cloud->points[i].x;
				if (max_y < cloud->points[i].y)
					max_y = cloud->points[i].y;
				if (max_z < cloud->points[i].z)
					max_z = cloud->points[i].z;
			}
			bound.min_x = min_x;
			bound.max_x = max_x;
			bound.min_y = min_y;
			bound.max_y = max_y;
			bound.min_z = min_z;
			bound.max_z = max_z;
		}

		//Get Bound and Center of a Point Cloud
		void get_cloud_bbx_cpt(const typename pcl::PointCloud<PointT>::Ptr &cloud, bounds_t &bound, centerpoint_t &cp)
		{
			get_cloud_bbx(cloud, bound);
			cp.x = 0.5 * (bound.min_x + bound.max_x);
			cp.y = 0.5 * (bound.min_y + bound.max_y);
			cp.z = 0.5 * (bound.min_z + bound.max_z);
		}

		void get_intersection_bbx(bounds_t &bbx_1, bounds_t &bbx_2, bounds_t &bbx_intersection, float bbx_boundary_pad = 2.0)
		{
			bbx_intersection.min_x = max_(bbx_1.min_x, bbx_2.min_x) - bbx_boundary_pad;
			bbx_intersection.min_y = max_(bbx_1.min_y, bbx_2.min_y) - bbx_boundary_pad;
			bbx_intersection.min_z = max_(bbx_1.min_z, bbx_2.min_z) - bbx_boundary_pad;
			bbx_intersection.max_x = min_(bbx_1.max_x, bbx_2.max_x) + bbx_boundary_pad;
			bbx_intersection.max_y = min_(bbx_1.max_y, bbx_2.max_y) + bbx_boundary_pad;
			bbx_intersection.max_z = min_(bbx_1.max_z, bbx_2.max_z) + bbx_boundary_pad;
		}

		void merge_bbx(std::vector<bounds_t> &bbxs, bounds_t &bbx_merged)
		{
			bbx_merged.min_x = DBL_MAX;
			bbx_merged.min_y = DBL_MAX;
			bbx_merged.min_z = DBL_MAX;
			bbx_merged.max_x = -DBL_MAX;
			bbx_merged.max_y = -DBL_MAX;
			bbx_merged.max_z = -DBL_MAX;

			for (int i = 0; i < bbxs.size(); i++)
			{
				bbx_merged.min_x = min_(bbx_merged.min_x, bbxs[i].min_x);
				bbx_merged.min_y = min_(bbx_merged.min_y, bbxs[i].min_y);
				bbx_merged.min_z = min_(bbx_merged.min_z, bbxs[i].min_z);
				bbx_merged.max_x = max_(bbx_merged.max_x, bbxs[i].max_x);
				bbx_merged.max_y = max_(bbx_merged.max_y, bbxs[i].max_y);
				bbx_merged.max_z = max_(bbx_merged.max_z, bbxs[i].max_z);
			}
		}

		//Get Bound of Subsets of a Point Cloud
		void get_sub_bbx(typename pcl::PointCloud<PointT>::Ptr &cloud, vector<int> &index, bounds_t &bound)
		{
			typename pcl::PointCloud<PointT>::Ptr temp_cloud(new pcl::PointCloud<PointT>);
			for (int i = 0; i < index.size(); i++)
			{
				temp_cloud->push_back(cloud->points[index[i]]);
			}
			get_cloud_bbx(temp_cloud, bound);
		}

		bool get_ring_map(const typename pcl::PointCloud<PointT>::Ptr &cloud_in, ring_map_t &ring_map) //check it later
		{
			for (int i = 0; i < cloud_in->points.size(); i++)
			{
				PointT pt = cloud_in->points[i];
				float dist = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
				float hor_ang = std::atan2(pt.y, pt.x);
				float ver_ang = std::asin(pt.z / dist);

				float u = 0.5 * (1 - hor_ang / M_PI) * ring_map.width;
				float v = (1 - (ver_ang + ring_map.f_up) / (ring_map.f_up + ring_map.f_down)) * ring_map.height;

				ring_map.ring_array[(int)v][(int)u] = i; //save the indice of the point
			}
		}

	protected:
	private:
	};
} // namespace lo

//TODO: reproduce the code with better data structure

#endif //_INCLUDE_UTILITY_HPP_