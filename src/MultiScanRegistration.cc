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

#include "MultiScanRegistration.h"
#include "math_utils.h"

//使用原始的激光雷达坐标
#define USE_RAW_LIDAR_COORDINATE 1

namespace ORB_SLAM2
{

MultiScanMapper::MultiScanMapper(const float &lowerBound,
								 const float &upperBound,
								 const uint16_t &nScanRings)
	: _lowerBound(lowerBound),
	  _upperBound(upperBound),
	  _nScanRings(nScanRings),
	  _factor((nScanRings - 1) / (upperBound - lowerBound))
{
}

void MultiScanMapper::set(const float &lowerBound,
						  const float &upperBound,
						  const uint16_t &nScanRings)
{
	_lowerBound = lowerBound;
	_upperBound = upperBound;
	_nScanRings = nScanRings;
	_factor = (nScanRings - 1) / (upperBound - lowerBound);
}

int MultiScanMapper::getRingForAngle(const float &angle)
{
	return int(((angle * 180 / M_PI) - _lowerBound) * _factor + 0.5);
}

MultiScanRegistration::MultiScanRegistration(const MultiScanMapper &scanMapper)
	: _scanMapper(scanMapper), _systemDelay(0){};

bool MultiScanRegistration::setupRegistrationParams(const std::string &lidarName, const RegistrationParams &config, const float vAngleMin, const float vAngleMax, const int nScanRings)
{
	_registrationParams = config;

	if (lidarName == "VLP-16")
	{
		_scanMapper = MultiScanMapper::Velodyne_VLP_16();
	}
	else if (lidarName == "HDL-32")
	{
		_scanMapper = MultiScanMapper::Velodyne_HDL_32();
	}
	else if (lidarName == "HDL-64E")
	{
		_scanMapper = MultiScanMapper::Velodyne_HDL_64E();
	}
	else if (lidarName == "CFANS-16")
	{
		_scanMapper = MultiScanMapper::CFANS_16();
	}
	else if (lidarName == "CFANS-32")
	{
		_scanMapper = MultiScanMapper::CFANS_32();
	}
	else if (lidarName == "USER-DEFINE")
	{
		if (vAngleMin >= vAngleMax)
		{
			std::cerr << "Invalid vertical range (min >= max)" << std::endl;
			return false;
		}
		else if (nScanRings < 2)
		{
			std::cerr << "Invalid number of scan rings (n < 2)" << std::endl;
			return false;
		}
		_scanMapper.set(vAngleMin, vAngleMax, nScanRings);
	}
	else
	{
		char error_message[128] = {0};
		sprintf(error_message, "Invalid lidar parameter: %s (only \"VLP-16\", \"HDL-32\" and \"HDL-64E\" or \"USER-DEFINE\" are supported)", lidarName.c_str());
		std::cerr << error_message << std::endl;
		return false;
	}

	return true;
}

void MultiScanRegistration::publishResult(pcl::PointCloud<Point_T> &cornerPointsSharpOut,
										  pcl::PointCloud<Point_T> &cornerPointsLessSharpOut,
										  pcl::PointCloud<Point_T> &surfacePointsFlatOut,
										  pcl::PointCloud<Point_T> &surfacePointsLessFlatOut)
{
	cornerPointsSharpOut     = cornerPointsSharp();
	cornerPointsLessSharpOut = cornerPointsLessSharp();
	surfacePointsFlatOut     = surfacePointsFlat();
	surfacePointsLessFlatOut = surfacePointsLessFlat();
}

void MultiScanRegistration::handleIMUMessage(const Imu &imuIn)
{
}

void MultiScanRegistration::handleCloudMessage(const pcl::PointCloud<Point_T> &laserCloudIn,
											   const Time &scanTime,
											   pcl::PointCloud<Point_T> &cornerPointsSharpOut,
											   pcl::PointCloud<Point_T> &cornerPointsLessSharpOut,
											   pcl::PointCloud<Point_T> &surfacePointsFlatOut,
											   pcl::PointCloud<Point_T> &surfacePointsLessFlatOut)
{
	if (_systemDelay > 0)
	{
		--_systemDelay;
		return;
	}
	process(laserCloudIn, scanTime);
	publishResult(cornerPointsSharpOut, cornerPointsLessSharpOut, surfacePointsFlatOut, surfacePointsLessFlatOut);
}

void MultiScanRegistration::process(const pcl::PointCloud<Point_T> &laserCloudIn, const Time &scanTime)
{
	size_t cloudSize = laserCloudIn.size();

	// determine scan start and end orientations
	float startOri = -std::atan2(laserCloudIn[0].y, laserCloudIn[0].x);
	float endOri = -std::atan2(laserCloudIn[cloudSize - 1].y,
							   laserCloudIn[cloudSize - 1].x) +
				   2 * float(M_PI);
	if (endOri - startOri > 3 * M_PI)
	{
		endOri -= 2 * M_PI;
	}
	else if (endOri - startOri < M_PI)
	{
		endOri += 2 * M_PI;
	}

	bool halfPassed = false;
	Point_T point;
	_laserCloudScans.resize(_scanMapper.getNumberOfScanRings());
	// clear all scanline points
	std::for_each(_laserCloudScans.begin(), _laserCloudScans.end(), [](auto &&v) { v.clear(); });

	// extract valid points from input cloud
	for (int i = 0; i < cloudSize; i++)
	{

		//此处激光雷达已经被转到相机坐标系中(KITTI相机坐标系)
#if (USE_RAW_LIDAR_COORDINATE)
		point.x = laserCloudIn[i].x;
		point.y = laserCloudIn[i].y;
		point.z = laserCloudIn[i].z;
#else
		point.x = laserCloudIn[i].y;
		point.y = laserCloudIn[i].z;
		point.z = laserCloudIn[i].x;
#endif

		// skip NaN and INF valued points
		if (!pcl_isfinite(point.x) ||
			!pcl_isfinite(point.y) ||
			!pcl_isfinite(point.z))
		{
			continue;
		}

		// skip zero valued points
		if (point.x * point.x + point.y * point.y + point.z * point.z < 0.0001)
		{
			continue;
		}

		// calculate vertical point angle and scan ID
#if (USE_RAW_LIDAR_COORDINATE)
		float angle = std::atan(point.z / std::sqrt(point.x * point.x + point.y * point.y));
		// float angle = std::atan(-point.y / std::sqrt(point.x * point.x + point.z * point.z));
#else
		float angle = std::atan(point.y / std::sqrt(point.x * point.x + point.z * point.z));
#endif //USE_RAW_LIDAR_COORDINATE

		int scanID = _scanMapper.getRingForAngle(angle);
		if (scanID >= _scanMapper.getNumberOfScanRings() || scanID < 0)
		{
			continue;
		}

		// calculate horizontal point angle
#if (USE_RAW_LIDAR_COORDINATE)
		float ori = -std::atan2(point.y, point.x);
		// float ori = -std::atan2(-point.x, point.z);
#else
		float ori = -std::atan2(point.x, point.z);
#endif //USE_RAW_LIDAR_COORDINATE

		if (!halfPassed)
		{
			if (ori < startOri - M_PI / 2)
			{
				ori += 2 * M_PI;
			}
			else if (ori > startOri + M_PI * 3 / 2)
			{
				ori -= 2 * M_PI;
			}

			if (ori - startOri > M_PI)
			{
				halfPassed = true;
			}
		}
		else
		{
			ori += 2 * M_PI;

			if (ori < endOri - M_PI * 3 / 2)
			{
				ori += 2 * M_PI;
			}
			else if (ori > endOri + M_PI / 2)
			{
				ori -= 2 * M_PI;
			}
		}

		// calculate relative scan time based on point orientation
		float relTime = config().scanPeriod * (ori - startOri) / (endOri - startOri);
		point.intensity = scanID + relTime;

		projectPointToStartOfSweep(point, relTime);

		_laserCloudScans[scanID].push_back(point);
	}

	processScanlines(scanTime, _laserCloudScans);
}

} // namespace ORB_SLAM2
