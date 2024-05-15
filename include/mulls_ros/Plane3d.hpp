/*
 * @Author: kinggreat24
 * @Date: 2021-12-17 12:21:55
 * @LastEditTime: 2022-01-12 01:33:32
 * @LastEditors: kinggreat24
 * @Description: 
 * @FilePath: /orb_slam2_mapping/orb_slam2/include/Plane3d.hpp
 * 可以输入预定的版权声明、个性签名、空行等
 */

#ifndef PLANE_3D_H
#define PLANE_3D_H

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <math.h>

#include <ceres/jet.h>

namespace ORB_SLAM2
{
    template <class T>
    class Plane3D
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        friend Plane3D operator*(const Eigen::Transform<T, 3, Eigen::Isometry> &t, const Plane3D &plane);

        Plane3D()
        {
            Eigen::Matrix<T, 4, 1> v;
            v << 1., 0., 0., -1.;
            fromVector(v);
        }

        Plane3D(const Eigen::Matrix<T, 4, 1> &v)
        {
            fromVector(v);
        }

        inline Eigen::Matrix<T, 4, 1> toVector() const
        {
            return _coeffs;
        }

        inline const Eigen::Matrix<T, 4, 1> &coeffs() const { return _coeffs; }

        inline void fromVector(const Eigen::Matrix<T, 4, 1> &coeffs_)
        {
            _coeffs = coeffs_;
            normalize(_coeffs);
        }

        static T azimuth(const Eigen::Matrix<T, 3, 1> &v)
        {
            return ceres::atan2(v(1), v(0));
        }

        static T elevation(const Eigen::Matrix<T, 3, 1> &v)
        {
            return ceres::atan2(v(2), ceres::sqrt(v(0) * v(0) + v(1) * v(1)));
        }

        T distance() const
        {
            return -_coeffs(3);
        }

        Eigen::Matrix<T, 3, 1> normal() const
        {
            // return _coeffs.head<3>();
            return Eigen::Matrix<T, 3, 1>(_coeffs(0), _coeffs(1), _coeffs(2));
        }

        static Eigen::Matrix<T, 3, 3> rotation(const Eigen::Matrix<T, 3, 1> &v)
        {
            T _azimuth = azimuth(v);
            T _elevation = elevation(v);
            Eigen::AngleAxis<T> azimuth_v(_azimuth, Eigen::Matrix<T, 3, 1>::UnitZ());
            Eigen::AngleAxis<T> elevation_v(-_elevation, Eigen::Matrix<T, 3, 1>::UnitY());
            return (azimuth_v * elevation_v).toRotationMatrix();
        }

        //平面参数更新(z,e,d)
        inline void oplus(const Eigen::Matrix<T, 3, 1> &v)
        {
            //construct a normal from azimuth and evelation;
            T _azimuth = v[0];
            T _elevation = v[1];
            T s = std::sin(_elevation), c = std::cos(_elevation);
            Eigen::Matrix<T, 3, 1> n(c * std::cos(_azimuth), c * std::sin(_azimuth), s);

            // rotate the normal
            Eigen::Matrix<T, 3, 3> R = rotation(normal());
            T d = distance() + v[2];

            Eigen::Matrix<T, 3, 1> n_coeff = R * n;
            _coeffs(0) = n_coeff(0);
            _coeffs(1) = n_coeff(1);
            _coeffs(2) = n_coeff(2);
            // _coeffs.head<3>() = R * n;
            _coeffs(3) = -d;
            normalize(_coeffs);
        }

        inline Eigen::Matrix<T, 3, 1> ominus(const Plane3D &plane)
        {
            //construct the rotation that would bring the plane normal in (1 0 0)
            Eigen::Matrix<T, 3, 3> R = rotation(normal()).transpose();
            Eigen::Matrix<T, 3, 1> n = R * plane.normal();
            T d = distance() - plane.distance();
            return Eigen::Matrix<T, 3, 1>(azimuth(n), elevation(n), d);
        }

        inline Eigen::Matrix<T, 2, 1> ominus_ver(const Plane3D &plane)
        {
            //construct the rotation that would bring the plane normal in (1 0 0)
            Eigen::Matrix<T, 3, 1> v = normal().cross(plane.normal());
            Eigen::AngleAxisd ver(M_PI / 2, v / v.norm());
            Eigen::Matrix<T, 3, 1> b = ver * normal();

            Eigen::Matrix<T, 3, 3> R = rotation(b).transpose();
            Eigen::Matrix<T, 3, 1> n = R * plane.normal();
            return Eigen::Matrix<T, 2, 1>(azimuth(n), elevation(n));
        }

        inline Eigen::Matrix<T, 2, 1> ominus_par(const Plane3D &plane)
        {
            //construct the rotation that would bring the plane normal in (1 0 0)
            Eigen::Matrix<T, 3, 1> nor = normal();
            if (plane.normal().dot(nor) < 0)
                nor = -nor;
            Eigen::Matrix<T, 3, 3> R = rotation(nor).transpose();
            Eigen::Matrix<T, 3, 1> n = R * plane.normal();

            return Eigen::Matrix<T, 2, 1>(azimuth(n), elevation(n));
        }
        //protected:

        static inline void normalize(Eigen::Matrix<T, 4, 1> &coeffs)
        {
            Eigen::Matrix<T, 3, 1> n_coeff(coeffs(0), coeffs(1), coeffs(2));
            T n = n_coeff.norm();
            coeffs = coeffs * (1. / n);
            if (coeffs(3) < T(0.0))
                coeffs = -coeffs;
        }

        Eigen::Matrix<T, 4, 1> _coeffs;
    };

    // input t : transform matrix applying to the point
    template <typename T>
    Plane3D<T> operator*(const Eigen::Transform<T, 3, Eigen::Isometry> &t, const Plane3D<T> &plane)
    {
        Eigen::Matrix<T, 4, 1> v = plane._coeffs;
        Eigen::Matrix<T, 4, 1> v2;
        Eigen::Matrix<T, 3, 3> R = t.rotation();

        Eigen::Matrix<T, 3, 1> v_normal(v(0), v(1), v(2));
        Eigen::Matrix<T, 3, 1> v2_normal = R * v_normal;

        v2(0) = v2_normal(0);
        v2(1) = v2_normal(1);
        v2(2) = v2_normal(2);
        v2(3) = v(3) - t.translation().dot(v_normal);
        if (v2(3) < 0.0)
            v2 = -v2;
        return Plane3D<T>(v2);
    };

    // template <>
    // Plane3D<double> operator*(const Eigen::Transform<double, 3, Eigen::Isometry> &t, const Plane3D<double> &plane)
    // {
    //     Eigen::Matrix<double, 4, 1> v = plane._coeffs;
    //     Eigen::Matrix<double, 4, 1> v2;
    //     Eigen::Matrix<double, 3, 3> R = t.rotation();

    //     Eigen::Matrix<double, 3, 1> v_normal(v(0), v(1), v(2));
    //     Eigen::Matrix<double, 3, 1> v2_normal = R * v_normal;

    //     v2(0) = v2_normal(0);
    //     v2(1) = v2_normal(1);
    //     v2(2) = v2_normal(2);
    //     v2(3) = v(3) - t.translation().dot(v_normal);
    //     if (v2(3) < 0.0)
    //         v2 = -v2;
    //     return Plane3D<double>(v2);
    // };
} // namespace dre_slam

#endif //PLANE_3D_H