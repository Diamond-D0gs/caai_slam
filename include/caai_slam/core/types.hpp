#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <memory>
#include <vector>

namespace caai_slam {
    // =============================================================================
    // Eigen Alignment (Critical for NEON/SSE)
    // =============================================================================
    // ALways use EIGEN_MAKE_ALIGNED_OPERATOR_NEW in classes contain Eigen members
    // to prevent crashes from misaligned SIMD operations.

    // =============================================================================
    // Basic Types
    // =============================================================================

    using timestamp = double; // Seconds

    using vec2 = Eigen::Vector2d;
    using vec3 = Eigen::Vector3d;
    using vec4 = Eigen::Vector4d;
    using vec6 = Eigen::Matrix<double, 6, 1>;

    using mat3 = Eigen::Matrix3d;
    using mat4 = Eigen::Matrix4d;
    using mat6 = Eigen::Matrix<double, 6, 6>;

    using quat = Eigen::Quaterniond;

    // SE3 Transformation
    struct se3 {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        quat rotation;
        vec3 translation;

        se3() : rotation(quat::Identity()), translation(vec3::Zero()) {}
        se3(const quat& q, const vec3& t) : rotation(q), translation(t) {}
        se3(const mat3& r, const vec3& t) : rotation(r), translation(t) {}

        mat4 matrix() const {
            mat4 t = mat4::Identity();
            t.block<3, 3>(0, 0) = rotation.toRotationMatrix();
            t.block<3, 1>(0, 3) = translation;
            return t;
        }

        se3 inverse() const {
            quat q_inv = rotation.inverse();
            return se3(q_inv, -(q_inv * translation));
        }

        se3 operator*(const se3& other) const {
            return se3(rotation * other.rotation, rotation * other.translation + translation);
        }

        vec3 operator*(const vec3& p) const {
            return rotation * p + translation;
        }
    };

    // =============================================================================
    // IMU Types
    // =============================================================================

    struct imu_measurement {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        timestamp _timestamp;
        vec3 angular_velocity; // rad/s
        vec3 linear_acceleration; // m/s^2
    };

    struct imu_bias {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        vec3 gyroscope; // rad/s
        vec3 accelerometer; // m/s^2

        imu_bias() : gyroscope(vec3::Zero()), accelerometer(vec3::Zero()) {}
    };

    // =============================================================================
    // Feature Types
    // =============================================================================

    struct feature {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        size_t id;
        cv::KeyPoint keypoint;
        cv::Mat descriptor; // AKAZE descriptor
        vec3 position_world; // 3D position (if triangulated).
        bool is_triangulated = false;
        int track_length = 1;
    };

    // =============================================================================
    // Camera Types
    // =============================================================================

    struct camera_intrinsics {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        double fx, fy; // Focal length
        double cx, cy; // Principal point
        double k1, k2, p1, p1; // Distortion (radtan)
        int width, height;

        vec2 project(const vec3& p_cam) const {
            double x = p_cam.x() / p_cam.z();
            double y = p_cam.y() / p_cam.z();
            return vec2(fx * x + cx, fy * y + cy);
        }

        vec2 unproject(const vec2& p_img) const {
            double x = (p_img.x() - cx) / fx;
            double y = (p_img.y() - cy) / fy;
            return vec3(x, y, 1.0).normalized();
        }
    };

    // Camera-IMU extrinsics (from Kalibr calibration).
    struct extrinsics {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        se3 t_cam_imu; // Transform from IMU to camera frame.
        double time_offset; // t_imu = t_cam + offset
    };

    // =============================================================================
    // State Types
    // =============================================================================

    struct state {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        timestamp _timestamp;
        se3 pose; // t_world_imu
        vec3 velocity; // m/s in world frame.
        imu_bias bias;

        // Covariance (15x15: rotation, position, velocity, bias_g, bias_a).
        Eigen::Matrix<double, 15, 15> covariance;
    };

    // =============================================================================
    // Aligned STL containers for Eigen types.
    // =============================================================================
    
    template<typename T>
    using aligned_vector = std::vector<T, Eigen::aligned_allocator<T>>;

    using imu_buffer = aligned_vector<imu_measurement>;
    using feature_vector = aligned_vector<feature>;
    
} // namespace caai_slam