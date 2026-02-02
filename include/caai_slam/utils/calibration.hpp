#pragma once

#include "caai_slam/core/types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace caai_slam {
    /**
     * @brief Utility class for camera calibration operations
     * 
     * Handles the conversion between distored (raw) and undistored (rectified)
     * image coordinates using pinhole radial-tangential model.
     */
    class calibration {
    private:
        camera_intrinsics intrinsics;
        extrinsics _extrinsics;

        cv::Mat k; // Intrinsic matrix
        cv::Mat d; // Distortion coefficients

        // Cached rectification maps
        cv::Mat map0, map1;
        
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        /**
         * @param intrinsics Camera intrinsic parameters (k and distortion coefficients)
         * @param extrinsics Camera-IMU extrinsics parameters (t_cam_imu)
         */
        calibration(const camera_intrinsics& intrinsics, const extrinsics& extrinsics, const cv::Size& image_size);

        /**
         * @brief Undistort an entire image
         * 
         * Useful for visualization or dense reconstruction.
         * 
         * @param raw_image Input distorted image
         * @param rectified_image Output undistorted image
         */
        void undistort_image(const cv::Mat& raw_image, cv::Mat& rectified_image);

        /**
         * @brief Undistort a set of keypoints
         * 
         * Converts feature locations from distorted pixel coordinates
         * to idealized pinhole coordinates (undistorted pixel).
         * 
         * @param raw_kps Input vector of detected keypoints
         * 
         * @return Vector of undistorted keypoints
         */
        std::vector<cv::KeyPoint> undistort_keypoints(const std::vector<cv::KeyPoint>& raw_kps);

        /**
         * @brief Convert a point from pixel coordinates to normalized image coordinates
         * 
         * Applies inverse K matrix. Assume point is already undistorted.
         * 
         * @param pixel_point Point in undistorted pixel coordinates (u, v)
         * 
         * @return Point in normalized image plane (x, y)
         */
        inline vec2 pixel_to_normalized(const vec2& pixel_point) const { return vec2((pixel_point.x() - intrinsics.cx) / intrinsics.fx, (pixel_point.y() - intrinsics.cy) / intrinsics.fy); }

        /**
         * @brief Get the camera matrix (K) as an OpenCV Mat
         * 
         * @return Camera matrix
         */
        inline cv::Mat get_camera_matrix() const { return k.clone(); }

        /**
         * @brief Get the distortion coefficients (D) as an OpenCV Mat
         * 
         * @return 4x1 or 5x1 vector of distortion coefficients
         */
        inline cv::Mat get_dist_coeffs() const { return d.clone(); }

        /**
         * @brief Get the extrinsics object
         * 
         * @return constant reference to the extrinsics object
         */
        const extrinsics& get_extrinsics() const { return _extrinsics; }
    };

} // namespace caai_slam