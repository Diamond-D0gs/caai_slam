#include "caai_slam/utils/calibration.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace caai_slam {
    calibration::calibration(const camera_intrinsics& intrinsics, const extrinsics& extrinsics, const cv::Size& image_size) : intrinsics(intrinsics), _extrinsics(extrinsics) {
        // Constant K matrix
        k = cv::Mat::eye(3, 3, CV_64F);
        k.at<double>(0, 0) = intrinsics.fx;
        k.at<double>(1, 1) = intrinsics.fy;
        k.at<double>(0, 2) = intrinsics.cx;
        k.at<double>(1, 2) = intrinsics.cy;

        // Construct D vector (k1, k2, p1, p2)
        d = cv::Mat::zeros(4, 1, CV_64F);
        d.at<double>(0) = intrinsics.k1;
        d.at<double>(1) = intrinsics.k2;
        d.at<double>(2) = intrinsics.p1;
        d.at<double>(3) = intrinsics.p2;

        // Initialize rectification maps
        // We use K as the new camera matrix to keep the same view
        cv::initUndistortRectifyMap(k, d, cv::Mat(), k, image_size, CV_16SC2, map0, map1);
    }

    void calibration::undistort_image(const cv::Mat& raw_image, cv::Mat& rectified_image) {
        if (!raw_image.empty())
            cv::remap(raw_image, rectified_image, map0, map1, cv::INTER_LINEAR);
    }

    std::vector<cv::KeyPoint> calibration::undistort_keypoints(const std::vector<cv::KeyPoint>& raw_kps) {
        if (raw_kps.empty())
            return {};

        std::vector<cv::Point2f> pts_in, pts_out;
        pts_in.reserve(raw_kps.size());
        for (const auto& kp : raw_kps)
            pts_in.push_back(kp.pt);

        // cv::undistortPoints by default returns normalized coordinates. To get pixels back we pass P = K.
        cv::undistortPoints(pts_in, pts_out, k, d, cv::Mat(), k);

        std::vector<cv::KeyPoint> result(raw_kps.begin(), raw_kps.end());
        for (size_t i = 0; i < result.size(); ++i)
            result[i].pt = pts_out[i]; // Update position of copied keypoints

        return result;
    }

} // namespace caai_slam