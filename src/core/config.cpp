#include "caai_slam/core/config.hpp"

#include <opencv2/core/persistence.hpp>
#include <opencv2/core/eigen.hpp>

namespace caai_slam {
    bool config::loadFromYAML(const std::string& filename) {
        const cv::FileStorage fs(filename, cv::FileStorage::READ);

        if (!fs.isOpened()) {
            std::cerr << "Config: Failed to open file " << filename << std::endl;
            return false;
        }

        // =========================================================================
        // Camera Parameters
        // =========================================================================

        const cv::FileNode n_cam = fs["Camera"];

        if (n_cam.empty()) {
            std::cerr << "Config: Missing 'Camera' node" << std::endl;
            return false;
        }

        camera.width = static_cast<int32_t>(n_cam["width"]);
        camera.height = static_cast<int32_t>(n_cam["height"]);
        camera.fx = static_cast<double>(n_cam["fx"]);
        camera.fy = static_cast<double>(n_cam["fy"]);
        camera.cx = static_cast<double>(n_cam["cx"]);
        camera.cy = static_cast<double>(n_cam["cy"]);

        // Distortion (assuming RadTan: k1, k2, p1, p2)
        camera.k1 = static_cast<double>(n_cam["k1"]);
        camera.k2 = static_cast<double>(n_cam["k2"]);
        camera.p1 = static_cast<double>(n_cam["p1"]);
        camera.p2 = static_cast<double>(n_cam["p2"]);

        // =========================================================================
        // Extrinsics
        // =========================================================================

        const cv::FileNode n_ext = fs["Extrinsics"];

        if (n_ext.empty()) {
            std::cout << "Config: Missing 'Extrinsics' node" << std::endl;
            return false;
        }
        
        // Expecting a 4x4 matrix in YAML under "T_cam_imu" or standard EuRoC format T_BS (Body to Sensor).
        cv::Mat t_cv;
        n_ext["T_cam_imu"] >> t_cv;

        if (!t_cv.empty() && t_cv.rows == 4 && t_cv.cols == 4) {
            mat4 t_eigen;
            cv::cv2eigen(t_cv, t_eigen);

            // Store T_cam_imu
            _extrinsics.t_cam_imu = se3(t_eigen.block<3,3>(0,0), t_eigen.block<3,1>(0,3));
        }

        // Time offset
        _extrinsics.time_offset = static_cast<double>(n_ext["time_offset"]);

        // =========================================================================
        // IMU Parameters
        // =========================================================================

        const cv::FileNode n_imu = fs["IMU"];

        if (n_imu.empty()) {
            std::cerr << "Config: Missing 'IMU' Node." << std::endl;
            return false;
        }

        imu.accel_noise_density = static_cast<double>(n_imu["accel_noise_density"]);
        imu.gyro_noise_density = static_cast<double>(n_imu["gyro_noise_density"]);
        imu.accel_random_walk = static_cast<double>(n_imu["accel_random_walk"]);
        imu.gyro_random_walk = static_cast<double>(n_imu["gyro_random_walk"]);
        imu.frequency = static_cast<uint32_t>(static_cast<double>(n_imu["frequency"])); 
        // Double cast required due to no direct uint32_t cast operator.

        // =========================================================================
        // Algorithm Parameters
        // =========================================================================

        // Frontend
        const cv::FileNode n_front = fs["Frontend"];

        if (n_front.empty()) {
            std::cerr << "Config: Missing 'Frontend' Node." << std::endl;
            return false;
        }

        frontend.max_features = static_cast<int32_t>(n_front["max_features"]);
        frontend.akaze_threshold = static_cast<float>(n_front["akaze_threshold"]);
        frontend.min_matches_tracking = static_cast<int32_t>(n_front["min_matches_tracking"]);
        frontend.min_matches_init = static_cast<int32_t>(n_front["min_matches_init"]);

        // Backend
        const cv::FileNode n_back = fs["Backend"];

        if (n_back.empty()) {
            std::cerr << "Config: Missing 'Backend' Node." << std::endl;
            return false;
        }

        backend.lag_time = static_cast<double>(n_back["lag_time"]);
        backend.relinearize_threshold = static_cast<double>(n_back["relinearize_threshold"]);

        // Loop Closure
        const cv::FileNode n_loop = fs["LoopClosure"];

        if (n_loop.empty()) {
            std::cerr << "Config: Missing 'LoopClosure' Node." << std::endl;
            return false;
        }

        loop.enable = static_cast<int32_t>(n_loop["enable"]) != 0;
        loop.similarity_threshold = static_cast<float>(n_loop["similarity_threshold"]);

        return true;
    }

    void config::print() const {
        std::cout << "--- CAAI-SLAM Configuration ---\n";
        std::cout << "Camera: " << camera.width << "x" << camera.height << " fx=" << camera.fx << " fy=" << camera.fy << "\n";
        std::cout << "IMU Rate: " << imu.frequency << " Hz\n";
        std::cout << "Backend Lag: " << backend.lag_time << " s\n";
        std::cout << "Frontend Max Feat: " << frontend.max_features << "\n";
        std::cout << "-------------------------------" << std::endl;
    }

} // namespace caai_slam