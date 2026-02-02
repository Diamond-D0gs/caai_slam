#include "caai_slam/vio/vio_initializer.hpp"

#include <numeric>
#include <cmath>

namespace caai_slam {
    bool vio_initializer::is_static() const {
        if (imu_buffer.size() < 10)
            return false;

        vec3 mean_acc = vec3::Zero();
        for (const auto& m : imu_buffer)
            mean_acc += m.linear_acceleration;
        mean_acc /= static_cast<double>(imu_buffer.size());
        
        double var_acc = 0.0;
        for (const auto& m : imu_buffer)
            var_acc += (m.linear_acceleration - mean_acc).squaredNorm();
        var_acc /= static_cast<double>(imu_buffer.size());

        // Static theshold: variance acceleration magnitude
        return var_acc < STATIC_THRESHOLD;
    }

    initialization_status vio_initializer::try_initialize(state& out_initial_state) {
        std::lock_guard<std::mutex> lock(mutex);

        // 1. Ensure sufficientdata (at least 1 second)
        if (imu_buffer.size() < static_cast<size_t>(_config.imu.frequency))
            return initialization_status::NOT_READY;

        // 2. Stationary check
        if (!is_static())
            return initialization_status::FAILED;

        // 3. Estimate gravity and gyro bias
        vec3 mean_acc = vec3::Zero(), mean_gyro = vec3::Zero();

        for (const auto& m : imu_buffer) {
            mean_acc += m.linear_acceleration;
            mean_gyro += m.angular_velocity;
        }

        mean_acc /= static_cast<double>(imu_buffer.size());
        mean_gyro /= static_cast<double>(imu_buffer.size());

        // 4. Align Z-axis with gravity
        // We assume mean_acc points upwards (against gravity) in the body frame when static.
        // We want r_wb such that r_wb * mean_acc = [0, 0, g]
        // This will set the initial orientation.
        quat q_world_imu = quat::FromTwoVectors(mean_acc.normalized(), vec3::UnitZ());

        // Remove yaw (align X axis horizontally)
        // Extract Euler angles (ZYX order typically) to zero out the yaw componenet.
        const vec3 euler = q_world_imu.toRotationMatrix().eulerAngles(2, 1, 0);

        // Reconstruct without yaw (euler[0])
        q_world_imu = quat(Eigen::AngleAxisd(0.0, vec3::UnitZ()) * Eigen::AngleAxisd(euler[1], vec3::UnitY()) * Eigen::AngleAxisd(euler[2], vec3::UnitX()));

        // 5. Construct initial state
        out_initial_state._timestamp = imu_buffer.back()._timestamp;
        out_initial_state.pose = se3(q_world_imu, vec3::Zero());
        out_initial_state.velocity = vec3::Zero();
        
        // Set initial biases
        out_initial_state.bias.gyroscope = mean_gyro;
        // Accelerometer bias is hard to distinguish from gravity without rotation, assume zero for static start.
        out_initial_state.bias.accelerometer = vec3::Zero();

        // Initial covariance (prior uncertainty)
        // High uncertainty on position, low on velocity/roll/pitch.
        out_initial_state.covariance = Eigen::Matrix<double, 15, 15>::Identity() * 0.1;

        return initialization_status::SUCCESS;
    }

    void vio_initializer::add_imu(const imu_measurement& meas) {
        std::lock_guard<std::mutex> lock(mutex);

        // Keep only the last 2 seconds of data for initialization check (approximate), assuming that 200Hz -> ~400 samples.
        const size_t max_samples = static_cast<size_t>(_config.imu.frequency * 2);
        while (imu_buffer.size() >= max_samples)
            imu_buffer.pop_front();

        imu_buffer.push_back(meas);
    }

    void vio_initializer::add_frame(const std::shared_ptr<frame>& f) {
        std::lock_guard<std::mutex> lock(mutex);

        // Keep buffer size constrained to manage memory footprint
        while (frame_buffer.size() >= FRAME_BUFFER_MAX_SIZE)
            frame_buffer.pop_front();

        frame_buffer.push_back(f);
    }

    void vio_initializer::reset() {
        std::lock_guard<std::mutex> lock(mutex);
        frame_buffer.clear();
        imu_buffer.clear();
    }

} // namespace caai_slam