#include "caai_slam/utils/time_sync.hpp"

#include <algorithm>
#include <stdexcept>

namespace caai_slam {
    /**
     * @brief Linearly interpolate between two IMU measurements
     * 
     * @param m_0 First measurement
     * @param m_1 Second measurement
     * @param time Target timestamp
     * 
     * @return Interpolated measurement
     */
    inline imu_measurement interpolate(const imu_measurement& m_0, const imu_measurement& m_1, const double time) {
        imu_measurement m_out = {};
        m_out._timestamp = time;

        const double dt = m_1._timestamp - m_0._timestamp;
        if (dt <= 1e-9)
            return m_0;

        const double alpha = (time - m_0._timestamp) / dt;

        m_out.linear_acceleration = (1.0 - alpha) * m_0.linear_acceleration + alpha * m_1.linear_acceleration;
        m_out.angular_velocity = (1.0 - alpha) * m_0.angular_velocity + alpha * m_1.angular_velocity;

        return m_out;
    }

    std::vector<imu_measurement> time_sync::get_imu_between(const double t_cam_start, const double t_cam_end, const std::deque<imu_measurement>& imu_buffer) const {
        if (imu_buffer.empty())
            return {};

        const double t_start = cam_to_imu(t_cam_start), t_end = cam_to_imu(t_cam_end);
        if (t_end < t_start)
            return {};

        // 1. Find the first measurements >= t_start and >= t_end
        const auto lower_bound_func = [](const imu_measurement& m, const double t) { return m._timestamp < t; };
        const auto it_start = std::lower_bound(imu_buffer.begin(), imu_buffer.end(), t_start, lower_bound_func);
        const auto it_end = std::lower_bound(imu_buffer.begin(), imu_buffer.end(), t_end, lower_bound_func);

        // Check boundary validity, previous measurements are needed to interpolate between the start and end points.
        if (it_start == imu_buffer.begin() || it_end == imu_buffer.begin() || it_start == imu_buffer.end() || it_end == imu_buffer.end())
            return {};

        std::vector<imu_measurement> result;

        // 2. Interpolate start point
        result.push_back(interpolate(*(it_start - 1), *it_start, t_start));

        // 3. Add all measurements strictly inside (t_start, t_end)
        for (auto it = it_start; it != it_end; ++it)
            if (it->_timestamp > t_start && it->_timestamp < t_end)
                result.push_back(*it);

        // 4. Interpolate end point (exactly to t_end)
        result.push_back(interpolate(*(it_end - 1), *it_end, t_end));

        return result;
    }

} // namespace caai_slam