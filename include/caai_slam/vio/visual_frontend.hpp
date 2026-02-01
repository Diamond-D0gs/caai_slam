#pragma once

#include "caai_slam/frontend/feature_extractor.hpp"
#include "caai_slam/frontend/feature_matcher.hpp"
#include "caai_slam/mapping/local_map.hpp"
#include "caai_slam/frontend/frame.hpp"
#include "caai_slam/core/config.hpp"
#include "caai_slam/core/types.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <mutex>

namespace caai_slam {
    /**
     * @brief Manages the visual tracking pipeline
     * 
     * Coordinates feature extraction, frame-to-frame tracking, and frame-to-map tracking.
     * It is reponsible for making keyframe insertion decisions.
     */
    class visual_frontend {
    private:
        config _config;

        std::shared_ptr<local_map> _local_map;

        std::unique_ptr<feature_extractor> extractor;
        std::unique_ptr<feature_matcher> matcher;

        std::shared_ptr<frame> last_frame;

        // Thread safety
        mutable std::mutex mutex;

        /**
         * @brief Track features from the previous frame to the current frame
         * 
         * @param curr_frame The current frame
         * @param prev_frame The previous frame
         * 
         * @return Number of successful matches
         */
        uint32_t track_last_frame(std::shared_ptr<frame>& curr_frame, const std::shared_ptr<frame>& prev_frame);

        /**
         * @brief Project points from the local map into the current frame to find matches
         * 
         * @param curr_frame The current frame
         * 
         * @return Number of map points successfully associcated
         */
        uint32_t track_local_map(std::shared_ptr<frame>& curr_frame);

        /**
         * @brief Remove outlier map point associations using a geometric check
         * 
         * @param curr_frame The frame to clean
         */
        void outlier_rejection(std::shared_ptr<frame>& curr_frame);

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        /**
         * @param cfg System configuration settings
         * @param l_map Pointer to the local map for spatial tracking
         */
        visual_frontend(const config& cfg, std::shared_ptr<local_map> l_map);

        /**
         * @brief Process a new incoming image
         * 
         * @param image Grayscale input images
         * @param ts Acquisition timestamp
         * @param predicted_pose Initial pose estimate (e.g., from IMU)
         * 
         * @return The processed frame containing tracking results
         */
        std::shared_ptr<frame> process_image(const cv::Mat& image, const timestamp ts, const se3& predicted_pose);

        /**
         * @brief Determine if the current frame should be promoted to a keyframe
         * 
         * @param curr_frame The frame currently being processed
         * @param last_kf The most recently added keyframe
         * 
         * @return True if a new keyframe is required, false otherwise
         */
        bool need_new_keyframe(const std::shared_ptr<frame>& curr_frame, const std::shared_ptr<keyframe>& last_kf);
    };

} // namespace caai_slam