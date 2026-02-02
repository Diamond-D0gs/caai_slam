#pragma once

#include "caai_slam/core/types.hpp"

namespace caai_slam {
    /**
     * @brief Triangulate a single 3D point from two views using DLT.
     * 
     * @param pose_0 Pose of the first camera (t_world_cam_0)
     * @param point_0 Normalized coords (x/z, y/z) in the first image
     * @param pose_1 Pose of the second camera (t_world_cam_1)
     * @param point_1 Normalized coords (x/z, y/z) in the second image
     * @param out_point_world Output triangulated point in the world frame
     * 
     * @return True if triangulation was successful and the point is in front of both cameras, false if invalid or has negative depth.
     */
    bool triangulate_dlt(const se3& pose_0, const vec2& point_0, const se3& pose_1, const vec2& point_1, vec3& out_point_world);

    /**
     * @brief Compute the parallax angle subtended by a 3D point as seen from two camera poses
     * 
     * The parallax is the angle between the vectors from the 3D point to each camera center.
     * A larger parallax indicates a wider baseline relative to the point's depth, which
     * generally yields a more reliable triangulation. Useful as a gating check before
     * triangulating a new map point.
     * 
     * @param pose_0 Pose of the first camera (t_world_cam_0)
     * @param pose_1 Pose of the second camera (t_world_cam_1)
     * @param point_world The 3D point in world coordinates
     * 
     * @return The parallax angle in degrees
     */
    double compute_parallax(const se3& pose_0, const se3& pose_1, const vec3& point_world);

} // namespace caai_slam