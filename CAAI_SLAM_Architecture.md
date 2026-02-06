# CAAI-SLAM: Architecture & Structure Summary

**Florida Atlantic University - Center for Connected Autonomy and AI**

**Version:** 0.1.0  
**Language:** C++17  
**Dependencies:** GTSAM, OpenCV, TEASER++, FBoW, Eigen3

---

## Executive Summary

CAAI-SLAM is a **Visual-Inertial SLAM (VIO) system** combining:
- **Frontend**: AKAZE feature detection + frame-to-frame/frame-to-map tracking
- **Backend**: GTSAM's IncrementalFixedLagSmoother for efficient graph optimization
- **Loop Closure**: FBoW place recognition + TEASER++ robust registration
- **Mapping**: Keyframe database, covisibility graph, and local map management

**Key Design Choice**: Trade-off between graph-based optimization accuracy and real-time performance on resource-constrained platforms (Raspberry Pi 5, Snapdragon mobile).

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CAAI-SLAM System                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────┐         ┌─────────────────────┐           │
│  │   Input Pipeline    │         │   Sensor Data       │           │
│  ├─────────────────────┤         ├─────────────────────┤           │
│  │ • Camera Image      │◄────┬───┤ • Camera (20-30 Hz) │           │
│  │ • IMU Measurements  │     │   │ • IMU (200+ Hz)     │           │
│  │ • Timestamps        │     │   │                     │           │
│  └──────────┬──────────┘     │   └─────────────────────┘           │
│             │                │                                       │
│             └────────────┬───┘                                       │
│                          ▼                                           │
│        ┌─────────────────────────────────────────┐                 │
│        │      Time Synchronization Module        │                 │
│        │ (Handles camera-IMU clock offset/drift) │                 │
│        └──────────────┬──────────────────────────┘                 │
│                       │                                             │
│          ┌────────────┴────────────┐                               │
│          ▼                         ▼                               │
│   ┌──────────────────┐     ┌──────────────────────┐               │
│   │ VIO Initializer  │     │ IMU Preintegration   │               │
│   │                  │     │ (High-freq buffer)   │               │
│   │ • Static align   │     │ • GTSAM integration  │               │
│   │ • Gravity est.   │     │ • Bias tracking      │               │
│   │ • Bias init      │     │                      │               │
│   └──────────┬───────┘     └──────────┬───────────┘               │
│              │                        │                            │
│              └──────────┬─────────────┘                            │
│                         ▼                                          │
│         ┌───────────────────────────┐                             │
│         │    Visual Frontend        │                             │
│         ├───────────────────────────┤                             │
│         │ • AKAZE Features          │                             │
│         │ • Feature Matcher (kNN)   │                             │
│         │ • Frame-to-Frame Track    │                             │
│         │ • Frame-to-Map Track      │                             │
│         │ • Keyframe Decision Logic │                             │
│         └──────────┬────────────────┘                             │
│                    │                                              │
│         ┌──────────┴──────────┐                                   │
│         ▼                     ▼                                   │
│    ┌─────────────┐     ┌─────────────────────┐                  │
│    │  Local Map  │     │  Fixed-Lag Smoother │                  │
│    │  (Tracking) │     │  (GTSAM Backend)    │                  │
│    │             │     │                     │                  │
│    │ • Active KF │     │ • Pose optimization │                  │
│    │ • Map Pts   │     │ • Vel. & Bias est.  │                  │
│    │ • Covis Gr. │     │ • Auto marginalize  │                  │
│    │ • Culling   │     │ • Memory bounded    │                  │
│    └──────┬──────┘     └──────────┬──────────┘                  │
│           │                       │                              │
│           │         ┌─────────────┘                              │
│           │         │                                            │
│           ▼         ▼                                            │
│       ┌──────────────────────────┐                              │
│       │   Loop Closure Module    │                              │
│       ├──────────────────────────┤                              │
│       │ 1. FBoW Place Recog.     │                              │
│       │ 2. Descriptor Matching   │                              │
│       │ 3. TEASER++ 3D-3D Reg.   │                              │
│       │ 4. Pose Correction       │                              │
│       └──────────┬───────────────┘                              │
│                  │                                              │
│    ┌─────────────┴─────────────┐                               │
│    ▼                           ▼                               │
│ ┌─────────────────┐      ┌──────────────────┐                │
│ │ Keyframe DB     │      │ Loop Constraint  │                │
│ │ (Global Pose    │      │ Injection        │                │
│ │  History)       │      │ (Pose prior)     │                │
│ └─────────────────┘      └──────────────────┘                │
│                                                               │
└─────────────────────────────────────────────────────────────────┘

                        ▼
                 [Trajectory Output]
                 [Map Visualization]
```

---

## Module Breakdown

### 1. Core System (`core/`)

#### **slam_system.hpp/cpp**
**Orchestrator** - Coordinates all subsystems and manages state machine.

**State Machine:**
```
NOT_INITIALIZED ──┐
                  ├─► INITIALIZING ──► TRACKING ──┐
                  │                               │
                  └──────────────────────────────┘
                     (Reset on loss)
```

**Responsibilities:**
- Initialize subsystems from YAML config
- Route IMU measurements to preintegration
- Route images through frontend
- Trigger keyframe creation and backend optimization
- Handle loop closure corrections

**Key Methods:**
- `process_imu()` - Integrate IMU measurements
- `process_image()` - Track visual features
- `get_current_state()` - Return pose, velocity, bias
- `reset()` - Clear all state for restart

---

#### **config.hpp/cpp**
**Configuration Parser** - Loads all parameters from YAML.

**Sections:**
```yaml
Camera:
  - Intrinsics (fx, fy, cx, cy)
  - Distortion (k1, k2, p1, p2)
  
Extrinsics:
  - T_cam_imu (calibration from Kalibr)
  - time_offset (seconds)

IMU:
  - Noise densities, random walk, frequency

Frontend:
  - AKAZE threshold, max features
  - Match ratio threshold, parallax min
  
Backend:
  - Lag time, relinearization settings
  
LoopClosure:
  - Enable flag, similarity threshold
```

---

#### **types.hpp**
**Type Definitions** - Foundation types with Eigen alignment.

**Critical Types:**
```cpp
struct se3 {           // SE(3) transformation
  Quaternion rotation;
  Vector3 translation;
};

struct imu_bias {      // Accelerometer + gyro bias
  Vector3 gyroscope;
  Vector3 accelerometer;
};

struct state {         // Full system state
  SE3 pose;            // t_world_imu
  Vector3 velocity;
  imu_bias bias;
  Matrix<double, 15, 15> covariance;  // 6 pose + 3 vel + 3 gyro_bias + 3 accel_bias
};

struct imu_measurement {
  double timestamp;
  Vector3 angular_velocity;
  Vector3 linear_acceleration;
};

template<T> using aligned_vector  // Eigen-aligned STL containers
```

**Design Note:** `EIGEN_MAKE_ALIGNED_OPERATOR_NEW` macro used throughout to prevent SIMD crashes on ARM/x86 platforms.

---

### 2. Frontend (`frontend/`)

#### **feature_extractor.hpp/cpp**
**Purpose**: Detect and describe features using AKAZE.

**Pipeline:**
```
Raw Image
    │
    ├─► AKAZE::detectAndCompute()
    │   • Threshold (config.frontend.akaze_threshold)
    │   • 4 octaves, M-LDB descriptor
    │
    ├─► Sort by response strength
    │
    └─► Limit to max_features
        (default 1000)
        
Output: vector<KeyPoint>, Mat(N x 61 bits)
```

**Implementation Notes:**
- Binary descriptors (M-LDB) → NORM_HAMMING distance
- Response strength = feature confidence
- Limiting prevents feature clustering

---

#### **feature_matcher.hpp/cpp**
**Purpose**: Match descriptors between frames using k-NN + Lowe's ratio test.

**Algorithm:**
```
Inputs: descriptors_query, descriptors_train

1. cv::BFMatcher(NORM_HAMMING)
   ├─► knnMatch(k=2)
   │   Output: two nearest neighbors per query feature
   │
2. Lowe's Ratio Test
   ├─► if (dist[0] < ratio_thresh * dist[1])
   │   └─► Accept match
   │   else
   │       └─► Reject (ambiguous)
   │
Output: vector<DMatch> (0-based indexing)
```

**Key Parameter:** `match_ratio_thresh` (default 0.8)
- Lower = stricter filtering
- Trade-off: Fewer but more reliable matches

---

#### **frame.hpp/cpp**
**Purpose**: Represents a transient image frame during tracking.

**Structure:**
```cpp
struct frame {
  uint64_t id;                    // Unique identifier
  double timestamp;
  
  // Visual data
  vector<KeyPoint> keypoints;     // AKAZE detections
  Mat descriptors;                // AKAZE binary descriptors
  
  // Pose estimate (from tracking)
  SE3 pose;                       // t_world_imu
  Vector3 velocity;
  imu_bias bias;
  
  // Feature-to-map associations (crucial for tracking)
  vector<shared_ptr<MapPoint>> map_points;  // Same size as keypoints
};
```

**Usage:** Frames are **temporary** objects created per image, processed through tracking, then either discarded or converted to keyframes.

**Key Method:**
- `has_map_point(idx)` - Check if feature has valid 3D association
- `get_camera_center()` - Compute camera position in world frame

---

#### **keyframe.hpp/cpp**
**Purpose**: Persistent keyframe with optimized poses and BoW vectors.

**Structure:**
```cpp
struct keyframe {
  uint64_t id;                                // Global ID
  double timestamp;
  gtsam::Pose3 pose;                          // Optimized by GTSAM
  
  // Visual data (same as frame)
  vector<KeyPoint> keypoints;
  Mat descriptors;
  vector<shared_ptr<MapPoint>> map_points;
  
  // Loop closure
  fbow::fBow bow_vec;                         // Sparse BoW (TF-IDF)
  fbow::fBow2 feat_vec;                       // Direct index
  
  // Topology
  vector<shared_ptr<KeyFrame>> connected_keyframes;
  vector<int32_t> connected_weights;          // Shared point counts
};
```

**Key Method:**
- `compute_bow()` - Convert AKAZE descriptors → BoW vectors
- `set_pose()` / `get_pose()` - Thread-safe pose access

---

#### **map_point.hpp/cpp**
**Purpose**: 3D landmark observed by multiple keyframes.

**Structure:**
```cpp
struct map_point {
  uint64_t id;
  Vector3 position;               // World frame
  Mat descriptor;                 // Representative descriptor
  
  // Observations
  vector<pair<weak_ptr<KeyFrame>, size_t>> observations;
  
  // Flags
  bool is_bad;                    // For culling
  uint64_t last_observed_frame_id;
};
```

**Lifecycle:**
```
Triangulated (2 keyframes see it)
    ├─► add_observation() [multiple KF calls]
    │
    ├─► Optimize by GTSAM
    │
    ├─ If outlier or <2 observations:
    │   └─► mark is_bad = true
    │
    └─► Culled by local_map::cull_map_points()
```

---

### 3. Visual-Inertial Odometry (VIO) (`vio/`)

#### **vio_initializer.hpp/cpp**
**Purpose**: Static initialization phase - estimate gravity and initial biases.

**Algorithm:**
```
Input: Buffered IMU (200+ samples at 200Hz = 1+ second)
       Buffered Frames (typically 5-10)

1. Static Detection
   ├─► Compute variance of acceleration magnitude
   │   └─► If variance < threshold → device is stationary
   │
2. Gravity Estimation
   ├─► Mean acceleration vector over buffer
   │   └─► Points "upward" (opposite gravity in body frame)
   │
3. Orientation Initialization
   ├─► Align body Z-axis with gravity (down)
   │   └─► q = Quaternion::FromTwoVectors(mean_acc, -Z_axis)
   │
4. Yaw Alignment (optional)
   ├─► Zero yaw by projecting to XY plane
   │   └─► Prevents initial spinning
   │
5. Output State
   └─► pose = identity (world = imu initial frame)
       velocity = zero
       bias_g = mean gyro
       bias_a = zero (hard to estimate without motion)
```

**Key Parameters:**
- `STATIC_THRESHOLD = 0.05` m²/s⁴
- `FRAME_BUFFER_MAX_SIZE = 20`

---

#### **imu_preintegration.hpp/cpp**
**Purpose**: Compress high-frequency IMU measurements into relative motion factors.

**Design Pattern:**
```
Uses GTSAM's PreintegratedCombinedMeasurements:
  └─ Accumulates: ΔRotation, ΔVelocity, ΔPosition
  └─ Tracks bias effect on each
  └─ Outputs as a single CombinedImuFactor
```

**Workflow:**
```
for each IMU sample:
  ├─► integrate(measurement, dt)
  │   └─► preintegrated->integrateMeasurement(accel, gyro, dt)
  │
  [Loop until next keyframe]
  
  └─► get_and_reset(new_bias)
      └─► Returns accumulated factor
      └─► Resets for next interval
```

**Covariance Handling:**
```
Noise Model (GTSAM):
  ├─ accelerometer noise density: config.imu.accel_noise_density
  ├─ gyroscope noise density: config.imu.gyro_noise_density
  ├─ accel random walk: config.imu.accel_random_walk
  └─ gyro random walk: config.imu.gyro_random_walk
  
Converted to discrete covariance matrices for optimization.
```

---

#### **visual_frontend.hpp/cpp**
**Purpose**: Frame-to-frame and frame-to-map tracking, keyframe decisions.

**Tracking Pipeline:**
```
Input: Raw image, predicted_pose (from IMU)

1. Feature Extraction
   ├─► feature_extractor->detect_and_compute(image)
   │
2. Frame-to-Frame Tracking
   ├─► feature_matcher->match(curr, prev)
   │   └─► Inherit map_points from previous frame
   │
3. Frame-to-Map Tracking
   ├─► local_map->get_map_points_in_view(pose)
   │   └─► Project map points; match descriptors
   │
4. Outlier Rejection
   ├─► RANSAC PnP (OpenCV)
   │   └─► Remove points with reprojection error > 2.0 px
   │
5. Keyframe Decision
   ├─► need_new_keyframe()?
   │   ├─ If tracked_points < min_threshold → YES
   │   ├─ If time > 0.5s since last KF → YES
   │   ├─ If displacement > 0.3m → YES
   │   └─ Else → NO
   │
Output: frame with pose and map_point associations
```

**Key Insight:** 
- Many frames are **not** keyframes
- Only frames with significant new information → keyframes
- Reduces backend workload

---

### 4. Backend Optimization (`backend/`)

#### **graph_optimizer.hpp/cpp**
**Purpose**: GTSAM ISAM2 incremental optimization (legacy, used for loop closure).

**Factor Graph Structure:**
```
Variables:
  X(i)  → Pose of keyframe i
  V(i)  → Velocity at keyframe i
  B(i)  → IMU bias at keyframe i
  L(j)  → Landmark (map point) j

Factors:
  PriorFactor(X(0), initial_pose)        [Initial pose prior]
  CombinedImuFactor(X(i), V(i), B(i) → X(i+1), V(i+1), B(i+1))
  ProjectionFactor(X(i), L(j) → measured_pixel)
  BetweenFactor(X(i) → X(j), measured_pose)  [Loop closure]
```

**Optimization Flow:**
```
add_keyframe(kf, imu_factor, prev_kf_id)
  ├─ Add CombinedImuFactor
  ├─ Add BetweenFactor for bias random walk
  ├─ Add ProjectionFactors for visible map points
  └─ Store in new_factors buffer

optimize()
  ├─ Update ISAM2 with batch of factors
  ├─ Extra iterations if loop closure
  └─ Return latest estimated state
```

---

#### **fixed_lag_smoother.hpp/cpp**
**Purpose**: GTSAM's IncrementalFixedLagSmoother - memory-bounded sliding window optimization.

**Key Advantage over ISAM2:**
```
ISAM2:
  └─ Keeps all poses in optimization
  └─ Memory grows unbounded
  └─ Good for short sequences

FixedLagSmoother:
  └─ Marginalizes poses older than lag_time (e.g., 5 seconds)
  └─ Constant memory footprint
  └─ Automatic Schur complement reduction
  └─ Ideal for real-time, long-duration SLAM
```

**Algorithm:**
```
add_keyframe(kf, imu_meas, prev_kf_id)
  ├─ Accumulate new factors in buffer
  │
optimize()
  ├─ Call smoother->update(factors, values, timestamps)
  │   └─ GTSAM's FixedLagSmoother::update() internally:
  │       ├─ Identifies old keys (timestamp < current_time - lag_time)
  │       ├─ Runs ISAM2 update on active window
  │       ├─ Marginalizes old keys via Schur complement
  │       └─ Discards marginalized variables
  │
  ├─ Detect which keyframes were marginalized
  │   └─ Used to prune local_map and loop detector
  │
  └─ Return marginalized_kf_ids
```

**Configuration:**
- `lag_time = 5.0` seconds (e.g., 10-15 keyframes at 2-3 Hz)
- `relinearize_threshold = 0.1`
- `relinearize_skip = 1`

---

#### **marginalization.hpp/cpp**
**Purpose**: Compute marginal covariances and pose entropy (uncertainty).

**Use Cases:**
1. **Covariance Extraction** - For visualization/uncertainty propagation
2. **Entropy Calculation** - Measure pose uncertainty (active keyframing signal)
3. **Convergence Detection** - Track when estimate stabilizes

**Computation:**
```
compute(graph, values)
  └─► Build gtsam::Marginals object
      (Inverts information matrix → expensive!)
      
get_pose_covariance(kf_id)
  └─► Return 6×6 pose covariance

get_pose_entropy(kf_id)
  └─► H = 0.5 * log(det(Σ)) + 0.5*k*(1 + log(2π))
      └─ Higher entropy = more uncertain
```

---

### 5. Loop Closure (`loop/`)

#### **loop_detector.hpp/cpp**
**Purpose**: Query FBoW database + geometric verification via TEASER++.

**Two-Stage Pipeline:**

**Stage 1: Place Recognition (FBoW)**
```
Input: Current keyframe's BoW vector

1. Accumulate scores from inverted index
   └─ For each word in KF->BoW:
       └─ Retrieve all KF's that contain that word
       └─ Accumulate: score += word_value_current * word_value_candidate

2. Filter candidates
   ├─ Exclude active keyframes (too close spatially)
   ├─ Exclude recent keyframes by ID (temporal exclusion)
   ├─ Keep only score > similarity_threshold

3. Sort and return top 3 candidates
```

**Stage 2: Geometric Verification (TEASER++)**
```
For each BoW candidate:
  
1. Extract matched 3D points
   ├─ Match descriptors between KF and candidate
   ├─ Apply Lowe's ratio test
   └─ Use only triangulated map points
   
2. Setup 3D-3D registration problem
   ├─ src_cloud = map points in query KF's frame
   ├─ target_cloud = map points in candidate KF's frame
   └─ Need minimum 12+ inliers
   
3. TEASER++ robust registration
   ├─ Input: two point clouds
   ├─ Output: R (rotation), t (translation)
   │          valid_flag, inlier_count
   │
   └─► Can handle up to 99% outliers via graduated non-convexity
   
4. Validate
   └─ If inliers >= min_matches_geom → Loop detected!
```

---

#### **geometric_verification.hpp/cpp**
Standalone geometric verification (separate from loop_detector).

**Interface:**
```cpp
result verify_3d_3d(query_kf, match_kf)
  ├─ Calls TEASER++ solver
  └─ Returns: {success, relative_pose, inlier_count}
```

---

#### **place_recognition.hpp/cpp**
Simplified BoW database (alternative to loop_detector).

**Methods:**
- `load_vocabulary(path)` - Load FBoW .fbow file
- `add_keyframe(kf)` - Index BoW vectors
- `query(kf, max_results)` - Top N candidates by similarity
- `clear()` - Reset database

---

#### **common.hpp/cpp**
Helper functions for loop closure.

**Key Function:**
```cpp
void get_matched_points(
    match_ratio_thresh,
    query_kf, candidate_kf,
    &src_cloud,      // 3×N matrix
    &target_cloud    // 3×N matrix
)
```

Extracts 3D point correspondences for TEASER++.

---

### 6. Mapping (`mapping/`)

#### **local_map.hpp/cpp**
**Purpose**: Active working set of keyframes and map points for tracking.

**Data Structures:**
```cpp
std::deque<shared_ptr<KeyFrame>> active_keyframes;
std::unordered_set<shared_ptr<MapPoint>> active_map_points;
covisibility_graph covis_graph;
```

**Responsibilities:**

1. **Keyframe Management**
   - `add_keyframe(kf)` → Insert into active window
   - `prune_old_keyframes(timestamp)` → Remove old KF (when GTSAM marginalizes)

2. **Map Point Management**
   - `add_map_point(mp)` → Insert new triangulated point
   - `fuse_map_points(target, victim)` → Merge duplicates after loop closure
   - `cull_map_points()` → Remove outliers (<2 observations or marked bad)

3. **Tracking Queries**
   - `get_map_points_in_view(camera_pose)` → Visible points for frame-to-map matching
   - `get_covisible_keyframes(kf)` → Spatial neighbors

**Pruning Strategy:**
```
When fixed_lag_smoother marginalizes KF:
  1. Loop closure corrects poses → recompute shared point counts
  2. Old KF passes age check → remove from active window
  3. KF leaves observations → remove isolated map points
  4. Archive old KF to keyframe_database for loop queries
```

---

#### **covisibility_graph.hpp/cpp**
**Purpose**: Topological graph of keyframe connectivity via shared map points.

**Data Structure:**
```cpp
std::unordered_map<uint64_t, std::unordered_map<uint64_t, int32_t>> adjacency;
// adjacency[kf_a][kf_b] = number of shared map points
```

**Key Methods:**

1. **`update(kf)`** - Recompute neighbors after KF added/KF points changed
   ```
   For each map point observed by KF:
     ├─ Get all KFs observing that point
     └─ Increment shared count
   
   Filter edges (weight < 15) → temporal neighbors
   Update KF's internal cache + neighbors' caches (bidirectional)
   ```

2. **`get_connected_keyframes(kf, min_weight)`** - Retrieve neighbors
   
3. **`get_best_covisibility_keyframes(kf, n)`** - Top N by edge weight

4. **`get_connected_component(kf, depth)`** - BFS for local BA window

5. **`remove_keyframe(kf)`** - Erase node and all incident edges

**Why Separate from GTSAM?**
```
GTSAM Factor Graph (dense, optimization-focused):
  └─ "Optimize KF X1, X2, ..., Xn with landmarks"
  └─ You specify what to optimize; GTSAM does math
  
Covisibility Graph (sparse, query-focused):
  └─ "Which KFs see the most shared points?"
  └─ Decides what window to hand to GTSAM
  └─ O(1) adjacency lookups vs. factor graph traversal
```

---

#### **keyframe_database.hpp/cpp**
**Purpose**: Global persistent storage of all keyframes.

**Structure:**
```cpp
std::unordered_map<uint64_t, shared_ptr<KeyFrame>> keyframes;
uint64_t last_id;
```

**Methods:**
- `add(kf)` - Store keyframe
- `get(id)` - Retrieve by ID
- `contains(id)` - Existence check
- `get_all_keyframes()` - Return sorted list (for trajectory saving)
- `remove(id)` - Delete (optional cleanup)

**Relationship to Loop Closure:**
```
LoopDetector queries both:
  1. All keyframes in database (for BoW matching)
  2. Only active keyframes in local_map (for tracking)
  
Old KF's still in database → can close loops months later
But not in local_map → don't interfere with tracking
```

---

### 7. Utilities (`utils/`)

#### **calibration.hpp/cpp**
**Purpose**: Camera calibration utilities (undistortion, projection).

**Functions:**
- `undistort_image(raw, rectified)` - Full image undistortion (slow, for viz)
- `undistort_keypoints(kps)` - Keypoint-wise undistortion (fast)
- `pixel_to_normalized(px)` - Convert to normalized image coordinates
- `get_camera_matrix()`, `get_dist_coeffs()` - Export OpenCV matrices

**Note:** Frontend usually works with raw (distorted) pixels; undistortion happens as needed.

---

#### **time_sync.hpp/cpp**
**Purpose**: Camera-IMU timestamp synchronization.

**Problem:**
```
Camera timestamp: 1000.0 s
IMU timestamp: 1000.045 s (50ms offset + potential drift)

Both on different hardware clocks → offset + drift over time
```

**Solution:**
```cpp
struct TimeSync {
  double offset;       // From Kalibr calibration
  double drift_rate;   // Optional online estimation
  
  cam_to_imu(t_cam) = t_cam + offset + drift_rate * t_cam
  imu_to_cam(t_imu) = (t_imu - offset) / (1 + drift_rate)
};
```

**Critical Method: `get_imu_between(t_cam_start, t_cam_end, imu_buffer)`**
```
Input: Camera frame interval [t_cam_start, t_cam_end]
       Deque of raw IMU measurements

Output: Vector of interpolated IMU measurements aligned to frame interval

Algorithm:
  1. Convert camera times to IMU times
  2. Binary search IMU buffer for range
  3. Linear interpolation at boundaries
  └─ Ensures exact start/end timestamps
  4. Return slice of measurements
  
Why interpolation?
  └─ IMU 200Hz, Camera 30Hz
  └─ Frame timestamp rarely aligns exactly with IMU sample
  └─ Without boundary interpolation, preintegration drifts
```

---

#### **triangulation.hpp/cpp**
**Purpose**: 3D reconstruction from two views.

**Functions:**

1. **`triangulate_dlt(pose_0, px_0, pose_1, px_1, &out_pt)`**
   ```
   Input: Two camera poses, normalized image coordinates
   
   Algorithm: Direct Linear Transform (DLT)
     1. Setup 4×4 design matrix from two views
     2. SVD decomposition
     3. Last column of V = homogeneous solution
     4. Normalize by w coordinate
     5. Chirality check (both cameras see point in front)
   
   Output: 3D world position or false if invalid
   ```

2. **`compute_parallax(pose_0, pose_1, point_3d)`**
   ```
   Input: Two poses, 3D point
   
   Computes: Angle between vectors from point to cameras
   
   Use case: Gate new map point creation
     └─ Small parallax → bad triangulation
     └─ Filter: require parallax > 1.0° (configurable)
   ```

---

## Data Flow & Execution Timeline

### **Initialization Phase**
```
t=0: System starts
     ├─ vio_initializer buffers IMU (1+ second)
     │
     ├─ visual_frontend extracts features
     │   └─ Zero pose (body frame)
     │
     └─ When is_static() && buffered_frames > 0:
         └─ vio_initializer::try_initialize()
            ├─ Estimate gravity from mean IMU accel
            ├─ Align world Z with gravity
            └─ Return initial state
            
     ├─ fixed_lag_smoother::initialize(first_kf, init_state)
     │   └─ Add pose/vel/bias priors
     │
     └─ Status = TRACKING
```

### **Tracking Phase (per frame)**
```
IMU arrives at 200 Hz:
  └─ time_sync converts timestamp
  └─ imu_preintegration::integrate()

Image arrives at 30 Hz:
  ├─ 1. VIO Prediction
  │  └─ imu_preintegration::predict(current_state)
  │     └─ Pose ≈ pose + velocity*dt + gravity*dt²
  │
  ├─ 2. Visual Tracking
  │  └─ visual_frontend::process_image(raw, ts, predicted_pose)
  │     ├─ AKAZE feature extraction
  │     ├─ Frame-to-frame matching
  │     ├─ Frame-to-map matching
  │     ├─ RANSAC outlier rejection
  │     └─ Return curr_frame (pose refined by tracking)
  │
  ├─ 3. Keyframe Decision?
  │  └─ visual_frontend::need_new_keyframe(curr_frame, last_kf)
  │     └─ If YES, proceed to step 4; else return (no backend update)
  │
  ├─ 4. Create Keyframe
  │  └─ Convert frame → keyframe
  │     └─ Copy visual data + pose
  │
  ├─ 5. Triangulation
  │  └─ Match curr_kf with prev_kf
  │     ├─ DLT triangulation for new correspondences
  │     ├─ Parallax gating (> 1.0°)
  │     └─ Add map points to local_map
  │
  ├─ 6. Backend Optimization
  │  └─ fixed_lag_smoother::add_keyframe(new_kf, imu_factors, prev_kf_id)
  │     ├─ Add CombinedImuFactor
  │     ├─ Add ProjectionFactors for map points
  │     └─ Stage new factors (not yet optimized)
  │
  ├─ 7. Optimize
  │  └─ fixed_lag_smoother::optimize()
  │     ├─ Call GTSAM's IncrementalFixedLagSmoother::update()
  │     ├─ Automatic marginalization of old KF
  │     └─ Return marginalized_kf_ids
  │
  ├─ 8. Prune Local Map
  │  └─ local_map::prune_old_keyframes(curr_ts)
  │     ├─ Remove KF's older than lag_time
  │     ├─ Remove isolated map points
  │     └─ Update covisibility_graph
  │
  ├─ 9. Update State Cache
  │  └─ current_state = latest_backend_state
  │     └─ Sync imu_preintegration bias
  │
  ├─ 10. Loop Closure Detection
  │  └─ loop_detector::detect_loop(new_kf, active_kfs)
  │     ├─ FBoW place recognition
  │     ├─ TEASER++ geometric verification
  │     └─ If match found:
  │         ├─ Add BetweenFactor to loop_closure_optimizer
  │         ├─ Optimize full pose graph
  │         └─ Correct all keyframe poses
  │
  └─ 11. Archive
     └─ keyframe_database::add(new_kf)
        └─ Store for future loop closure
```

---

## Threading Model

**Single-threaded design** (by default):
```
main() thread:
  ├─ process_imu(meas)           // Serial
  ├─ process_image(img, ts)      // Serial
  └─ Frontend + Backend all sync
```

**Thread-safety:**
- All shared data structures have `std::mutex`
- `slam_system` protects `current_state` with mutex
- Keyframe/MapPoint use internal mutexes for pose/position updates

**Async Optimization (optional enhancement):**
```
Tracker thread:
  └─ Visual frontend + IMU integration (real-time)
  
Backend thread:
  └─ Loop detection + graph optimization (lower priority)
```

---

## Configuration Example (EuRoC Format)

```yaml
Camera:
  width: 640
  height: 480
  fx: 458.654
  fy: 457.296
  cx: 367.215
  cy: 248.375
  k1: -0.28340811
  k2: 0.07395907
  p1: 0.00019359
  p2: 1.76187114e-05

Extrinsics:
  T_cam_imu: !!opencv-matrix
    rows: 4
    cols: 4
    data: [0.0148655, -0.999880, 0.00414029, 0.0430238,
           0.999557, 0.0149393, 0.025621, -0.0103545,
           -0.0257744, 0.00374618, 0.999662, -0.0609601,
           0, 0, 0, 1]
  time_offset: 0.0  # t_imu = t_cam + offset

IMU:
  accel_noise_density: 0.0096
  gyro_noise_density: 0.00016
  accel_random_walk: 0.003
  gyro_random_walk: 4.0e-05
  frequency: 200

Frontend:
  max_features: 1000
  akaze_threshold: 0.001
  match_ratio_thresh: 0.8
  min_matches_tracking: 15
  min_matches_init: 30
  parallax_min: 1.0  # degrees

Backend:
  lag_time: 5.0  # seconds
  relinearize_threshold: 0.1
  relinearize_skip: 1

LoopClosure:
  enable: true
  similarity_threshold: 0.05
  min_matches_geom: 12
  exclude_recent_n: 20
```

---

## Performance & Memory Profile

### **Memory Footprint**

| Component | Size | Notes |
|-----------|------|-------|
| GTSAM (minimal build) | 2-5 MB | Binary only |
| OpenCV (AKAZE, BFMatcher) | ~10 MB | Included |
| TEASER++ | ~1 MB | Sparse headers |
| FBoW | ~2 MB | Vocabulary varies (5-100 MB) |
| Active keyframes (1000 KF × ~100 KB) | ~100 MB | Pose, descriptors, points |
| Map points (5000 MP × ~50 B) | ~250 KB | Position only, referenced |
| **Total** | **~200-400 MB** | Depends on lag window |

### **Runtime Performance**

| Stage | Target | Hardware |
|-------|--------|----------|
| Feature extraction | <10 ms | Raspberry Pi 5 |
| Tracking | <30 ms | Raspberry Pi 5 |
| Backend (GTSAM) | <50 ms | Snapdragon |
| Loop detection | <100 ms | Off-main-thread |

---

## Design Decisions & Rationale

### **Why Fixed-Lag Smoother over EKF/MSCKF?**
```
EKF/MSCKF (Filter-based):
  ✗ Cannot "undo" past estimates
  ✗ No persistent map (features discarded after leaving view)
  ✗ Brittle initialization
  ✓ Constant memory & latency

FixedLagSmoother (Graph-based):
  ✓ Re-linearization corrects past errors
  ✓ Persistent map for re-observation
  ✓ Robust initialization
  ✓ Auto-marginalization bounds memory
  ✗ Higher computational cost
```

**CAAI-SLAM chooses:** Graph-based for **map quality** + **NEON/AVX** + **bounded memory** = real-time on modern hardware.

---

### **Why TEASER++ over Horn's Algorithm?**
```
Horn (1987, closed-form):
  ✓ Fast & simple
  ✗ Brittle to outliers (>10-20%)

TEASER++ (2020, graduated non-convexity):
  ✓ Handles up to 99% outliers
  ✓ Certifiable (guarantees optimality)
  ✓ Robust to scale/noise
  ✗ Slightly slower (acceptable for loop closure)
```

**CAAI-SLAM chooses:** TEASER++ for **loop robustness** in real-world conditions.

---

### **Why FBoW + GTSAM Backend + TEASER++ (3-part system)?**
```
Component 1: FBoW (Place Recognition)
  ├─ Fast coarse candidate retrieval (O(vocab_size))
  └─ Candidate set = 3-5 keyframes
  
Component 2: Descriptor Matching
  ├─ Refine candidates with visual similarity
  └─ Extract 3D-3D correspondences
  
Component 3: TEASER++ (Geometric Verification)
  ├─ Robust registration from noisy correspondences
  └─ Veto false positives
  
Advantage over single-method:
  └─ Cascaded filtering → speed + accuracy
  └─ Each stage rejects ~90% of candidates
```

---

## Future Optimization Opportunities

### **Memory Reduction**
- [ ] Descriptor compression (PCA, binary quantization)
- [ ] Map point sparser than 0.05m grid
- [ ] Lazy-load loop detector vocabulary

### **Speed Improvements**
- [ ] Async loop closure (separate thread)
- [ ] Incremental loop constraint correction
- [ ] Multi-threading backend (GTSAM supports batched factors)
- [ ] GPU feature extraction (CUDA AKAZE)

### **Robustness**
- [ ] IMU bias convergence tracking
- [ ] Rolling shutter handling
- [ ] Fisheye/wide-angle lens support
- [ ] Feature re-detection on tracking loss

---

## References

1. **GTSAM**: https://gtsam.org/ - Factor graph optimization
2. **TEASER++**: https://github.com/MIT-SPARK/TEASER-plusplus - Robust registration
3. **FBoW**: https://github.com/rmsalinas/fbow - Bag-of-words place recognition
4. **EuRoC Dataset**: http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets - Benchmark
5. **Kalibr**: https://github.com/ethz-asl/kalibr - Camera-IMU calibration

---

**Document Version:** 1.0  
**Last Updated:** February 2026  
**Maintainer:** Gabriel, CA-AI Research Group, FAU
