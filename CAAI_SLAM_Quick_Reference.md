# CAAI-SLAM: Module Dependency & Quick Reference

## Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                    CORE SYSTEM (slam_system)                 │
│              Routes data to all subsystems                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
   ┌─────────┐      ┌──────────┐     ┌──────────────┐
   │ Config  │      │TimeSyncDB│     │VIOInitialize │
   │(YAML)   │      └──────────┘     └──────────────┘
   └─────────┘            ▲                   │
                          │                   │
                          └───┬───────────────┘
                              │
                              ▼
                      ┌─────────────────┐
                      │ IMUPreintegr.   │
                      │ (GTSAM Native)  │
                      └────────┬────────┘
                               │
        ┌──────────────────────┴──────────────────────┐
        │                                             │
        ▼                                             ▼
   ┌──────────────┐                         ┌─────────────────┐
   │   Frontend   │                         │FixedLagSmoother │
   │              │◄────┐                   │  (GTSAM Native) │
   ├──────────────┤     │                   ├─────────────────┤
   │• AKAZE Extr. │     │                   │• ISAM2 Core     │
   │• Matcher     │     │                   │• Auto Margin.   │
   │• Frame Track │     │                   │• Factor Graph   │
   │• Map Track   │     │    ┌──────────────┤• Covariance Comp│
   │• KF Decision │     │    │              └────────┬────────┘
   └──────┬───────┘     │    │                       │
          │             │    │              ┌────────┴────────┐
          └─────────────┼────┼──────────────┼────────┐        │
                        │    │              │        │        │
                        ▼    ▼              ▼        ▼        ▼
                   ┌──────────────┐   ┌─────────┐┌──────┐┌─────────┐
                   │  LocalMap    │   │KeyFrame  ││MapPt.││Marginal.│
                   │              │   │Database  ││Struct││Covar.   │
                   ├──────────────┤   └─────────┘└──────┘└─────────┘
                   │• Active KF's │
                   │• Active MP's │
                   │• CovisGraph  │
                   │• Culling     │
                   └──────┬───────┘
                          │
        ┌─────────────────┴──────────────────┐
        │                                    │
        ▼                                    ▼
   ┌──────────────────┐           ┌──────────────────────┐
   │  LoopDetector    │           │LoopClosureOptimizer  │
   │                  │           │   (GTSAM ISAM2)      │
   ├──────────────────┤           └────────┬─────────────┘
   │• FBoW Query      │                    │
   │• Descriptor Match│                    └──────┐
   │• TEASER++ Reg.  │                           │
   └──────┬───────────┘                          │
          │                                      │
          └──────────────────┬───────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ Pose Corrected │
                    │  Trajectory    │
                    └────────────────┘
```

---

## Class Relationships (UML-lite)

```
┌─────────────────────────────────────────────────────────┐
│ TYPES & PRIMITIVES                                       │
├─────────────────────────────────────────────────────────┤
│ • se3 (pose)                                             │
│ • imu_bias                                               │
│ • state (pose + vel + bias + covariance)                 │
│ • imu_measurement                                        │
│ • camera_intrinsics / extrinsics                         │
│ • aligned_vector<T> (Eigen-aligned STL)                 │
└──────────┬──────────────────────────────────────────────┘
           │ used by all
           ▼
┌────────────────────────────────────────────────────────────────┐
│ FRONTEND PIPELINE                                               │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│ raw_image ──► feature_extractor ──► vector<KeyPoint>            │
│               (AKAZE)                cv::Mat descriptors        │
│                   │                                              │
│                   └──► frame::frame()                            │
│                            │                                     │
│                            ├──► feature_matcher                  │
│                            │    (kNN + Lowe's ratio)             │
│                            │         │                           │
│                            │         ├──► Frame-to-Frame Track  │
│                            │         └──► Frame-to-Map Track     │
│                            │              (from local_map)       │
│                            │                 │                   │
│                            └─────────────────┤                   │
│                                            ▼                    │
│                               visual_frontend::process_image()  │
│                                  Returns: frame with pose        │
│                                           & associations         │
│                                                                  │
│ Decision: need_new_keyframe()?                                  │
│   ├─ NO  → cycle (frame discarded)                              │
│   └─ YES → convert frame → keyframe → Backend                   │
└────────────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────────┐
│ MAP POINT TRIANGULATION & TRACKING                              │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│ prev_keyframe ──► feature_matcher ──► matches                   │
│ curr_keyframe                            │                      │
│                                          ├─► [DLT Triangulation]
│                                          │   • Parallax gating   │
│                                          │   • Chirality check   │
│                                          │                      │
│                                          └─► map_point::map_point()
│                                              └─► local_map::add_map_point()
│                                              │                  │
│                                              └──► observations[]│
│                                                   └─ keyframe    │
│                                                                  │
│ Linking: frame/keyframe::map_points[i] ◄──────► MapPoint*      │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────────┐
│ BACKEND OPTIMIZATION (Fixed-Lag Smoother)                       │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│ new_keyframe ──► fixed_lag_smoother::add_keyframe()             │
│ + imu_factors       ├─ Add CombinedImuFactor                    │
│                     ├─ Add ProjectionFactors                    │
│                     └─ Stage in buffer                          │
│                                                                  │
│ Periodically:                                                    │
│   fixed_lag_smoother::optimize()                                │
│     ├─ GTSAM::FixedLagSmoother::update()                        │
│     ├─ Auto-marginalize old KF (age > lag_time)                 │
│     └─ Return: marginalized_kf_ids + latest_state               │
│                                                                  │
│ Output: Optimized KF poses + MapPoint positions                 │
│         Covariance estimates                                    │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
           │
           ├──────────────┐
           │              │
           ▼              ▼
┌──────────────────┐  ┌────────────────────┐
│  local_map       │  │ keyframe_database  │
│  (active window) │  │  (persistent)      │
│                  │  │                    │
│ • Add KF         │  │ • Stores all KF's  │
│ • Prune old KF   │  │ • Query by ID      │
│ • Add MP         │  │ • For loop closure │
│ • Fuse MP's      │  │                    │
│ • Cull bad MP    │  │ BoW vectors:       │
│ • CovisGraph     │  │ • bow_vec (sparse) │
│   update         │  │ • feat_vec (index) │
│                  │  │                    │
│ Output: Visible  │  │ Output: Place cand.│
│ points for       │  │          feature   │
│ tracking         │  │          vectors   │
└──────────────────┘  └────────────────────┘
           │                     │
           └──────────┬──────────┘
                      ▼
┌────────────────────────────────────────────────────────────────┐
│ LOOP CLOSURE DETECTION                                          │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│ current_keyframe:                                                │
│   ├─ Compute BoW vector (if not done)                           │
│   └─ Query loop_detector                                        │
│                                                                  │
│ loop_detector::detect_loop(curr_kf, active_kfs):                │
│   ├─ Stage 1: FBoW Place Recog                                  │
│   │   └─ query_database(BoW) → top 3 candidates                │
│   │                                                              │
│   ├─ For each candidate:                                        │
│   │   └─ Stage 2: Geometric Verification                       │
│   │       ├─ get_matched_points(descriptors)                   │
│   │       ├─ TEASER++ robust registration                      │
│   │       │   └─ Relative SE3 pose                             │
│   │       └─ If valid: return loop_result                      │
│   │                                                              │
│   └─ If no match: return empty                                 │
│                                                                  │
│ Output: loop_result {                                           │
│   is_detected,       // bool                                    │
│   query_kf,          // shared_ptr                              │
│   match_kf,          // shared_ptr                              │
│   t_match_query,     // se3                                     │
│   inliers            // int                                     │
│ }                                                                │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
           │
           ▼
   [If match found]:
           │
           ├─ loop_closure_optimizer::add_loop_constraint()
           │  (Separate GTSAM graph for pose graph optimization)
           │
           └─ Correct all keyframe poses
              └─ Propagate to local_map + frontend
```

---

## Thread-Safe Data Access Patterns

```
═══════════════════════════════════════════════════════════════
TYPE 1: Read-Only / Single-Write-Many-Read (shared_mutex)
═══════════════════════════════════════════════════════════════

Example: keyframe_database::get(id)

std::shared_lock<std::shared_mutex> lock(mutex);  // Read lock
return keyframes.find(id)->second;


═══════════════════════════════════════════════════════════════
TYPE 2: Exclusive Modification (mutex)
═══════════════════════════════════════════════════════════════

Example: local_map::add_keyframe(kf)

std::unique_lock<std::shared_mutex> lock(mutex);  // Write lock
active_keyframes.push_back(kf);
covis_graph.update(kf);


═══════════════════════════════════════════════════════════════
TYPE 3: Mutable Per-Object (per-keyframe mutex)
═══════════════════════════════════════════════════════════════

Example: keyframe::set_pose(p)

struct keyframe {
  mutable std::mutex mutex;  // Per-object lock
  gtsam::Pose3 pose;
};

void keyframe::set_pose(const gtsam::Pose3& p) {
  std::lock_guard<std::mutex> lock(mutex);
  pose = p;
}


═══════════════════════════════════════════════════════════════
TYPE 4: Cache with Occasional Full Recompute (recursive_mutex)
═══════════════════════════════════════════════════════════════

Example: marginalization::get_pose_covariance(kf_id)

This allows:
  - get_pose_covariance() [read]
  - get_pose_entropy() [read] (uses get_pose_covariance_unlocked)
  - compute() [recompute] (full graph)

std::lock_guard<std::recursive_mutex> lock(mutex);
```

---

## Quick Configuration Reference

### **Essential Parameters for EuRoC/VIO**

```yaml
# Sensor calibration (from Kalibr)
Extrinsics:
  T_cam_imu: [R(3x3), t(3x1)]   # IMU→Camera rotation + translation
  time_offset: 0.0               # t_imu = t_cam + offset

# Detector parameters
Frontend:
  max_features: 1000             # Limit features per frame
  akaze_threshold: 0.001         # Sensitivity (lower = more features)
  match_ratio_thresh: 0.8        # Lowe's ratio test (Unique enough?)
  min_matches_tracking: 15       # Minimum tracked points
  parallax_min: 1.0              # Minimum parallax for triangulation (degrees)

# Backend parameters
Backend:
  lag_time: 5.0                  # Sliding window (seconds)
  relinearize_threshold: 0.1     # ISAM2 re-linearization trigger

# Loop closure
LoopClosure:
  enable: true
  similarity_threshold: 0.05     # FBoW score threshold
  min_matches_geom: 12           # Inliers required (TEASER++)
  exclude_recent_n: 20           # Don't match neighbors (by keyframe ID)
```

### **Performance Tuning**

| Goal | Action |
|------|--------|
| Faster tracking | ↑ `max_features`, ↓ `akaze_threshold` |
| Fewer tracking failures | ↓ `min_matches_tracking` |
| Fewer false triangulations | ↑ `parallax_min` |
| More robust loop closure | ↓ `similarity_threshold`, ↓ `min_matches_geom` |
| Lower memory | ↓ `lag_time` (fewer active KF's) |
| More accurate optimization | ↑ `lag_time`, ↓ `relinearize_threshold` |

---

## Build Flags Summary

```bash
# Platform-specific compilation
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_SIMD=ON \
      -DMINIMAL_BUILD=ON \  # For embedded (Raspberry Pi)
      -DUSE_SYSTEM_GTSAM=ON \
      -DUSE_SYSTEM_TEASER=ON \
      ..

# ARM64 (Snapdragon, Raspberry Pi 5)
cmake ... -DCMAKE_CXX_FLAGS="-march=armv8-a"

# x86-64 (Desktop/Laptop)
cmake ... -DCMAKE_CXX_FLAGS="-mavx2 -mfma"

# Size optimization
cmake ... -DCMAKE_BUILD_TYPE=MinSizeRel \
          -DGTSAM_ENABLE_BOOST_SERIALIZATION=OFF \
          -DGTSAM_WITH_TBB=OFF

# Build
make -j$(nproc)

# Verify NEON (ARM)
llvm-objdump -d libgtsam.so | grep fmla  # Should see SIMD instructions
```

---

## Module Testing Checklist

```
┌─ Core Types (types.hpp)
│  └─ ✓ SE3 multiplication, inversion
│  └─ ✓ Aligned vector construction
│
├─ Frontend (frontend/)
│  └─ ✓ AKAZE feature count consistent
│  └─ ✓ Matcher finds correct correspondences
│  └─ ✓ Frame pose initialized to predicted
│
├─ VIO (vio/)
│  └─ ✓ Initializer converges on static motion
│  └─ ✓ IMU preintegration delta matches hand calc
│  └─ ✓ State prediction reasonable
│
├─ Backend (backend/)
│  └─ ✓ GTSAM graph builds without errors
│  └─ ✓ Fixed-lag smoother marginalizes old KF
│  └─ ✓ Optimization improves residuals
│
├─ Mapping (mapping/)
│  └─ ✓ Local map culls old keyframes
│  └─ ✓ Covisibility graph updates bidirectionally
│  └─ ✓ Map point fusion works
│
├─ Loop Closure (loop/)
│  └─ ✓ FBoW vocabulary loads
│  └─ ✓ Place recognition finds candidates
│  └─ ✓ TEASER++ registers outlier-robust
│
└─ Utilities (utils/)
   └─ ✓ Time sync interpolates boundaries
   └─ ✓ DLT triangulation produces valid points
   └─ ✓ Parallax computation matches theory
```

---

## Common Issues & Debugging

### **Issue: Low tracking quality**
```
Symptoms: Features lost, tracking resets to init

Diagnosis:
  1. Check IMU preintegration bias estimate
     └─ Print imu_preintegration->get_current_bias()
  
  2. Check feature extraction
     └─ Visualize detected keypoints
     └─ Is AKAZE threshold too high?
  
  3. Check frame-to-map matching
     └─ Are map points visible in current frame?
     └─ Verify get_map_points_in_view() filtering

Solution:
  • Lower akaze_threshold (more features)
  • Lower match_ratio_thresh (weaker ratio test)
  • Reduce min_matches_tracking threshold
  • Check camera calibration (T_cam_imu correct?)
```

### **Issue: Memory growing unbounded**
```
Symptoms: RAM usage increases over time

Diagnosis:
  1. Is fixed_lag_smoother marginalizing?
     └─ Check log output for "Smoother Indeterminant"
     └─ Verify optimize() called per keyframe
  
  2. Are map points being culled?
     └─ Check local_map::num_map_points()
     └─ Should plateau around 5000-10000
  
  3. Are keyframes archived?
     └─ keyframe_database size should grow
     └─ local_map active_keyframes should stay ~20-50

Solution:
  • Lower backend.lag_time (marginalize more aggressively)
  • Call local_map::cull_map_points() more frequently
  • Verify keyframe_database::add() is called
```

### **Issue: Loop closure not triggering**
```
Symptoms: Revisit areas but no loop closure correction

Diagnosis:
  1. Is loop_detector vocabulary loaded?
     └─ Check loop_detector->load_vocabulary() return
  
  2. Are candidates found?
     └─ Print query_database results
  
  3. Is geometric verification failing?
     └─ Check TEASER++ inlier count
     └─ Verify min_matches_geom threshold

Solution:
  • Lower similarity_threshold (more permissive)
  • Lower min_matches_geom (fewer required inliers)
  • Verify descriptors are AKAZE compatible (binary)
  • Check vocabulary file integrity
```

---

## Performance Profiling Commands

```bash
# Get CPU/memory usage during run
time ./caai_slam_node --config euroc.yaml --dataset dataset_path

# Detailed timing with perf
perf record -g ./caai_slam_node ...
perf report

# Memory profiling (valgrind)
valgrind --tool=massif ./caai_slam_node ...
ms_print massif.out.* | head -50

# Thread inspection
gdb ./caai_slam_node
(gdb) info threads
(gdb) thread 1
(gdb) bt  # Backtrace
```

---

**Quick Reference Version:** 1.0  
**Last Updated:** February 2026
