# CAAI-SLAM: State Machines & Execution Flows

## System State Machine

```
┌────────────────────────────────────────────────────────────────────┐
│                    CAAI-SLAM State Machine                         │
│                (slam_system::system_status enum)                   │
└────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────┐
                    │   NOT_INITIALIZED       │
                    │  (Waiting for first     │
                    │   image & IMU)          │
                    └──────────┬──────────────┘
                               │ First image arrives
                               │ status.store(INITIALIZING)
                               ▼
                    ┌─────────────────────────┐
                    │   INITIALIZING          │
                    │ • Buffer IMU (1+ sec)   │
                    │ • Buffer frames (5+)    │
                    │ • Check is_static()     │
                    │ • Estimate gravity      │
                    │ • Create first KF       │
                    │ • Initialize backend    │
                    └──────────┬──────────────┘
                               │
                      ┌────────┴────────┐
                      │                 │
                  SUCCESS            FAIL
                      │                 │
                      ▼                 │
                 ┌─────────┐            │
                 │TRACKING◄─┘           │
                 │         │            │
                 │ ◄───────┼────────────┘
                 │         │            Loop back
                 └────┬────┘
                      │
                  (Lost signal)
                  (Tracking < min matches)
                      │
                      ▼
                 ┌─────────┐
                 │LOST (not│
                 │impl.)   │
                 └─────────┘

═══════════════════════════════════════════════════════════════════════

Transitions Triggered By:
  • process_image(cv::Mat, ts) → State dispatch in process_image()
  • process_imu(imu_measurement) → Buffer to current state's handler
  • reset() → NOT_INITIALIZED

Status Queries:
  • get_status() → Returns current state (atomic)
```

---

## Initialization Sequence Diagram

```
TIME
  │
  │    IMU Buffer                Frame Buffer              System Status
  │   [starts empty]            [starts empty]           [NOT_INIT]
  │
  ├─ t=0  process_imu()  ────► [imu_0]
  │        vio_initializer::add_imu(imu_0)
  │
  ├─ t=0  process_image()────► [frame_0]         ──► Status = INITIALIZING
  │        vio_initializer::add_frame(frame_0)   (transition on first image)
  │
  ├─ t=Δt process_imu()  ────► [imu_0, imu_1]
  │
  ├─ t=2Δt process_imu()────► [imu_0, ..., imu_n]  (accumulated)
  │        ... more IMU samples
  │
  ├─ t=1.2s process_image() ─► [frame_0, frame_1]
  │         vio_initializer::try_initialize()
  │            ├─ Check imu_buffer.size() > 200 ✓
  │            ├─ Call is_static()
  │            │  └─ Compute var(acceleration) < 0.05 ✓
  │            ├─ Estimate gravity
  │            │  └─ mean_acc = sum(accel) / N
  │            ├─ Align Z-axis: q = FromTwoVectors(mean_acc, Z)
  │            ├─ Zero yaw (optional)
  │            └─ Return state {
  │                 pose = SE3(q, origin),
  │                 velocity = 0,
  │                 bias = {mean_gyro, 0}
  │               }
  │
  ├─ INIT SUCCESS ──────────► Create first keyframe
  │                           fixed_lag_smoother::initialize(kf, state)
  │                           └─ Add pose/vel/bias priors
  │
  │                           local_map::add_keyframe(kf)
  │                           keyframe_database::add(kf)
  │
  │                           imu_preintegration::reset(init_bias)
  │                           └─ Clear integration, set new bias
  │
  ├─ Status = TRACKING ────── Ready for normal operation
  │
  └─ Continue with normal tracking...

═══════════════════════════════════════════════════════════════════════
Possible Outcomes:

FAILURE: not enough static time
  → Keep buffering, try_initialize() returns NOT_READY
  → Transition when more data arrives

FAILURE: not enough feature variation
  → Try_initialize() returns FAILED
  → System stays INITIALIZING (restart on next image burst)

SUCCESS: Gravity estimated, pose initialized
  → Status = TRACKING
  → Backend ready to optimize
```

---

## Tracking Loop (Per Frame)

```
                 ┌─────────────────────────────────────┐
                 │  Image arrives: process_image()     │
                 └────────────┬────────────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │ Status = TRACKING?   │
                    └──┬──────────────┬────┘
                       │              │
                      YES            NO (INITIALIZING)
                       │              │ ─► process_initialization()
                       │              │    [See init diagram above]
                       ▼              │
                 ┌──────────────────────┐
                 │ process_tracking()   │
                 └────────┬─────────────┘
                          │
                          ▼
            ┌──────────────────────────────────┐
            │ 1. IMU PREDICTION                │
            │    (High-frequency state prop)   │
            └──┬───────────────────────────────┘
               │
               ├─ Get current_state (pose, vel, bias)
               ├─ Call imu_preintegration::predict(current_state)
               │  └─ Integrate all accumulated IMU since last keyframe
               │  └─ Output: predicted_pose ≈ current_state.pose
               │            + velocity*Δt + gravity*Δt²
               └─ Result: predicted_state
                          │
                          ▼
            ┌──────────────────────────────────┐
            │ 2. VISUAL TRACKING               │
            │    (Feature extraction & match)  │
            └──┬───────────────────────────────┘
               │
               ├─ visual_frontend::process_image(image, ts, pred_pose)
               │
               ├─ A. Feature Detection
               │  └─ feature_extractor::detect_and_compute(image)
               │     └─ AKAZE: ~100-300 keypoints + descriptors
               │
               ├─ B. Create Frame
               │  └─ auto curr_frame = std::make_shared<frame>(...)
               │     └─ curr_frame->pose = predicted_pose
               │
               ├─ C. Track from Previous Frame
               │  └─ IF last_frame exists:
               │      ├─ feature_matcher::match(curr, last)
               │      ├─ For each match:
               │      │  └─ curr_frame->map_points[i] = last->map_points[m.trainIdx]
               │      └─ Returns match count
               │
               ├─ D. Track from Local Map
               │  └─ local_map::get_map_points_in_view(curr_pose)
               │     ├─ Project map points through camera
               │     ├─ Filter by frustrum + depth
               │     └─ Return 3D-2D candidates
               │
               │  └─ For each map point mp:
               │      ├─ Best descriptor match from curr_frame descriptors
               │      └─ curr_frame->map_points[i] = mp
               │
               ├─ E. Outlier Rejection
               │  └─ IF tracked_points > 10:
               │      ├─ Collect 2D-3D correspondences
               │      ├─ cv::solvePnPRansac()
               │      └─ Nullify outliers (reprojection_error > 2.0px)
               │
               └─ Result: curr_frame with pose + map_point associations
                          │
                          ▼
            ┌──────────────────────────────────┐
            │ 3. KEYFRAME DECISION             │
            │    (Should we add to backend?)   │
            └──┬───────────────────────────────┘
               │
               ├─ visual_frontend::need_new_keyframe(curr_frame, last_kf)
               │
               ├─ Criteria:
               │  ├─ tracked_points < min_threshold → YES
               │  ├─ Δt > 0.5 seconds → YES
               │  ├─ displacement > 0.3m → YES
               │  └─ Else → NO
               │
               └┬─────────────────────┬──────────────┐
                │                     │              │
               NO                    YES          RETURN
                │                     │            (cycle)
                │                     ▼
                │          ┌────────────────────────────────┐
                │          │ 4. CREATE KEYFRAME              │
                │          │    & TRIANGULATION              │
                │          └──┬─────────────────────────────┘
                │             │
                │             ├─ new_kf = std::make_shared<keyframe>(...)
                │             ├─ Copy visual data from curr_frame
                │             │  └─ descriptors, keypoints, map_points
                │             │
                │             ├─ Match with previous keyframe
                │             │  └─ feature_matcher::match(new_kf, last_kf)
                │             │
                │             ├─ For each match (no prior association):
                │             │  ├─ DLT triangulation
                │             │  │  └─ triangulate_dlt(pose_0, px_0, pose_1, px_1)
                │             │  ├─ Parallax check: angle > 1.0° ✓
                │             │  ├─ Chirality check: depth > 0 in both frames ✓
                │             │  └─ Create map_point + add observations
                │             │
                │             ├─ Add to maps
                │             │  ├─ local_map::add_keyframe(new_kf)
                │             │  ├─ keyframe_database::add(new_kf)
                │             │  └─ loop_detector::add_keyframe(new_kf)
                │             │
                │             └─ Result: new_kf in all data structures
                │                        │
                │                        ▼
                │          ┌────────────────────────────────┐
                │          │ 5. BACKEND OPTIMIZATION         │
                │          │    (GTSAM FixedLagSmoother)     │
                │          └──┬─────────────────────────────┘
                │             │
                │             ├─ Get preintegrated IMU since last_kf
                │             │  └─ imu_preintegration::get_and_reset(bias)
                │             │     └─ Returns accumulated CombinedImuFactor
                │             │
                │             ├─ Add keyframe to backend
                │             │  └─ fixed_lag_smoother::add_keyframe(
                │             │       new_kf,
                │             │       imu_factors,
                │             │       last_kf->id
                │             │     )
                │             │     ├─ Stage CombinedImuFactor
                │             │     ├─ Stage BetweenFactor (bias RW)
                │             │     ├─ Stage ProjectionFactors
                │             │     └─ Buffer accumulated
                │             │
                │             ├─ Optimize
                │             │  └─ auto marginalized = fixed_lag_smoother::optimize()
                │             │     ├─ GTSAM::update(factors, values)
                │             │     ├─ Auto-marginalize KF's age > lag_time
                │             │     └─ Return marginalized_kf_ids
                │             │
                │             ├─ Prune Local Map
                │             │  └─ local_map::prune_old_keyframes(curr_ts)
                │             │     ├─ Remove KF's passed age threshold
                │             │     ├─ Remove isolated map points
                │             │     └─ Update covisibility_graph
                │             │
                │             ├─ Update State Cache
                │             │  └─ current_state = fixed_lag_smoother::get_latest_state()
                │             │     └─ Sync imu_preintegration bias
                │             │
                │             └─ Result: Optimized trajectory + local map
                │                        │
                │                        ▼
                │          ┌────────────────────────────────┐
                │          │ 6. LOOP CLOSURE DETECTION       │
                │          │    (FBoW + TEASER++)            │
                │          └──┬─────────────────────────────┘
                │             │
                │             ├─ loop_detector::detect_loop(new_kf, active_kfs)
                │             │
                │             ├─ Stage 1: Place Recognition (FBoW)
                │             │  ├─ Compute BoW if needed
                │             │  │  └─ new_kf->compute_bow(vocab)
                │             │  ├─ Query database
                │             │  │  └─ Accumulate word scores
                │             │  │  └─ Exclude active keyframes + recent
                │             │  └─ Return top 3 candidates
                │             │
                │             ├─ Stage 2: Geometric Verification
                │             │  │
                │             │  └─ For each candidate:
                │             │     ├─ get_matched_points(thresh, kf_query, kf_cand)
                │             │     │  └─ kNN match + ratio test on descriptors
                │             │     │  └─ Extract 3D-3D correspondences
                │             │     │
                │             │     ├─ teaser_solver->solve(src, target)
                │             │     │  └─ Robust registration (up to 99% outliers)
                │             │     │
                │             │     ├─ Validate inliers
                │             │     │  └─ If inliers >= min_matches_geom:
                │             │     │     └─ LOOP DETECTED!
                │             │     │
                │             │     └─ Return loop_result
                │             │
                │             └─┬──────────────────┬─────────────┐
                │               │                  │             │
                │            MATCH                NO MATCH    (Continue)
                │               │                  │
                │               ▼                  ▼
                │          ┌──────────┐      last_kf = new_kf
                │          │ Correct  │
                │          │ Poses    │ ─────┘
                │          └──────────┘
                │             │
                │             ├─ loop_closure_optimizer::add_loop_constraint()
                │             │  └─ Add BetweenFactor to pose graph
                │             │
                │             ├─ loop_closure_optimizer::optimize()
                │             │  └─ Full pose graph BA
                │             │
                │             ├─ Propagate corrected poses
                │             │  └─ Update keyframe_database + local_map
                │             │
                │             └─ last_kf = new_kf
                │
                └─ Cycle continues (wait for next image)

═══════════════════════════════════════════════════════════════════════
Timing Breakdown (Typical EuRoC @30 Hz):

  Feature extraction:     ~5 ms
  Matching:              ~3 ms
  RANSAC:                ~2 ms
  ─────────────────────────────
  Frontend subtotal:     ~10 ms  (must be < 33 ms)
  
  GTSAM optimization:    ~30 ms  (typically done every 2-3 keyframes)
  Loop closure:         ~100 ms  (async/separate thread recommended)
  ─────────────────────────────
  Backend subtotal:      ~50 ms (amortized)
  
  Total per frame:       ~15 ms (tracking only, no backend every frame)
  With backend:          ~50 ms (when keyframe added)
```

---

## Backend Optimization Timing

```
Keyframe i      Keyframe i+1      Keyframe i+2      Keyframe i+3
      │               │                 │                 │
      │               │                 │                 │
      ├─ add_keyframe │                 │                 │
      │  └─ Stage factors              │                 │
      │                                │                 │
      ├─ optimize()           ┌────────┤                 │
      │  ├─ GTSAM update      │        │                 │
      │  └─ Marginalize old   │        │                 │
      │                       │        │                 │
      │                       │        ├─ add_keyframe   │
      │                       │        │  └─ Stage factors
      │                       │        │                 │
      │                       │        ├─ optimize()     │
      │                       │        │  └─ Update, marginal.
      │                       │        │                 │
      │                       │        │                 ├─ add_keyframe
      │                       │        │                 │  └─ Stage
      │                       │        │                 │
      │                       │        │                 ├─ optimize()
      │                       │        │                 │  └─ Update
      │                       │        │                 │
      └──────────────────────┴────────┴─────────────────┘

Fixed-Lag Window:
  ├──── 5.0 seconds ────┐
  │                     │
  │   Active KF's       │  Marginalized
  │   ~15 KF's at       │  KF's archived
  │   2-3 Hz            │  to database
  │                     │
  └─────────────────────┴───►
                        Automatic Schur complement
```

---

## Loop Closure Detection Flow

```
Current Keyframe (new_kf)
         │
         ├─ Compute BoW vector
         │  └─ fbow::Vocabulary::transform(descriptors, bow_vec, feat_vec)
         │
         ▼
┌──────────────────────────────────────────┐
│ FBoW Place Recognition (FAST)            │
├──────────────────────────────────────────┤
│                                           │
│ For each word_id in new_kf.bow_vec:      │
│   └─ Retrieve inverted_index[word_id]    │
│      └─ List of all KF's with that word  │
│                                           │
│ Accumulate scores:                       │
│   score[candidate] +=                    │
│     new_kf.bow_vec[word_id] ×            │
│     candidate.bow_vec[word_id]           │
│                                           │
│ Filter:                                   │
│   ├─ Exclude active KF's (spatial)       │
│   ├─ Exclude recent KF's (temporal)      │
│   ├─ Keep only score > threshold         │
│                                           │
│ Sort & return top 3                      │
│                                           │
└──────────────┬───────────────────────────┘
               │ candidates = [kf_a, kf_b, kf_c]
               │
               ▼
For each candidate in candidates:
│
├─ ┌──────────────────────────────────────┐
│  │ Descriptor-based Matching (MEDIUM)   │
│  ├──────────────────────────────────────┤
│  │                                       │
│  │ cv::BFMatcher::knnMatch(              │
│  │   new_kf.descriptors,                │
│  │   candidate.descriptors,             │
│  │   matches, k=2                       │
│  │ )                                     │
│  │                                       │
│  │ Lowe's Ratio Test:                   │
│  │   for each match pair (m[0], m[1]):  │
│  │     if m[0].distance < 0.8*m[1]:     │
│  │       accepted_matches.push(m[0])    │
│  │                                       │
│  │ Extract 3D-3D point clouds:           │
│  │   ├─ src_cloud: map points in KF i   │
│  │   ├─ target_cloud: map points in KF j│
│  │   └─ Only matched + triangulated     │
│  │                                       │
│  │ Need: src_cloud.cols() >= 12         │
│  │                                       │
│  └──────────────┬───────────────────────┘
│                 │ src_cloud, target_cloud
│                 │
│                 ▼
│  ┌──────────────────────────────────────┐
│  │ TEASER++ Registration (ROBUST)       │
│  ├──────────────────────────────────────┤
│  │                                       │
│  │ teaser_solver->solve(src, target)    │
│  │   ├─ Graduated Non-Convexity (GNC)  │
│  │   ├─ Certifiable optimality          │
│  │   ├─ Robust to ~99% outliers         │
│  │   │                                   │
│  │   └─ Output:                         │
│  │       ├─ solution.R (rotation)       │
│  │       ├─ solution.t (translation)    │
│  │       ├─ solution.valid (bool)       │
│  │       └─ solution.inliers (indices)  │
│  │                                       │
│  │ Count inliers:                       │
│  │   for each point:                    │
│  │     error = ||R*src - target||       │
│  │     if error < noise_bound:          │
│  │       ++inlier_count                 │
│  │                                       │
│  │ Validate:                            │
│  │   if inlier_count >= min_matches:    │
│  │     ✓ LOOP MATCH FOUND!              │
│  │                                       │
│  │   else:                              │
│  │     ✗ Try next candidate             │
│  │                                       │
│  └──────────────────────────────────────┘
│
└─ Try next candidate

Final Result:
  ├─ loop_result {
  │   is_detected: true,
  │   query_kf: new_kf,
  │   match_kf: kf_match,
  │   t_match_query: SE3(R, t),
  │   inliers: count
  │ }
  │
  └─ Add pose constraint to loop_closure_optimizer
     └─ Correct all keyframe poses
```

---

## IMU Preintegration State Diagram

```
State: NOT_INTEGRATING (initial)
  │
  ├─ called: reset(bias)
  │
  ▼
State: INTEGRATING
  ├─ preintegrated* != nullptr
  ├─ current_bias set
  │
  ├─ for each IMU sample:
  │  ├─ integrate(meas, dt)
  │  │  └─ preintegrated->integrateMeasurement(accel, gyro, dt)
  │  │
  │  ├─ Accumulate:
  │  │  ├─ ΔRotation (exp(∫ω dt))
  │  │  ├─ ΔVelocity (∫a dt)
  │  │  ├─ ΔPosition (∫∫a dt²)
  │  │  └─ Bias sensitivity matrices
  │  │
  │  └─ Update covariance
  │
  ├─ Can call: predict(state)
  │  ├─ Uses integrated delta
  │  ├─ Applies bias sensitivity
  │  └─ Returns predicted_state
  │
  ├─ When keyframe added:
  │  ├─ get_and_reset(new_bias)
  │  │  ├─ Returns accumulated preintegrated
  │  │  ├─ Resets integration buffer
  │  │  └─ Sets new_bias
  │  │
  │  └─ State → INTEGRATING (next interval)

Covariance Evolution:
  Σ(0) = initial noise
  │
  Σ(t) = Σ(t-1) + [accel_noise, gyro_noise] * dt
  │
  Σ(T) = accumulated uncertainty over interval
         (grows with sqrt(T) for white noise)
```

---

## Local Map Lifecycle

```
┌────────────────────────────────────┐
│ Local Map (Active Working Set)     │
├────────────────────────────────────┤
│                                     │
│ Deque<KeyFrame>: [KF_i, ... KF_n]  │
│ Set<MapPoint>: active 3D points    │
│ CovisibilityGraph: connectivity    │
│                                     │
└────────────────────────────────────┘
         ▲           ▲           ▼
         │           │           │
    ADD KF      REMOVE OLD    QUERY FOR
    (tracking)  KF (prune)    TRACKING
         │           │           │
         │           │           └─► get_map_points_in_view(pose)
         │           │               ├─ Project to camera
         │           │               ├─ Frustrum + depth check
         │           │               └─ Return visible {MapPoint*}
         │           │
         │           └─ prune_old_keyframes(timestamp)
         │               ├─ Find KF's age > lag_time
         │               ├─ Remove from active_keyframes
         │               ├─ For each KF:
         │               │  ├─ mp->remove_observation(kf)
         │               │  ├─ If mp.observations.empty():
         │               │  │  └─ active_map_points.erase(mp)
         │               │  └─ covis_graph.remove_keyframe(kf)
         │               │
         │               └─ Archive KF to keyframe_database
         │
         └─ add_keyframe(new_kf)
             ├─ active_keyframes.push_back(new_kf)
             ├─ For each mp in new_kf.map_points:
             │  └─ active_map_points.insert(mp)
             └─ covis_graph.update(new_kf)
                 └─ Count shared points with all others
                 └─ Recompute edges + caches

═══════════════════════════════════════════════════════════════════

Timeline:

t=0   KF_0 (1st)          ├─ Start lag window
      │
t=0.3 KF_1                ├─ Active
      │
t=0.6 KF_2                ├─ Active
      │
t=0.9 KF_3                ├─ Active
      │
...
t=4.7 KF_14               ├─ Active (front of window)
      │
t=5.0 KF_15               ├─ Check age:
      │                    │  last_kf.timestamp = 5.0
      │                    │  threshold = 5.0 - 5.0 = 0.0
      │                    │  KF_0 age = 5.0 > 0.0 → REMOVE
      ▼
     KF_0 → PRUNE          ├─ No more in local_map
            ↓              ├─ But still in keyframe_database
         ARCHIVE           └─ Available for loop closure queries

Active window shifts forward continuously,
keeping memory bounded at ~15-20 KF's.
```

---

## Covisibility Graph Update

```
Step 1: COUNT SHARED POINTS
─────────────────────────────────

Current keyframe: KF_new
Map points:      [MP_0, MP_1, ..., MP_200]

For each MP in KF_new.map_points:
  └─ mp->get_observations() → {(KF_i, idx_i), (KF_j, idx_j), ...}
     ├─ Is KF_i != KF_new?
     │  └─ shared_count[KF_i]++
     │
     └─ Is KF_j != KF_new?
        └─ shared_count[KF_j]++

Result: shared_count = {
  KF_a: 27,    // 27 shared points
  KF_b: 15,
  KF_c: 8,
  KF_d: 3      // Below threshold
}

Step 2: FILTER BY THRESHOLD
─────────────────────────────────

For each (neighbor, weight) in shared_count:
  ├─ If weight >= min_weight (15):
  │  └─ Keep edge
  └─ Else:
     └─ Discard edge (temporal neighbors only)

Result: valid_neighbors = {
  KF_a: 27,
  KF_b: 15,
  KF_c: 8     ✗ Removed
}

Step 3: UPDATE CENTRAL ADJACENCY MAP
─────────────────────────────────────

// Bidirectional update
adjacency[KF_new][KF_a] = 27
adjacency[KF_a][KF_new] = 27

adjacency[KF_new][KF_b] = 15
adjacency[KF_b][KF_new] = 15

// Old edges removed
adjacency[KF_new].erase(KF_d)  // No longer connected
adjacency[KF_d].erase(KF_new)

Step 4: UPDATE KF INTERNAL CACHES
─────────────────────────────────

For KF_new:
  ├─ Sort by weight (descending)
  ├─ KF_new.connected_keyframes = [KF_a, KF_b, ...]
  └─ KF_new.connected_weights = [27, 15, ...]

For each neighbor (KF_a, KF_b, ...):
  ├─ Update their cache to include KF_new
  └─ Re-sort (maintaining descending weight)

Result: Query get_connected_keyframes(KF_new) 
         returns [KF_a, KF_b] in O(1) time

═════════════════════════════════════════════════════════════════

Visual Example:

Before update:
  KF_1 ──15── KF_2 ──20── KF_3
           │
           └──8── KF_4

After adding KF_new with:
  shared_count[KF_1] = 25
  shared_count[KF_2] = 18
  shared_count[KF_3] = 5 (filtered)

New graph:
  KF_1 ──15── KF_2 ──20── KF_3
           │        
           │        ┌────25────┐
           │        │           │
           └──8── KF_4      KF_new
                            └──18── KF_2

Covisibility now reflects actual map point sharing.
```

---

**State Machine & Flow Version:** 1.0  
**Last Updated:** February 2026
