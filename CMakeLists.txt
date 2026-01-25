cmake_minimum_required(VERSION 3.16)
project(CAAI_SLAM VERSION 0.1.0 LANGUAGES CXX)

# =============================================================================
# Build Configuration
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type defaults
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Options
option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_EXAMPLES "Build example applications" ON)
option(USE_SYSTEM_GTSAM "Use system-installed GTSAM instead of building from source" ON)
option(USE_SYSTEM_TEASER "Use system-installed TEASER++ instead of building from source" ON)
option(ENABLE_SIMD "Enable SIMD optimizations (SSE4/AVX on x86, NEON on ARM)" ON)
option(MINIMAL_BUILD "Strip unnecessary components for smaller binary" OFF)

# =============================================================================
# Platform Detection
# =============================================================================
if(WIN32)
    message(STATUS "Configuring for Windows")
    set(PLATFORM_WINDOWS TRUE)
    add_definitions(-DPLATFORM_WINDOWS)
    # Windows-specific flags
    add_definitions(-D_USE_MATH_DEFINES)
    add_definitions(-DNOMINMAX)
elseif(UNIX AND NOT APPLE)
    message(STATUS "Configuring for Linux")
    set(PLATFORM_LINUX TRUE)
    add_definitions(-DPLATFORM_LINUX)
else()
    message(WARNING "Unsupported platform - some features may not work")
endif()

# =============================================================================
# Architecture Detection & SIMD Configuration
# =============================================================================
include(CheckCXXCompilerFlag)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    set(ARCH_ARM64 TRUE)
    message(STATUS "Architecture: ARM64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    set(ARCH_ARM32 TRUE)
    message(STATUS "Architecture: ARM32")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(ARCH_X64 TRUE)
    message(STATUS "Architecture: x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i686|i386")
    set(ARCH_X86 TRUE)
    message(STATUS "Architecture: x86")
endif()

# SIMD flags based on architecture
if(ENABLE_SIMD)
    if(ARCH_ARM64)
        # ARM64 has NEON by default
        set(SIMD_FLAGS "-march=armv8-a")
        add_definitions(-DUSE_NEON)
        message(STATUS "SIMD: NEON (ARM64)")
    elseif(ARCH_ARM32)
        check_cxx_compiler_flag("-mfpu=neon" COMPILER_SUPPORTS_NEON)
        if(COMPILER_SUPPORTS_NEON)
            set(SIMD_FLAGS "-mfpu=neon -mfloat-abi=hard")
            add_definitions(-DUSE_NEON)
            message(STATUS "SIMD: NEON (ARM32)")
        endif()
    elseif(ARCH_X64 OR ARCH_X86)
        if(MSVC)
            # MSVC uses /arch flags
            check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
            if(COMPILER_SUPPORTS_AVX2)
                set(SIMD_FLAGS "/arch:AVX2")
                add_definitions(-DUSE_AVX2)
                message(STATUS "SIMD: AVX2")
            else()
                set(SIMD_FLAGS "/arch:SSE4.2")
                add_definitions(-DUSE_SSE4)
                message(STATUS "SIMD: SSE4.2")
            endif()
        else()
            # GCC/Clang
            check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
            check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE42)
            if(COMPILER_SUPPORTS_AVX2)
                set(SIMD_FLAGS "-mavx2 -mfma")
                add_definitions(-DUSE_AVX2)
                message(STATUS "SIMD: AVX2 + FMA")
            elseif(COMPILER_SUPPORTS_SSE42)
                set(SIMD_FLAGS "-msse4.2")
                add_definitions(-DUSE_SSE4)
                message(STATUS "SIMD: SSE4.2")
            endif()
        endif()
    endif()
endif()

# =============================================================================
# Compiler Flags
# =============================================================================
if(MSVC)
    # MSVC flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP ${SIMD_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /O1 /Ob1 /DNDEBUG")
    
    # Disable specific warnings
    add_definitions(/wd4267 /wd4244 /wd4305)  # Conversion warnings
else()
    # GCC/Clang flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra ${SIMD_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -Os -DNDEBUG")
    
    if(MINIMAL_BUILD)
        # Flags for minimal binary size (from conversation)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
    endif()
endif()

# =============================================================================
# Dependencies
# =============================================================================

# --- Eigen3 (Required by GTSAM, TEASER++) ---
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")

# Eigen alignment for NEON/SSE (from conversation)
add_definitions(-DEIGEN_MAKE_ALIGNED_OPERATOR_NEW)

# --- OpenCV (AKAZE, Feature Matching) ---
find_package(OpenCV 4.0 REQUIRED COMPONENTS
    core
    imgproc
    features2d
    calib3d
    highgui
    imgcodecs
)
message(STATUS "OpenCV found: ${OpenCV_VERSION}")

# --- GTSAM (Graph Backend, IMU Preintegration) ---
if(USE_SYSTEM_GTSAM)
    find_package(GTSAM REQUIRED)
    message(STATUS "GTSAM found: ${GTSAM_VERSION}")
else()
    # Build GTSAM from source with minimal configuration
    # (Settings from conversation for stripped-down build)
    include(FetchContent)
    
    set(GTSAM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GTSAM_BUILD_EXAMPLES_ALWAYS OFF CACHE BOOL "" FORCE)
    set(GTSAM_BUILD_UNSTABLE OFF CACHE BOOL "" FORCE)
    set(GTSAM_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    set(GTSAM_ENABLE_BOOST_SERIALIZATION OFF CACHE BOOL "" FORCE)  # Critical size saver
    set(GTSAM_WITH_TBB OFF CACHE BOOL "" FORCE)  # Disable threading dependency
    set(GTSAM_USE_SYSTEM_EIGEN ON CACHE BOOL "" FORCE)
    set(GTSAM_BUILD_WITH_MARCH_NATIVE OFF CACHE BOOL "" FORCE)
    
    FetchContent_Declare(
        gtsam
        GIT_REPOSITORY https://github.com/borglab/gtsam.git
        GIT_TAG        4.2  # Stable release
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(gtsam)
    message(STATUS "GTSAM will be built from source (minimal config)")
endif()

# --- TEASER++ (Robust Registration) ---
if(USE_SYSTEM_TEASER)
    find_package(teaserpp QUIET)
    if(NOT teaserpp_FOUND)
        message(WARNING "TEASER++ not found in system, will attempt to fetch")
        set(USE_SYSTEM_TEASER OFF)
    else()
        message(STATUS "TEASER++ found")
    endif()
endif()

if(NOT USE_SYSTEM_TEASER)
    include(FetchContent)
    
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_TEASER_FPFH OFF CACHE BOOL "" FORCE)
    set(BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
    set(BUILD_DOC OFF CACHE BOOL "" FORCE)
    
    FetchContent_Declare(
        teaserpp
        GIT_REPOSITORY https://github.com/MIT-SPARK/TEASER-plusplus.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(teaserpp)
    message(STATUS "TEASER++ will be built from source")
endif()

# --- FBoW (Fast Bag of Words) ---
# Integrated directly in thirdparty/ to avoid build system conflicts
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/fbow-master/CMakeLists.txt")
    message(STATUS "Using bundled FBoW from thirdparty/")
    add_subdirectory(thirdparty/fbow-master)
else()
    message(FATAL_ERROR "FBoW source not found in thirdparty/fbow-master. Please run: \n"
            "  cd thirdparty && git clone https://github.com/rmsalinas/fbow.git")
endif()

# =============================================================================
# Source Files
# =============================================================================
set(SLAM_SOURCES
    src/core/slam_system.cpp
    src/core/config.cpp
    
    # Frontend (Feature Detection/Matching)
    src/frontend/feature_extractor.cpp
    src/frontend/feature_matcher.cpp
    src/frontend/frame.cpp
    src/frontend/keyframe.cpp
    
    # Visual-Inertial Odometry
    src/vio/imu_preintegration.cpp
    src/vio/vio_initializer.cpp
    src/vio/visual_frontend.cpp
    
    # Backend (Graph Optimization)
    src/backend/graph_optimizer.cpp
    src/backend/fixed_lag_smoother.cpp
    src/backend/marginalization.cpp
    
    # Loop Closure
    src/loop/loop_detector.cpp
    src/loop/place_recognition.cpp
    src/loop/geometric_verification.cpp
    
    # Mapping
    src/mapping/local_map.cpp
    src/mapping/keyframe_database.cpp
    src/mapping/covisibility_graph.cpp
    
    # Utilities
    src/utils/time_sync.cpp
    src/utils/calibration.cpp
    src/utils/triangulation.cpp
)

set(SLAM_HEADERS
    include/caai_slam/core/slam_system.hpp
    include/caai_slam/core/config.hpp
    include/caai_slam/core/types.hpp
    
    include/caai_slam/frontend/feature_extractor.hpp
    include/caai_slam/frontend/feature_matcher.hpp
    include/caai_slam/frontend/frame.hpp
    include/caai_slam/frontend/keyframe.hpp
    
    include/caai_slam/vio/imu_preintegration.hpp
    include/caai_slam/vio/vio_initializer.hpp
    include/caai_slam/vio/visual_frontend.hpp
    
    include/caai_slam/backend/graph_optimizer.hpp
    include/caai_slam/backend/fixed_lag_smoother.hpp
    include/caai_slam/backend/marginalization.hpp
    
    include/caai_slam/loop/loop_detector.hpp
    include/caai_slam/loop/place_recognition.hpp
    include/caai_slam/loop/geometric_verification.hpp
    
    include/caai_slam/mapping/local_map.hpp
    include/caai_slam/mapping/keyframe_database.hpp
    include/caai_slam/mapping/covisibility_graph.hpp
    
    include/caai_slam/utils/time_sync.hpp
    include/caai_slam/utils/calibration.hpp
    include/caai_slam/utils/triangulation.hpp
)

# =============================================================================
# Library Target
# =============================================================================
add_library(caai_slam SHARED ${SLAM_SOURCES} ${SLAM_HEADERS})

target_include_directories(caai_slam
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(caai_slam
    PUBLIC
        Eigen3::Eigen
        ${OpenCV_LIBS}
        gtsam
    PRIVATE
        teaser_registration
        fbow
)

# Set library properties
set_target_properties(caai_slam PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${SLAM_HEADERS}"
)

# =============================================================================
# Executable Targets
# =============================================================================
if(BUILD_EXAMPLES)
    # Main CAAI-SLAM executable
    add_executable(caai_slam_node
        src/main.cpp
    )
    target_link_libraries(caai_slam_node PRIVATE caai_slam)
    
    # Dataset evaluation tool
    add_executable(evaluate_dataset
        examples/evaluate_dataset.cpp
    )
    target_link_libraries(evaluate_dataset PRIVATE caai_slam)
    
    # Calibration tool
    add_executable(calibrate_camera_imu
        examples/calibrate_camera_imu.cpp
    )
    target_link_libraries(calibrate_camera_imu PRIVATE caai_slam)
endif()

# =============================================================================
# Testing
# =============================================================================
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    
    add_executable(test_feature_extraction tests/test_feature_extraction.cpp)
    target_link_libraries(test_feature_extraction PRIVATE caai_slam GTest::gtest_main)
    add_test(NAME FeatureExtraction COMMAND test_feature_extraction)
    
    add_executable(test_imu_preintegration tests/test_imu_preintegration.cpp)
    target_link_libraries(test_imu_preintegration PRIVATE caai_slam GTest::gtest_main)
    add_test(NAME IMUPreintegration COMMAND test_imu_preintegration)
    
    add_executable(test_graph_optimizer tests/test_graph_optimizer.cpp)
    target_link_libraries(test_graph_optimizer PRIVATE caai_slam GTest::gtest_main)
    add_test(NAME GraphOptimizer COMMAND test_graph_optimizer)
endif()

# =============================================================================
# Installation
# =============================================================================
include(GNUInstallDirs)

install(TARGETS caai_slam
    EXPORT caai_slam-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/caai_slam
)

if(BUILD_EXAMPLES)
    install(TARGETS caai_slam_node evaluate_dataset calibrate_camera_imu
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

install(EXPORT caai_slam-targets
    FILE caai_slam-targets.cmake
    NAMESPACE caai_slam::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/caai_slam
)

# =============================================================================
# Package Configuration
# =============================================================================
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/caai_slam-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/caai_slam-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/caai_slam
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/caai_slam-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/caai_slam-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/caai_slam-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/caai_slam
)

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "========== CAAI_SLAM Configuration ==========")
message(STATUS "  Version:          ${PROJECT_VERSION}")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  Platform:         ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Architecture:     ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  SIMD flags:       ${SIMD_FLAGS}")
message(STATUS "  OpenCV version:   ${OpenCV_VERSION}")
message(STATUS "  System GTSAM:     ${USE_SYSTEM_GTSAM}")
message(STATUS "  System TEASER++:  ${USE_SYSTEM_TEASER}")
message(STATUS "  Build tests:      ${BUILD_TESTS}")
message(STATUS "  Build examples:   ${BUILD_EXAMPLES}")
message(STATUS "  Minimal build:    ${MINIMAL_BUILD}")
message(STATUS "=================================================")
message(STATUS "")