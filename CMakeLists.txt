cmake_minimum_required(VERSION 3.16)
project(CAAI_SLAM VERSION 0.1.0 LANGUAGES C CXX)

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
option(BUILD_VISUALIZER "Build the ImGui/OpenGL visualizer (Win32 ARM64 optimized)" ON)
option(USE_SYSTEM_GTSAM "Use system-installed GTSAM instead of building from source" ON)
option(USE_SYSTEM_TEASER "Use system-installed TEASER++ instead of building from source" ON)
option(ENABLE_SIMD "Enable SIMD optimizations (SSE4/AVX on x86, NEON on ARM)" ON)
option(MINIMAL_BUILD "Strip unnecessary components for smaller binary" OFF)
option(USE_VCPKG "Use vcpkg for dependency management (recommended for Windows)" OFF)
option(BUILD_TOOLS "Build utility tools (vocab generator, etc.)" ON)

# =============================================================================
# vcpkg Integration (Optional)
# =============================================================================
if(USE_VCPKG)
    if(DEFINED CMAKE_TOOLCHAIN_FILE)
        message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
    else()
        message(WARNING "USE_VCPKG=ON but CMAKE_TOOLCHAIN_FILE not set. Set via: -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake")
    endif()
endif()

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
    
    # Win32 API definitions
    add_definitions(-DWIN32_LEAN_AND_MEAN)
    add_definitions(-D_WINDOWS)
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
    message(STATUS "Architecture: ARM64 (e.g., Snapdragon X Elite)")
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

# SIMD flags based on architecture and compiler
if(ENABLE_SIMD)
    if(ARCH_ARM64)
        if(MSVC)
            # MSVC ARM64: NEON is auto-enabled; /arch is semantic only
            message(STATUS "SIMD: NEON (ARM64, MSVC auto-enabled)")
            add_definitions(-DUSE_NEON)
            # No additional /arch flag needed
        else()
            # GCC/Clang ARM64
            set(SIMD_FLAGS "-march=armv8-a")
            add_definitions(-DUSE_NEON)
            message(STATUS "SIMD: NEON (ARM64, explicit march)")
        endif()
    elseif(ARCH_ARM32)
        if(MSVC)
            message(STATUS "SIMD: NEON (ARM32, MSVC)")
            add_definitions(-DUSE_NEON)
        else()
            check_cxx_compiler_flag("-mfpu=neon" COMPILER_SUPPORTS_NEON)
            if(COMPILER_SUPPORTS_NEON)
                set(SIMD_FLAGS "-mfpu=neon -mfloat-abi=hard")
                add_definitions(-DUSE_NEON)
                message(STATUS "SIMD: NEON (ARM32)")
            endif()
        endif()
    elseif(ARCH_X64 OR ARCH_X86)
        if(MSVC)
            # MSVC uses /arch flags
            check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
            if(COMPILER_SUPPORTS_AVX2)
                set(SIMD_FLAGS "/arch:AVX2")
                add_definitions(-DUSE_AVX2)
                message(STATUS "SIMD: AVX2 (MSVC)")
            else()
                set(SIMD_FLAGS "/arch:SSE4.2")
                add_definitions(-DUSE_SSE4)
                message(STATUS "SIMD: SSE4.2 (MSVC)")
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
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /ZI /Od")
    
    # Disable specific warnings
    add_definitions(/wd4267 /wd4244 /wd4305)  # Conversion warnings
    add_definitions(/wd4996)  # Deprecation warnings
    
    # Character set
    add_definitions(/DUNICODE /D_UNICODE)
else()
    # GCC/Clang flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra ${SIMD_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -Os -DNDEBUG")
    
    if(MINIMAL_BUILD)
        # Flags for minimal binary size
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

# Eigen alignment for NEON/SSE (critical for ARM64)
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
    include(FetchContent)
    
    set(GTSAM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GTSAM_BUILD_EXAMPLES_ALWAYS OFF CACHE BOOL "" FORCE)
    set(GTSAM_BUILD_UNSTABLE ON CACHE BOOL "" FORCE)
    set(GTSAM_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    set(GTSAM_ENABLE_BOOST_SERIALIZATION OFF CACHE BOOL "" FORCE)
    set(GTSAM_WITH_TBB OFF CACHE BOOL "" FORCE)
    set(GTSAM_USE_SYSTEM_EIGEN ON CACHE BOOL "" FORCE)
    set(GTSAM_BUILD_WITH_MARCH_NATIVE OFF CACHE BOOL "" FORCE)
    
    FetchContent_Declare(
        gtsam
        GIT_REPOSITORY https://github.com/borglab/gtsam.git
        GIT_TAG        4.2
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(gtsam)
    message(STATUS "GTSAM will be built from source")
endif()

# --- TEASER++ (Robust Registration) ---
if(USE_SYSTEM_TEASER)
    find_package(teaserpp QUIET)
    if(NOT teaserpp_FOUND)
        message(WARNING "TEASER++ not found, will attempt to fetch")
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
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/fbow-master/CMakeLists.txt")
    message(STATUS "Using bundled FBoW from thirdparty/")
    add_subdirectory(thirdparty/fbow-master)
else()
    message(FATAL_ERROR "FBoW source not found. Run: cd thirdparty && git clone https://github.com/rmsalinas/fbow.git fbow-master")
endif()

# --- ImGui + OpenGL (For Visualizer) ---
if(BUILD_VISUALIZER)
    find_package(OpenGL REQUIRED)
    message(STATUS "OpenGL found")
    
    # Check for glfw3 or SDL2
    find_package(glfw3 QUIET)
    if(NOT glfw3_FOUND)
        message(STATUS "glfw3 not found, will use OpenCV for window (via ImGui backend)")
        set(USE_GLFW FALSE)
    else()
        message(STATUS "glfw3 found")
        set(USE_GLFW TRUE)
    endif()
    
    # Check for ImGui
    find_package(imgui QUIET)
    if(NOT imgui_FOUND)
        message(STATUS "ImGui not found, fetching from source...")
        include(FetchContent)
        
        FetchContent_Declare(
            imgui
            GIT_REPOSITORY https://github.com/ocornut/imgui.git
            GIT_TAG        v1.89.9
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(imgui)
        
        # Create ImGui library
        # 1. Define the common sources for all platforms
        set(IMGUI_SOURCES
            ${imgui_SOURCE_DIR}/imgui.cpp
            ${imgui_SOURCE_DIR}/imgui_demo.cpp
            ${imgui_SOURCE_DIR}/imgui_draw.cpp
            ${imgui_SOURCE_DIR}/imgui_widgets.cpp
            ${imgui_SOURCE_DIR}/imgui_tables.cpp
            ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
        )

        if(USE_GLFW)
            list(APPEND IMGUI_SOURCES ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp)
        endif()

        # 2. Conditionally append the Windows backend ONLY if on Windows
        if(WIN32)
            list(APPEND IMGUI_SOURCES ${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp)
        endif()

        # 3. Create the library using the constructed source list
        add_library(imgui STATIC ${IMGUI_SOURCES})

        target_include_directories(imgui PUBLIC
            ${imgui_SOURCE_DIR}
            ${imgui_SOURCE_DIR}/backends
        )

        target_link_libraries(imgui PUBLIC OpenGL::OpenGL)
        
        set(IMGUI_INCLUDE_DIRS ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
        set(imgui_LIBRARIES imgui)
    else()
        message(STATUS "ImGui found: ${imgui_DIR}")
    endif()
endif()

# =============================================================================
# Tools
# =============================================================================
if(BUILD_TOOLS)
    # AKAZE Vocabulary Generator for FBoW
    add_executable(generate_akaze_vocab
        tools/generate_akaze_vocab.cpp
    )
    target_include_directories(generate_akaze_vocab PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/fbow-master/src
    )
    target_link_libraries(generate_akaze_vocab PRIVATE
        ${OpenCV_LIBS}
        fbow
    )
    target_compile_features(generate_akaze_vocab PRIVATE cxx_std_17)
    # Link stdc++fs on older GCC (< 9)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        target_link_libraries(generate_akaze_vocab PRIVATE stdc++fs)
    endif()
    message(STATUS "Building tools: generate_akaze_vocab")
endif()

if(BUILD_TOOLS)
    install(TARGETS generate_akaze_vocab
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# =============================================================================
# Source Files
# =============================================================================

set(SLAM_SOURCES
    src/core/slam_system.cpp
    src/core/config.cpp
    
    # Frontend
    src/frontend/feature_extractor.cpp
    src/frontend/feature_matcher.cpp
    src/frontend/frame.cpp
    src/frontend/keyframe.cpp
    
    # VIO
    src/vio/imu_preintegration.cpp
    src/vio/vio_initializer.cpp
    src/vio/visual_frontend.cpp
    
    # Backend
    src/backend/graph_optimizer.cpp
    src/backend/fixed_lag_smoother.cpp
    src/backend/marginalization.cpp
    
    # Loop Closure
    src/loop/common.cpp
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
    src/utils/euroc_loader.cpp

    src/app/slam_app.cpp
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
    
    include/caai_slam/loop/common.hpp
    include/caai_slam/loop/loop_detector.hpp
    include/caai_slam/loop/place_recognition.hpp
    include/caai_slam/loop/geometric_verification.hpp
    
    include/caai_slam/mapping/local_map.hpp
    include/caai_slam/mapping/keyframe_database.hpp
    include/caai_slam/mapping/covisibility_graph.hpp
    
    include/caai_slam/utils/time_sync.hpp
    include/caai_slam/utils/calibration.hpp
    include/caai_slam/utils/triangulation.hpp
    include/caai_slam/utils/euroc_loader.hpp

    include/caai_slam/app/slam_app.hpp
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
        gtsam_unstable
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
    # Main CAAI-SLAM node (dataset playback + visualization)
    if(BUILD_VISUALIZER)
        # Find GLAD (OpenGL loader)
        find_package(glad QUIET)
        if(NOT glad_FOUND)
            message(STATUS "GLAD not found, fetching from source...")
            include(FetchContent)
            
            FetchContent_Declare(
                glad
                GIT_REPOSITORY https://github.com/Dav1dde/glad.git
                GIT_TAG        v0.1.36
                GIT_SHALLOW    TRUE
            )
            FetchContent_MakeAvailable(glad)
            
            set(glad_LIBRARIES glad)
        else()
            message(STATUS "GLAD found")
        endif()
        
        # Find GLM (Math library)
        find_package(glm QUIET)
        if(NOT glm_FOUND)
            message(STATUS "GLM not found, fetching from source...")
            include(FetchContent)
            
            FetchContent_Declare(
                glm
                GIT_REPOSITORY https://github.com/g-truc/glm.git
                GIT_TAG        0.9.9.8
                GIT_SHALLOW    TRUE
            )
            FetchContent_MakeAvailable(glm)
            
            set(GLM_INCLUDE_DIRS ${glm_SOURCE_DIR})
        else()
            message(STATUS "GLM found")
        endif()

        # ---------------------------------------------------------------------
        # Platform-Specific Main Selection
        # ---------------------------------------------------------------------
        if(PLATFORM_WINDOWS)
            if(ARCH_ARM64)
                message(STATUS "Selecting Main: Win32 ARM64")
                set(VISUALIZER_MAIN "src/platforms/win32/main_arm64.cpp")
            elseif(ARCH_X64)
                message(STATUS "Selecting Main: Win32 x64")
                set(VISUALIZER_MAIN "src/platforms/win32/main_x64.cpp")
            else()
                # Fallback for Win32 x86 or unknown
                message(WARNING "Unknown Windows Arch, using generic x64 main")
                set(VISUALIZER_MAIN "src/platforms/win32/main_x64.cpp")
            endif()
            
        elseif(ARDUINO) 
            # Note: You will likely set 'ARDUINO' in your toolchain file 
            # for the Uno Q later.
            message(STATUS "Selecting Main: Arduino Uno Q")
            set(VISUALIZER_MAIN "src/platforms/arduino/main_uno_q.cpp")
            
        else()
            # Default Linux/Unix
            message(STATUS "Selecting Main: Generic Linux")
            set(VISUALIZER_MAIN "src/platforms/linux/main.cpp")
        endif()
        
        add_executable(caai_slam_visualizer
            ${VISUALIZER_MAIN}
        )
        target_include_directories(caai_slam_visualizer PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${IMGUI_INCLUDE_DIRS}
            ${GLM_INCLUDE_DIRS}
        )
        target_link_libraries(caai_slam_visualizer PRIVATE
            caai_slam
            OpenGL::OpenGL
            ${imgui_LIBRARIES}
            ${glad_LIBRARIES}
        )
        
        # Link glfw if available
        if(USE_GLFW)
            target_link_libraries(caai_slam_visualizer PRIVATE glfw)
        else()
            # If GLFW not found, try to find it
            find_package(glfw3 REQUIRED)
            target_link_libraries(caai_slam_visualizer PRIVATE glfw)
        endif()
    else()
        message(STATUS "BUILD_VISUALIZER=OFF, skipping visualizer executable")
    endif()
    
    # Evaluation tool
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
    if(BUILD_VISUALIZER)
        install(TARGETS caai_slam_visualizer
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif()
    install(TARGETS evaluate_dataset calibrate_camera_imu
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
message(STATUS "  Version:              ${PROJECT_VERSION}")
message(STATUS "  Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "  Platform:             ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Architecture:         ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Compiler:             ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  SIMD:                 ${SIMD_FLAGS}")
message(STATUS "  OpenCV version:       ${OpenCV_VERSION}")
message(STATUS "  GTSAM (system):       ${USE_SYSTEM_GTSAM}")
message(STATUS "  TEASER++ (system):    ${USE_SYSTEM_TEASER}")
message(STATUS "  Build tests:          ${BUILD_TESTS}")
message(STATUS "  Build examples:       ${BUILD_EXAMPLES}")
message(STATUS "  Build visualizer:     ${BUILD_VISUALIZER}")
message(STATUS "  Use vcpkg:            ${USE_VCPKG}")
message(STATUS "  Minimal build:        ${MINIMAL_BUILD}")
message(STATUS "==============================================")
message(STATUS "")