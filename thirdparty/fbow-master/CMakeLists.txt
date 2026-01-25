cmake_minimum_required(VERSION 3.16)
project(fbow_bundled)

find_package(OpenCV REQUIRED)

# 1. Define the library
add_library(fbow SHARED
    src/fbow.cpp
    src/vocabulary_creator.cpp
)

# 2. Configuration
target_compile_features(fbow PUBLIC cxx_std_17)

# 3. Includes
# BUILD_INTERFACE: Used when building the project
# INSTALL_INTERFACE: Used by consumers of the installed project
target_include_directories(fbow PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fbow>
)

target_link_libraries(fbow PUBLIC ${OpenCV_LIBS})

if(WIN32)
    target_compile_definitions(fbow PRIVATE fbow_EXPORTS)
endif()

# 4. INSTALLATION - Critical Fix for "EXPORT" error
include(GNUInstallDirs)

# Install the library binaries and add them to the 'caai_slam-targets' export set
install(TARGETS fbow
    EXPORT caai_slam-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fbow
)

# Install the headers so 'caai_slam' headers can find them
install(FILES 
    src/fbow.h 
    src/vocabulary_creator.h 
    src/cpu.h 
    src/fbow_exports.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fbow
)