#include "caai_slam/app/slam_app.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include <iostream>
#include <cmath>

namespace caai_slam {

    // =========================================================================
    // Helper: manage a single OpenGL texture re-uploaded each frame
    // =========================================================================
    struct gl_texture {
        GLuint id = 0;
        int width = 0;
        int height = 0;

        void release() {
            if (id) { glDeleteTextures(1, &id); id = 0; }
            width = height = 0;
        }

        /// Upload a cv::Mat (GRAY, BGR, or BGRA → RGBA for ImGui).
        void upload(const cv::Mat& src) {
            if (src.empty()) return;

            cv::Mat rgba;
            if (src.channels() == 1)
                cv::cvtColor(src, rgba, cv::COLOR_GRAY2RGBA);
            else if (src.channels() == 3)
                cv::cvtColor(src, rgba, cv::COLOR_BGR2RGBA);
            else if (src.channels() == 4)
                cv::cvtColor(src, rgba, cv::COLOR_BGRA2RGBA);
            else
                return;

            const bool size_changed = (rgba.cols != width || rgba.rows != height);

            if (!id)
                glGenTextures(1, &id);

            glBindTexture(GL_TEXTURE_2D, id);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

            if (size_changed) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                             rgba.cols, rgba.rows, 0,
                             GL_RGBA, GL_UNSIGNED_BYTE, rgba.data);
                width  = rgba.cols;
                height = rgba.rows;
            } else {
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                rgba.cols, rgba.rows,
                                GL_RGBA, GL_UNSIGNED_BYTE, rgba.data);
            }
        }

        ImTextureID imgui_id() const {
            return reinterpret_cast<ImTextureID>(static_cast<intptr_t>(id));
        }
    };

    /**
     * @brief Linux-specific SLAM application with ImGui visualization
     */
    class linux_slam_app : public slam_app {
    public:
        linux_slam_app(const std::string& config_path) : slam_app(config_path) {}

        bool run(const std::string& dataset_root, const std::string& output_traj = "") {
            // 1. Initialize SLAM app (common)
            if (!initialize(dataset_root))
                return false;

            // 2. Initialize GLFW window
            if (!glfwInit()) {
                std::cerr << "[Linux] GLFW initialization failed" << std::endl;
                return false;
            }

            const char* glsl_version = "#version 150";
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
            glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

            GLFWwindow* window = glfwCreateWindow(1280, 720, "CAAI-SLAM Visualizer (Linux)", nullptr, nullptr);
            if (!window) {
                std::cerr << "[Linux] Failed to create GLFW window" << std::endl;
                glfwTerminate();
                return false;
            }

            glfwMakeContextCurrent(window);
            glfwSwapInterval(1); // Enable vsync

            // 3. Setup ImGui
            IMGUI_CHECKVERSION();
            ImGui::CreateContext();
            ImGuiIO& io = ImGui::GetIO();
            io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

            ImGui::StyleColorsDark();

            ImGui_ImplGlfw_InitForOpenGL(window, true);
            ImGui_ImplOpenGL3_Init(glsl_version);

            std::cout << "[Linux] Window and ImGui initialized" << std::endl;

            // 4. Main loop
            bool should_close = false;
            bool is_paused = false;
            bool auto_play = true;

            gl_texture frame_tex; // Persistent texture for the camera frame

            while (!glfwWindowShouldClose(window) && !should_close) {
                glfwPollEvents();

                // Start ImGui frame
                ImGui_ImplOpenGL3_NewFrame();
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();

                // Process next frame if not paused
                if (auto_play && !is_paused && !is_finished()) {
                    if (!process_next_frame()) {
                        // Dataset exhausted
                        auto_play = false;
                    }
                }

                // Render UI
                render_ui(should_close, is_paused, auto_play, frame_tex);

                // Rendering
                ImGui::Render();

                int display_w, display_h;
                glfwGetFramebufferSize(window, &display_w, &display_h);
                glViewport(0, 0, display_w, display_h);
                glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

                glfwSwapBuffers(window);
            }

            // 5. Cleanup
            frame_tex.release();

            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();

            glfwDestroyWindow(window);
            glfwTerminate();

            // 6. Save trajectory if requested
            if (!output_traj.empty())
                save_trajectory(output_traj);

            std::cout << "[Linux] Application closed cleanly" << std::endl;
            return true;
        }

    private:
        /// Draw keypoints onto the image and return a BGR copy for display.
        static cv::Mat draw_keypoints_overlay(const cv::Mat& gray,
                                               const std::vector<cv::KeyPoint>& kps) {
            cv::Mat display;
            if (gray.channels() == 1)
                cv::cvtColor(gray, display, cv::COLOR_GRAY2BGR);
            else
                display = gray.clone();

            if (!kps.empty())
                cv::drawKeypoints(display, kps, display,
                                  cv::Scalar(0, 255, 0),              // green
                                  cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            return display;
        }

        void render_ui(bool& should_close, bool& is_paused, bool& auto_play,
                       gl_texture& frame_tex) {
            const auto vis = get_vis_state();
            const auto perf = get_perf_stats();
            const double progress = get_progress();

            // =================================================================
            // Camera Frame window
            // =================================================================
            if (!vis.current_image.empty()) {
                cv::Mat display = draw_keypoints_overlay(vis.current_image,
                                                          vis.keypoints);
                frame_tex.upload(display);
            }

            if (frame_tex.id) {
                ImGui::SetNextWindowPos(ImVec2(420, 10), ImGuiCond_FirstUseEver);
                ImGui::SetNextWindowSize(
                    ImVec2(static_cast<float>(frame_tex.width) + 20,
                           static_cast<float>(frame_tex.height) + 60),
                    ImGuiCond_FirstUseEver);

                if (ImGui::Begin("Camera Frame", nullptr,
                                 ImGuiWindowFlags_NoScrollbar |
                                 ImGuiWindowFlags_NoScrollWithMouse)) {
                    // Fit image to available region while keeping aspect ratio
                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    float scale = std::min(avail.x / frame_tex.width,
                                           avail.y / frame_tex.height);
                    if (scale <= 0.0f) scale = 1.0f;
                    ImVec2 img_sz(frame_tex.width * scale,
                                  frame_tex.height * scale);

                    ImGui::Image(frame_tex.imgui_id(), img_sz);
                }
                ImGui::End();
            }

            // =================================================================
            // Status panel (left side)
            // =================================================================
            ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(400, 600), ImGuiCond_FirstUseEver);

            if (ImGui::Begin("CAAI-SLAM Status", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
                ImGui::SeparatorText("System Status");
                ImGui::Text("Status: %s", vis.status_message.c_str());

                const char* status_str = "UNKNOWN";
                switch (vis.status) {
                    case system_status::NOT_INITIALIZED:
                        status_str = "NOT_INITIALIZED"; break;
                    case system_status::INITIALIZING:
                        status_str = "INITIALIZING"; break;
                    case system_status::TRACKING:
                        status_str = "TRACKING"; break;
                }
                ImGui::Text("Mode: %s", status_str);

                ImGui::SeparatorText("Frame Processing");
                ImGui::Text("Current Frame: %zu / %zu", get_current_frame(), get_total_frames());
                ImGui::ProgressBar(static_cast<float>(progress), ImVec2(-1, 0), "");
                ImGui::Text("FPS: %.1f", perf.average_fps);
                ImGui::Text("Frame Time: %.2f ms", perf.total_time_ms);

                ImGui::SeparatorText("Mapping");
                ImGui::Text("Keyframes: %zu", vis.total_keyframes);
                ImGui::Text("Map Points: %zu", vis.total_map_points);
                ImGui::Text("Tracking Quality: %.1f%%", vis.tracking_quality * 100.0f);

                ImGui::SeparatorText("Pose Estimation");
                ImGui::Text("Position: (%.3f, %.3f, %.3f)",
                           vis.current_pose.translation.x(),
                           vis.current_pose.translation.y(),
                           vis.current_pose.translation.z());
                const auto q = quat(vis.current_pose.rotation);
                ImGui::Text("Orientation: (%.3f, %.3f, %.3f, %.3f)",
                           q.x(), q.y(), q.z(), q.w());

                ImGui::SeparatorText("Controls");
                ImGui::Checkbox("Auto Play", &auto_play);
                ImGui::SameLine();
                if (ImGui::Button(is_paused ? "Resume" : "Pause"))
                    is_paused = !is_paused;
                ImGui::SameLine();
                if (ImGui::Button("Reset")) {
                    reset_slam();
                    is_paused = true;
                    auto_play = false;
                }
                ImGui::SameLine();
                if (ImGui::Button("Exit"))
                    should_close = true;

                ImGui::End();
            }

            // =================================================================
            // Info window
            // =================================================================
            ImGui::SetNextWindowPos(
                ImVec2(420, frame_tex.id
                    ? static_cast<float>(frame_tex.height) + 90.0f : 10.0f),
                ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);

            if (ImGui::Begin("Information", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
                ImGui::SeparatorText("About");
                ImGui::TextWrapped(
                    "CAAI-SLAM: Visual-Inertial SLAM System\n"
                    "Center for Connected Autonomy and AI\n"
                    "Florida Atlantic University\n\n"
                    "Version: 0.1.0\n"
                    "Backend: GTSAM + FixedLagSmoother\n"
                    "Loop Closure: FBoW + TEASER++\n"
                );

                ImGui::SeparatorText("Platform");
                ImGui::Text("OS: Linux");
                ImGui::Text("Renderer: OpenGL 3.2 + ImGui");
                ImGui::Text("Windowing: GLFW3");

                ImGui::SeparatorText("Controls");
                ImGui::BulletText("Auto Play: Continuous processing");
                ImGui::BulletText("Pause/Resume: Manual frame control");
                ImGui::BulletText("Reset: Restart SLAM system");
                ImGui::BulletText("Exit: Save trajectory and quit");

                ImGui::End();
            }
        }
    };

} // namespace caai_slam

/**
 * @brief Linux entry point
 */
int main(int argc, char* argv[]) {
    std::string config_path = "config.yaml";
    std::string dataset_root = "";
    std::string output_trajectory = "trajectory.txt";

    // Parse arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_root> [config.yaml] [output.txt]" << std::endl;
        std::cerr << "  dataset_root: Path to EuRoC dataset" << std::endl;
        std::cerr << "  config.yaml:  Path to SLAM config (default: config.yaml)" << std::endl;
        std::cerr << "  output.txt:   Path to save trajectory (default: trajectory.txt)" << std::endl;
        return 1;
    }

    dataset_root = argv[1];
    if (argc > 2)
        config_path = argv[2];
    if (argc > 3)
        output_trajectory = argv[3];

    std::cout << "\n=== CAAI-SLAM Visualizer (Linux) ===" << std::endl;
    std::cout << "Config:    " << config_path << std::endl;
    std::cout << "Dataset:   " << dataset_root << std::endl;
    std::cout << "Output:    " << output_trajectory << std::endl;
    std::cout << "====================================\n" << std::endl;

    try {
        caai_slam::linux_slam_app app(config_path);
        if (!app.run(dataset_root, output_trajectory)) {
            std::cerr << "Application failed to run" << std::endl;
            return 1;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}