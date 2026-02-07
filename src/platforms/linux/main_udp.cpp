#include "caai_slam/app/slam_app.hpp"

#include "caai_slam/mapping/local_map.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <thread>
#include <atomic>

// Linux Socket Headers
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

namespace caai_slam {

    // =========================================================================
    // Helper: manage a single OpenGL texture
    // =========================================================================
    struct gl_texture {
        GLuint id = 0;
        int width = 0;
        int height = 0;

        void release() {
            if (id) { glDeleteTextures(1, &id); id = 0; }
            width = height = 0;
        }

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
     * @brief TCP Receiver that matches the Python 'Length-Prefixed' Protocol
     */
    class tcp_receiver {
    private:
        int server_fd = -1;
        int client_fd = -1;
        std::atomic<bool> connected{false};
        std::thread receiver_thread;
        std::atomic<bool> running{true};
        
        // Shared Data
        std::mutex data_mutex;
        cv::Mat latest_frame;
        bool new_frame_available = false;

    public:
        tcp_receiver() = default;
        ~tcp_receiver() { stop(); }

        bool start(int port) {
            // 1. Create Socket
            server_fd = socket(AF_INET, SOCK_STREAM, 0);
            if (server_fd == 0) {
                std::cerr << "[TCP] Socket creation failed" << std::endl;
                return false;
            }

            // 2. Bind options (Reuse Address)
            int opt = 1;
            setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));

            struct sockaddr_in address;
            address.sin_family = AF_INET;
            address.sin_addr.s_addr = INADDR_ANY; // Listen on 0.0.0.0
            address.sin_port = htons(port);

            if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
                std::cerr << "[TCP] Bind failed on port " << port << std::endl;
                return false;
            }

            // 3. Listen
            if (listen(server_fd, 1) < 0) {
                std::cerr << "[TCP] Listen failed" << std::endl;
                return false;
            }

            std::cout << "[TCP] Listening on port " << port << "..." << std::endl;

            // 4. Start Thread
            receiver_thread = std::thread(&tcp_receiver::listen_loop, this);
            return true;
        }

        void stop() {
            running = false;
            if (server_fd > 0) close(server_fd);
            if (client_fd > 0) close(client_fd);
            if (receiver_thread.joinable()) receiver_thread.join();
        }

        bool get_latest_frame(cv::Mat& frame) {
            std::lock_guard<std::mutex> lock(data_mutex);
            if (new_frame_available) {
                frame = latest_frame.clone();
                new_frame_available = false;
                return true;
            }
            return false;
        }

        bool is_connected() const { return connected; }

    private:
        // Helper to read exactly N bytes (handling TCP fragmentation)
        bool read_n_bytes(int fd, void* buffer, size_t n) {
            size_t total_read = 0;
            char* ptr = static_cast<char*>(buffer);
            while (total_read < n) {
                ssize_t bytes = recv(fd, ptr + total_read, n - total_read, 0);
                if (bytes <= 0) return false; // Error or Closed
                total_read += bytes;
            }
            return true;
        }

        void listen_loop() {
            while (running) {
                if (!connected) {
                    struct sockaddr_in client_addr;
                    socklen_t addrlen = sizeof(client_addr);
                    
                    // Accept (Blocking call - but server socket is closed on stop() to break this)
                    client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &addrlen);
                    if (client_fd < 0) {
                        if (!running) break;
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        continue;
                    }

                    std::cout << "[TCP] Client connected: " << inet_ntoa(client_addr.sin_addr) << std::endl;
                    connected = true;
                    
                    // Process Connection
                    while (running && connected) {
                        // 1. Read 4-byte header (Network Byte Order / Big Endian)
                        uint32_t net_len = 0;
                        if (!read_n_bytes(client_fd, &net_len, 4)) {
                            std::cerr << "[TCP] Client disconnected (Header)" << std::endl;
                            close(client_fd);
                            connected = false;
                            break;
                        }

                        uint32_t frame_len = ntohl(net_len);

                        // Safety check on size (e.g., max 10MB)
                        if (frame_len == 0 || frame_len > 10 * 1024 * 1024) {
                            std::cerr << "[TCP] Invalid frame size: " << frame_len << std::endl;
                            close(client_fd);
                            connected = false;
                            break;
                        }

                        // 2. Read Frame Data
                        std::vector<uchar> buffer(frame_len);
                        if (!read_n_bytes(client_fd, buffer.data(), frame_len)) {
                            std::cerr << "[TCP] Client disconnected (Body)" << std::endl;
                            close(client_fd);
                            connected = false;
                            break;
                        }

                        // 3. Decode
                        cv::Mat decoded = cv::imdecode(buffer, cv::IMREAD_COLOR);
                        if (!decoded.empty()) {
                            std::lock_guard<std::mutex> lock(data_mutex);
                            // Convert BGR (OpenCV) to Grayscale for SLAM immediately or later
                            // Keeping it raw here for visualization flexibility
                            latest_frame = decoded;
                            new_frame_available = true;
                        }
                    }
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
        }
    };

    /**
     * @brief Linux-specific SLAM application for Custom TCP Streams
     */
    class linux_tcp_slam_app : public slam_app {
        tcp_receiver _receiver;

    public:
        linux_tcp_slam_app(const std::string& config_path) : slam_app(config_path) {}
        ~linux_tcp_slam_app() { _receiver.stop(); }

        bool run_server(int port) {
            // 1. Init Window
            if (!glfwInit()) return false;
            
            // GL 3.2 + GLSL 150
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
            glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

            GLFWwindow* window = glfwCreateWindow(1280, 720, "CAAI-SLAM TCP Receiver", nullptr, nullptr);
            if (!window) { glfwTerminate(); return false; }

            glfwMakeContextCurrent(window);
            glfwSwapInterval(1);

            IMGUI_CHECKVERSION();
            ImGui::CreateContext();
            ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
            ImGui::StyleColorsDark();
            ImGui_ImplGlfw_InitForOpenGL(window, true);
            ImGui_ImplOpenGL3_Init("#version 150");

            // 2. Start TCP Server Thread
            if (!_receiver.start(port)) {
                std::cerr << "[App] Failed to start TCP server on port " << port << std::endl;
                return false;
            }

            // 3. Main Loop
            bool should_close = false;
            bool is_paused = false;
            gl_texture frame_tex;
            cv::Ptr<cv::AKAZE> viz_akaze = cv::AKAZE::create();
            
            double last_process_time = 0.0;
            const double imu_period = 1.0 / 200.0;

            vis_state.status_message = "Listening for incoming connection...";

            while (!glfwWindowShouldClose(window) && !should_close) {
                glfwPollEvents();

                // Check connection status
                if (!_receiver.is_connected()) {
                    vis_state.status_message = "Waiting for sender...";
                }

                // Process New Frames
                cv::Mat frame;
                if (!is_paused && _receiver.get_latest_frame(frame)) {
                    // Update Time
                    auto now_sys = std::chrono::system_clock::now();
                    double current_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                        now_sys.time_since_epoch()).count();

                    if (last_process_time == 0.0) last_process_time = current_time - 0.033;

                    // IMU Simulation (fill gaps)
                    double t_sim = last_process_time;
                    while (t_sim < current_time) {
                        t_sim += imu_period;
                        if (t_sim > current_time) break;
                        imu_measurement imu;
                        imu._timestamp = t_sim;
                        imu.linear_acceleration = vec3(0.0, 0.0, 9.81);
                        imu.angular_velocity = vec3(0.0, 0.0, 0.0);
                        _slam_system.process_imu(imu);
                    }

                    // Process Image
                    cv::Mat gray;
                    if (frame.channels() == 3) cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                    else if (frame.channels() == 4) cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
                    else gray = frame.clone();

                    _slam_system.process_image(gray, current_time);
                    last_process_time = current_time;

                    // Update Vis State
                    vis_state.current_image = frame.clone();
                    vis_state.current_pose = _slam_system.get_current_pose();
                    vis_state.status = _slam_system.get_status();
                    
                    std::vector<cv::KeyPoint> kps;
                    viz_akaze->detect(gray, kps);
                    vis_state.keypoints = std::move(kps);

                    auto lmap = _slam_system.get_local_map();
                    if (lmap) {
                        vis_state.total_keyframes = lmap->num_keyframes();
                        vis_state.total_map_points = lmap->num_map_points();
                    }

                    if (vis_state.status == system_status::TRACKING)
                         vis_state.status_message = "Tracking";
                }

                // Render
                ImGui_ImplOpenGL3_NewFrame();
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();
                
                render_ui(should_close, is_paused, frame_tex);

                ImGui::Render();
                int dw, dh;
                glfwGetFramebufferSize(window, &dw, &dh);
                glViewport(0, 0, dw, dh);
                glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT);
                ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
                glfwSwapBuffers(window);
            }

            frame_tex.release();
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
            glfwDestroyWindow(window);
            glfwTerminate();
            return true;
        }

    private:
        static cv::Mat draw_keypoints_overlay(const cv::Mat& gray, const std::vector<cv::KeyPoint>& kps) {
            cv::Mat display;
            if (gray.channels() == 1) cv::cvtColor(gray, display, cv::COLOR_GRAY2BGR);
            else display = gray.clone();
            
            if (!kps.empty())
                cv::drawKeypoints(display, kps, display, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            return display;
        }

        void render_ui(bool& should_close, bool& is_paused, gl_texture& frame_tex) {
            const auto vis = get_vis_state();
            
            if (!vis.current_image.empty()) {
                cv::Mat display = draw_keypoints_overlay(vis.current_image, vis.keypoints);
                frame_tex.upload(display);
            }

            if (frame_tex.id) {
                ImGui::SetNextWindowPos(ImVec2(420, 10), ImGuiCond_FirstUseEver);
                ImGui::SetNextWindowSize(ImVec2(frame_tex.width + 20.0f, frame_tex.height + 60.0f), ImGuiCond_FirstUseEver);
                if (ImGui::Begin("Remote Stream", nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    float scale = std::min(avail.x / frame_tex.width, avail.y / frame_tex.height);
                    if (scale <= 0.0f) scale = 1.0f;
                    ImGui::Image(frame_tex.imgui_id(), ImVec2(frame_tex.width * scale, frame_tex.height * scale));
                }
                ImGui::End();
            }

            ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Control Panel", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
                ImGui::Text("Status: %s", vis.status_message.c_str());
                ImGui::Text("Keyframes: %zu", vis.total_keyframes);
                ImGui::Text("Map Points: %zu", vis.total_map_points);
                ImGui::Text("Position: (%.2f, %.2f, %.2f)", 
                    vis.current_pose.translation.x(), vis.current_pose.translation.y(), vis.current_pose.translation.z());

                if (ImGui::Button(is_paused ? "Resume" : "Pause")) is_paused = !is_paused;
                ImGui::SameLine();
                if (ImGui::Button("Reset")) reset_slam();
                ImGui::SameLine();
                if (ImGui::Button("Exit")) should_close = true;
                ImGui::End();
            }
        }
    };
} 

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <PORT> [config.yaml]" << std::endl;
        return 1;
    }

    int port = std::stoi(argv[1]);
    std::string config_path = (argc > 2) ? argv[2] : "config.yaml";

    std::cout << "\n=== CAAI-SLAM Custom TCP Server ===" << std::endl;
    std::cout << "Listening on Port: " << port << std::endl;

    caai_slam::linux_tcp_slam_app app(config_path);
    if (!app.run_server(port)) return 1;

    return 0;
}