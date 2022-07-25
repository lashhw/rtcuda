#ifndef RTCUDA_PROFILER_HPP
#define RTCUDA_PROFILER_HPP

struct Profiler {
    typedef std::chrono::time_point<std::chrono::steady_clock> time_point_t;
    void start(std::string name);
    void stop();

    bool running = false;
    time_point_t start_time;
};

Profiler profiler;

void Profiler::start(std::string name) {
    assert(!running);
    running = true;
    std::cout << name << "... " << std::flush;
    start_time = std::chrono::steady_clock::now();
}

void Profiler::stop() {
    time_point_t end_time = std::chrono::steady_clock::now();
    assert(running);
    running = false;
    std::cout << "done (";
    float ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    std::cout << ms << "ms)" << std::endl;
}

#endif //RTCUDA_PROFILER_HPP
