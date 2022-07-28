#ifndef RTCUDA_UTILITY_CUH
#define RTCUDA_UTILITY_CUH

#define CHECK_CUDA(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file,
                int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

const float PI = 3.14159265358979323846f;

__host__ __device__ float deg_to_rad(float deg) {
    return deg * (PI / 180.f);
}

__device__ Vec3 random_in_unit_sphere(curandState &rand_state) {
    // TODO: change to faster implementation
    Vec3 p;
    do {
        p = Vec3(2.0f * curand_uniform(&rand_state) - 1.0f,
                 2.0f * curand_uniform(&rand_state) - 1.0f,
                 2.0f * curand_uniform(&rand_state) - 1.0f);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ void random_in_unit_disk(curandState &rand_state, float &x, float &y) {
    // TODO: change to faster implementation
    do {
        x = 2.0f * curand_uniform(&rand_state) - 1.0f;
        y = 2.0f * curand_uniform(&rand_state) - 1.0f;
    } while (x * x + y * y >= 1.0f);
}

// reference: C Wachter & N Binder, A Fast and Robust Method for Avoiding Self-Intersection
__device__ Vec3 offset_ray_origin(const Vec3 &p, const Vec3 &n) {
    constexpr float int_scale = 256.f;
    constexpr float float_scale = 1.f / 65536.f;
    constexpr float origin = 1.f / 32.f;

    int of_i_x = int_scale * n.x;
    int of_i_y = int_scale * n.y;
    int of_i_z = int_scale * n.z;

    float p_i_x = __int_as_float(__float_as_int(p.x) + (p.x < 0 ? -of_i_x : of_i_x));
    float p_i_y = __int_as_float(__float_as_int(p.y) + (p.y < 0 ? -of_i_y : of_i_y));
    float p_i_z = __int_as_float(__float_as_int(p.z) + (p.z < 0 ? -of_i_z : of_i_z));

    return Vec3(fabsf(p.x) < origin ? p.x + float_scale * n.x : p_i_x,
                fabsf(p.y) < origin ? p.y + float_scale * n.y : p_i_y,
                fabsf(p.z) < origin ? p.z + float_scale * n.z : p_i_z);
}

int clamp(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

#endif //RTCUDA_UTILITY_CUH
