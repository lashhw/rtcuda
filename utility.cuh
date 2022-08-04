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

// reference: C Wachter & N Binder, A Fast and Robust Method for Avoiding Self-Intersection
__device__ Vec3 offset_ray_origin(const Vec3 &p, const Vec3 &unit_n) {
    constexpr float int_scale = 256.f;
    constexpr float float_scale = 1.f / 65536.f;
    constexpr float origin = 1.f / 32.f;

    int of_i_x = int_scale * unit_n.x;
    int of_i_y = int_scale * unit_n.y;
    int of_i_z = int_scale * unit_n.z;

    float p_i_x = __int_as_float(__float_as_int(p.x) + (p.x < 0 ? -of_i_x : of_i_x));
    float p_i_y = __int_as_float(__float_as_int(p.y) + (p.y < 0 ? -of_i_y : of_i_y));
    float p_i_z = __int_as_float(__float_as_int(p.z) + (p.z < 0 ? -of_i_z : of_i_z));

    return Vec3(fabsf(p.x) < origin ? p.x + float_scale * unit_n.x : p_i_x,
                fabsf(p.y) < origin ? p.y + float_scale * unit_n.y : p_i_y,
                fabsf(p.z) < origin ? p.z + float_scale * unit_n.z : p_i_z);
}

__device__ float abs_dot(const Vec3 &v1, const Vec3 &v2) {
    return fabsf(dot(v1, v2));
}

__device__ float power_heuristic(float f_pdf, int g_pdf) {
    float f_pdf_2 = f_pdf * f_pdf;
    return f_pdf_2 / (f_pdf_2 + g_pdf * g_pdf);
}

__device__ bool same_hemisphere(const Vec3 &wo, const Vec3 &wi, const Vec3 &n) {
    return dot(wo, n) * dot(wi, n) < 0.f;
}

__device__ void uniform_sample_disk(curandState &rand_state, float &x, float &y) {
    float r = sqrtf(curand_uniform(&rand_state));
    float theta = TWO_PI * curand_uniform(&rand_state);
    __sincosf(theta, &y, &x);
    x *= r;
    y *= r;
}

__device__ Vec3 uniform_sample_sphere(curandState &rand_state) {
    float z = 1 - 2 * curand_uniform(&rand_state);
    float r = sqrtf(1 - z * z);
    float phi = TWO_PI * curand_uniform(&rand_state);
    float x, y;
    __sincosf(phi, &y, &x);
    return Vec3(r * x, r * y, z);
}

int clamp(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

#endif //RTCUDA_UTILITY_CUH
