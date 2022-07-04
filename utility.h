#ifndef RTCUDA_UTILITY_H
#define RTCUDA_UTILITY_H

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

__device__ Vec3 random_in_unit_sphere(curandState *local_rand_state) {
    Vec3 p;
    do {
        p = Vec3(2.0f*curand_uniform(local_rand_state)-1.0f,
                 2.0f*curand_uniform(local_rand_state)-1.0f,
                 2.0f*curand_uniform(local_rand_state)-1.0f);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ void random_in_unit_disk(curandState *local_rand_state, float &x, float &y) {
    do {
        x = 2.0f*curand_uniform(local_rand_state)-1.0f;
        y = 2.0f*curand_uniform(local_rand_state)-1.0f;
    } while (x*x+y*y >= 1.0f);
}

#endif //RTCUDA_UTILITY_H
