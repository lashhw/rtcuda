#ifndef RTCUDA_UTILITY_H
#define RTCUDA_UTILITY_H

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file,
                int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

__device__ Vec3 random_in_unit_sphere(curandState *local_rand_state) {
    // TODO: change to faster implementation
    Vec3 p;
    do {
        p = Vec3(2.0f*curand_uniform(local_rand_state)-1.0f,
                 2.0f*curand_uniform(local_rand_state)-1.0f,
                 2.0f*curand_uniform(local_rand_state)-1.0f);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ void random_in_unit_disk(curandState *local_rand_state, float &x, float &y) {
    // TODO: change to faster implementation
    do {
        x = 2.0f*curand_uniform(local_rand_state)-1.0f;
        y = 2.0f*curand_uniform(local_rand_state)-1.0f;
    } while (x*x+y*y >= 1.0f);
}

#endif //RTCUDA_UTILITY_H
