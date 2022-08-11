#ifndef RTCUDA_RENDER_CUH
#define RTCUDA_RENDER_CUH

enum RayType {
    PATH_RAY,
    AH_SHADOW_RAY,
    CH_SHADOW_RAY
};

// use structure-of-array for memory coalescing
struct RayPayload {
    RayType type[3 * NUM_WORKING_PATHS];
    int pixel_idx[3 * NUM_WORKING_PATHS];
    Ray ray[3 * NUM_WORKING_PATHS];
};

struct PathRayPayload {
    // intersection result
    bool hit_anything[NUM_WORKING_PATHS];
    Intersection isect[NUM_WORKING_PATHS];
    Primitive *d_isect_primitive[NUM_WORKING_PATHS];

    int bounces[NUM_WORKING_PATHS];
    Vec3 beta[NUM_WORKING_PATHS];
};

struct ShadowRayPayload {
    Triangle *d_target_triangle[NUM_WORKING_PATHS];
    Vec3 L[NUM_WORKING_PATHS];
};

__managed__ int num_mat_pending;
__managed__ int num_gen_pending;
__managed__ int num_ah_pending;
__managed__ int num_ch_pending;

__constant__ curandState *d_rand_states;
__constant__ Vec3 *d_framebuffer;

__constant__ int *d_mat_pending;
__constant__ int *d_gen_pending;
__constant__ int *d_ah_pending;
__constant__ int *d_ch_pending;

__constant__ bool *d_mat_pending_valid;
__constant__ bool *d_gen_pending_valid;
__constant__ bool *d_ah_pending_valid;
__constant__ bool *d_ch_pending_valid;

__constant__ int *d_mat_pending_compact;
__constant__ int *d_gen_pending_compact;
__constant__ int *d_ah_pending_compact;
__constant__ int *d_ch_pending_compact;

__constant__ RayPayload *d_ray_payload;
__constant__ PathRayPayload *d_path_ray_payload;
__constant__ ShadowRayPayload *d_ah_shadow_ray_payload;
__constant__ ShadowRayPayload *d_ch_shadow_ray_payload;

__constant__ int d_width;
__constant__ int d_height;
__constant__ int d_num_samples;
__constant__ int d_max_bounces;
__constant__ int d_camera_ray_end_id;
__constant__ Scene d_scene;
__constant__ Camera d_camera;

__global__ void init_framebuffer(int num_pixels) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_pixels) return;

    d_framebuffer[thread_id] = Vec3::make_zeros();
}

__global__ void init_rand_states(int rand_seed) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= NUM_WORKING_PATHS) return;

    curand_init(rand_seed, thread_id, 0, &d_rand_states[thread_id]);
}

__global__ void init_path_ray_payload() {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= NUM_WORKING_PATHS) return;

    // force the ray to be killed
    d_path_ray_payload->bounces[thread_id] = INT_MAX;
}

__global__ void init() {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= NUM_WORKING_PATHS) return;

    d_gen_pending_valid[thread_id] = false;
    d_mat_pending_valid[thread_id] = false;
    d_ah_pending_valid[thread_id] = false;
    d_ch_pending_valid[thread_id] = false;
    d_ch_pending_valid[NUM_WORKING_PATHS + thread_id] = false;
    d_ch_pending_valid[2 * NUM_WORKING_PATHS + thread_id] = false;

    int pixel_idx = d_ray_payload->pixel_idx[thread_id];
    bool hit_anything = d_path_ray_payload->hit_anything[thread_id];
    int bounces = d_path_ray_payload->bounces[thread_id];

    if (bounces == 0) {
        if (hit_anything) {
            Light *d_isect_area_light = d_path_ray_payload->d_isect_primitive[thread_id]->d_area_light;
            if (d_isect_area_light) {
                Vec3::atomic_add(&d_framebuffer[pixel_idx], d_isect_area_light->L);
            }
        } else {
            // TODO: add environment light
        }
    }

    if (bounces < d_max_bounces) {
        if (hit_anything) {
            d_mat_pending[thread_id] = thread_id;
            d_mat_pending_valid[thread_id] = true;
        }
    } else {
        d_gen_pending[thread_id] = thread_id;
        d_gen_pending_valid[thread_id] = true;
    }

    d_path_ray_payload->bounces[thread_id] = bounces + 1;
}

__global__ void mat() {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_mat_pending) return;

    int path_ray_id = d_mat_pending_compact[thread_id];

    int pixel_idx = d_ray_payload->pixel_idx[path_ray_id];
    Vec3 unit_wo = d_ray_payload->ray[path_ray_id].unit_d;
    Intersection isect = d_path_ray_payload->isect[path_ray_id];
    Primitive *d_isect_primitive = d_path_ray_payload->d_isect_primitive[path_ray_id];
    Material *d_mat = d_isect_primitive->d_mat;
    Vec3 multiplier = d_path_ray_payload->beta[path_ray_id] * d_scene.num_lights;
    curandState &rand_state = d_rand_states[path_ray_id];

    // generate next ray
    {
        Vec3 unit_n = isect.unit_n, unit_wi;
        float pdf;
        Vec3 f = d_mat->sample_f(unit_wo, rand_state, unit_n, unit_wi, pdf);

        // generate PATH_RAY
        d_ray_payload->ray[path_ray_id] = Ray::spawn_offset_ray(isect.p, unit_n, unit_wi);
        d_path_ray_payload->bounces[path_ray_id] += 1;
        d_path_ray_payload->beta[path_ray_id] *= f * dot(unit_wi, unit_n) / pdf;

        // add to d_ch_pending
        d_ch_pending[NUM_WORKING_PATHS + thread_id] = path_ray_id;
        d_ch_pending_valid[NUM_WORKING_PATHS + thread_id] = true;
    }

    // uniformly sample one light source
    if (d_scene.num_lights == 0) return;
    int light_idx = min((int)(curand_uniform(&rand_state) * d_scene.num_lights),
                        d_scene.num_lights - 1);
    const Light light = d_scene.d_lights[light_idx];

    // sample light source with MIS
    {
        Vec3 unit_wi, Li;
        float light_t, light_pdf;
        if (light.sample_Li(isect, rand_state, unit_wi, Li, light_t, light_pdf)) {
            Vec3 unit_n = dot(isect.unit_n, unit_wi) > 0.f ? isect.unit_n : -isect.unit_n;
            Vec3 f;
            float scattering_pdf;
            if (d_mat->get_f(unit_wo, unit_wi, unit_n, f, scattering_pdf)) {
                f *= dot(unit_wi, unit_n);

                // generate AH_SHADOW_RAY
                int ah_ray_id = NUM_WORKING_PATHS + path_ray_id;
                d_ray_payload->type[ah_ray_id] = AH_SHADOW_RAY;
                d_ray_payload->pixel_idx[ah_ray_id] = pixel_idx;
                d_ray_payload->ray[ah_ray_id] = Ray::spawn_offset_ray(isect.p, unit_n, unit_wi, light_t);
                d_ah_shadow_ray_payload->d_target_triangle[path_ray_id] = light.d_triangle;
                if (light.is_delta()) {
                    d_ah_shadow_ray_payload->L[path_ray_id] = multiplier * f * Li / light_pdf;
                } else {
                    float weight = power_heuristic(light_pdf, scattering_pdf);
                    d_ah_shadow_ray_payload->L[path_ray_id] = multiplier * f * Li * weight / light_pdf;
                }

                // add to d_ah_pending
                d_ah_pending[thread_id] = ah_ray_id;
                d_ah_pending_valid[thread_id] = true;
            }
        }
    }

    // sample BSDF with MIS
    {
        if (!light.is_delta()) {
            // sample a direction based on material's BSDF
            Vec3 unit_n = isect.unit_n, unit_wi;
            float scattering_pdf;
            Vec3 f = d_mat->sample_f(unit_wo, rand_state, unit_n, unit_wi, scattering_pdf);
            f *= dot(unit_wi, unit_n);

            // if BSDF is specular, there is no need to apply MIS
            float weight = 1.f;
            if (!d_mat->is_specular()) {
                float light_pdf = light.pdf_Li(isect, unit_wi);
                if (light_pdf == 0.f) return;
                weight = power_heuristic(scattering_pdf, light_pdf);
            }

            // generate CH_SHADOW_RAY
            int ch_ray_id = 2 * NUM_WORKING_PATHS + path_ray_id;
            d_ray_payload->type[ch_ray_id] = CH_SHADOW_RAY;
            d_ray_payload->pixel_idx[ch_ray_id] = pixel_idx;
            d_ray_payload->ray[ch_ray_id] = Ray::spawn_offset_ray(isect.p, unit_n, unit_wi);
            d_ch_shadow_ray_payload->d_target_triangle[path_ray_id] = d_isect_primitive->d_triangle;
            d_ch_shadow_ray_payload->L[path_ray_id] = multiplier * f * light.L * weight / scattering_pdf;

            // add to d_ch_pending
            d_ch_pending[2 * NUM_WORKING_PATHS + thread_id] = ch_ray_id;
            d_ch_pending_valid[2 * NUM_WORKING_PATHS + thread_id] = true;

            // TODO: add environment light
        }
    }
}

__global__ void gen(int camera_ray_start_id) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_gen_pending) return;

    int camera_ray_id = camera_ray_start_id + thread_id;
    if (camera_ray_id >= d_camera_ray_end_id) return;

    int pixel_idx = camera_ray_id / d_num_samples;
    int i = pixel_idx % d_width;
    int j = pixel_idx / d_width;

    int path_ray_id = d_gen_pending_compact[thread_id];

    d_ray_payload->type[path_ray_id] = PATH_RAY;
    d_ray_payload->pixel_idx[path_ray_id] = pixel_idx;
    d_ray_payload->ray[path_ray_id] = d_camera.get_ray((i + curand_uniform(&d_rand_states[path_ray_id])) / d_width,
                                                       (j + curand_uniform(&d_rand_states[path_ray_id])) / d_height);
    d_path_ray_payload->bounces[path_ray_id] = 0;
    d_path_ray_payload->beta[path_ray_id] = Vec3::make_ones();

    d_ch_pending[thread_id] = path_ray_id;
    d_ch_pending_valid[thread_id] = true;
}

// any-hit
__global__ void ah() {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_ah_pending) return;

    int ah_ray_id = d_ah_pending_compact[thread_id];
    int path_ray_id = ah_ray_id - NUM_WORKING_PATHS;

    DeviceStack stack;
    Ray ray = d_ray_payload->ray[ah_ray_id];
    Triangle *d_target_triangle = d_ah_shadow_ray_payload->d_target_triangle[path_ray_id];
    bool hit_anything = d_scene.bvh.traverse(d_target_triangle, stack, ray);

    if (!hit_anything) {
        Vec3::atomic_add(&d_framebuffer[d_ray_payload->pixel_idx[ah_ray_id]], d_ah_shadow_ray_payload->L[path_ray_id]);
    }
}

// closest-hit
__global__ void ch() {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_ch_pending) return;

    int ch_ray_id = d_ch_pending_compact[thread_id];
    int path_ray_id = ch_ray_id - 2 * NUM_WORKING_PATHS;

    DeviceStack stack;
    Ray ray = d_ray_payload->ray[ch_ray_id];
    Intersection isect;
    Primitive* d_isect_primitive;

    bool hit_anything = d_scene.bvh.traverse(stack, ray, isect, d_isect_primitive);

    if (d_ray_payload->type[ch_ray_id] == CH_SHADOW_RAY) {
        if (hit_anything) {
            if (d_ch_shadow_ray_payload->d_target_triangle[path_ray_id] == d_isect_primitive->d_triangle) {
                Vec3::atomic_add(&d_framebuffer[d_ray_payload->pixel_idx[ch_ray_id]], d_ch_shadow_ray_payload->L[path_ray_id]);
            }
        } else {
            // TODO: add environment light
        }
    } else {
        // type == PATH_RAY
        // save intersection result
        d_path_ray_payload->hit_anything[ch_ray_id] = hit_anything;
        d_path_ray_payload->isect[ch_ray_id] = isect;
        d_path_ray_payload->d_isect_primitive[ch_ray_id] = d_isect_primitive;
    }
}

__global__ void post_process_framebuffer(int num_pixels) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_pixels) return;

    Vec3 tmp = d_framebuffer[thread_id];
    tmp /= d_num_samples;
    tmp.sqrt_inplace();
    d_framebuffer[thread_id] = tmp;
}

template <typename T>
T* cuda_malloc_symbol(T* &symbol, const size_t size) {
    T* tmp;
    CHECK_CUDA(cudaMalloc(&tmp, size));
    CHECK_CUDA(cudaMemcpyToSymbol(symbol, &tmp, sizeof(T*)));
    return tmp;
}

void compact(int num_items, int *d_in, bool *d_flags, int *d_out, int *d_num_selected_out) {
    static void *d_temp_storage = NULL;
    static size_t temp_storage_bytes = 0;

    size_t new_temp_storage_bytes;
    CHECK_CUDA(cub::DeviceSelect::Flagged(NULL, new_temp_storage_bytes, d_in, d_flags,
                                          d_out, d_num_selected_out, num_items));

    if (new_temp_storage_bytes > temp_storage_bytes) {
        CHECK_CUDA(cudaFree(d_temp_storage));
        CHECK_CUDA(cudaMalloc(&d_temp_storage, new_temp_storage_bytes));
        temp_storage_bytes = new_temp_storage_bytes;
    }

    CHECK_CUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags,
                                          d_out, d_num_selected_out, num_items));
    CHECK_CUDA(cudaDeviceSynchronize());
}

void render(int width, int height, int num_samples, int max_bounces,
            Camera camera, Scene scene, std::vector<Vec3> &framebuffer) {
    // initialize some useful variables
    int num_pixels = width * height;
    int camera_ray_start_id = 0;
    int camera_ray_end_id = num_pixels * num_samples;

    // allocate memory on device
    cuda_malloc_symbol(d_rand_states, NUM_WORKING_PATHS * sizeof(curandState));
    Vec3 *d_framebuffer_ptr = cuda_malloc_symbol(d_framebuffer, num_pixels * sizeof(Vec3));
    int *d_mat_pending_ptr = cuda_malloc_symbol(d_mat_pending, NUM_WORKING_PATHS * sizeof(int));
    int *d_gen_pending_ptr = cuda_malloc_symbol(d_gen_pending, NUM_WORKING_PATHS * sizeof(int));
    int *d_ah_pending_ptr = cuda_malloc_symbol(d_ah_pending, NUM_WORKING_PATHS * sizeof(int));
    int *d_ch_pending_ptr = cuda_malloc_symbol(d_ch_pending, 3 * NUM_WORKING_PATHS * sizeof(int));
    bool *d_mat_pending_valid_ptr = cuda_malloc_symbol(d_mat_pending_valid, NUM_WORKING_PATHS * sizeof(bool));
    bool *d_gen_pending_valid_ptr = cuda_malloc_symbol(d_gen_pending_valid, NUM_WORKING_PATHS * sizeof(bool));
    bool *d_ah_pending_valid_ptr = cuda_malloc_symbol(d_ah_pending_valid, NUM_WORKING_PATHS * sizeof(bool));
    bool *d_ch_pending_valid_ptr = cuda_malloc_symbol(d_ch_pending_valid, 3 * NUM_WORKING_PATHS * sizeof(bool));
    int *d_mat_pending_compact_ptr = cuda_malloc_symbol(d_mat_pending_compact, NUM_WORKING_PATHS * sizeof(int));
    int *d_gen_pending_compact_ptr = cuda_malloc_symbol(d_gen_pending_compact, NUM_WORKING_PATHS * sizeof(int));
    int *d_ah_pending_compact_ptr = cuda_malloc_symbol(d_ah_pending_compact, NUM_WORKING_PATHS * sizeof(int));
    int *d_ch_pending_compact_ptr = cuda_malloc_symbol(d_ch_pending_compact, 3 * NUM_WORKING_PATHS * sizeof(int));
    cuda_malloc_symbol(d_ray_payload, sizeof(RayPayload));
    cuda_malloc_symbol(d_path_ray_payload, sizeof(PathRayPayload));
    cuda_malloc_symbol(d_ah_shadow_ray_payload, sizeof(ShadowRayPayload));
    cuda_malloc_symbol(d_ch_shadow_ray_payload, sizeof(ShadowRayPayload));

    // copy some variables to device
    CHECK_CUDA(cudaMemcpyToSymbol(d_width, &width, sizeof(int)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_height, &height, sizeof(int)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_num_samples, &num_samples, sizeof(int)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_max_bounces, &max_bounces, sizeof(int)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_camera_ray_end_id, &camera_ray_end_id, sizeof(int)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_scene, &scene, sizeof(Scene)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_camera, &camera, sizeof(Camera)));

    // initialize d_framebuffer
    constexpr int BLOCK_SIZE = 64;
    init_framebuffer<<<(num_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(num_pixels);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // initialize d_rand_states
    constexpr int RAND_SEED = 1;
    init_rand_states<<<(NUM_WORKING_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(RAND_SEED);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // initialize d_path_ray_payload
    init_path_ray_payload<<<(NUM_WORKING_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // start rendering
    while (true) {
        init<<<(NUM_WORKING_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>();
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        compact(NUM_WORKING_PATHS, d_mat_pending_ptr, d_mat_pending_valid_ptr, d_mat_pending_compact_ptr, &num_mat_pending);
        compact(NUM_WORKING_PATHS, d_gen_pending_ptr, d_gen_pending_valid_ptr, d_gen_pending_compact_ptr, &num_gen_pending);
        // termination condition (unable to spawn any ray)
        if (num_gen_pending == NUM_WORKING_PATHS && camera_ray_start_id >= camera_ray_end_id) break;

        if (num_mat_pending > 0) {
            mat<<<(num_mat_pending + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>();
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        if (num_gen_pending > 0) {
            gen<<<(num_gen_pending + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(camera_ray_start_id);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            camera_ray_start_id += num_gen_pending;
        }

        compact(NUM_WORKING_PATHS, d_ah_pending_ptr, d_ah_pending_valid_ptr, d_ah_pending_compact_ptr, &num_ah_pending);
        compact(3 * NUM_WORKING_PATHS, d_ch_pending_ptr, d_ch_pending_valid_ptr, d_ch_pending_compact_ptr, &num_ch_pending);

        if (num_ah_pending > 0) {
            ah<<<(num_ah_pending + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>();
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        if (num_ch_pending > 0) {
            ch<<<(num_ch_pending + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>();
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }

    // post-process framebuffer
    post_process_framebuffer<<<(num_pixels + BLOCK_SIZE - 1), BLOCK_SIZE>>>(num_pixels);

    // copy framebuffer to host
    framebuffer.resize(num_pixels);
    CHECK_CUDA(cudaMemcpy(framebuffer.data(), d_framebuffer_ptr, num_pixels * sizeof(Vec3), cudaMemcpyDeviceToHost));
}

#endif //RTCUDA_RENDER_CUH
