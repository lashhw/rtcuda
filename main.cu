#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <curand_kernel.h>

#include "Vec3.h"
#include "Ray.h"
#include "utility.h"
#include "HitRecord.h"
#include "Material.h"
#include "Primitive.h"
#include "Camera.h"

template <int RECURSION_DEPTH = 50>
__device__ __forceinline__ Vec3 get_color(const Ray &ray, Primitive **primitive_ptrs, int num_primitives,
                                          curandState *rand_state) {
    Ray cur_ray = ray;
    Vec3 cur_attenuation = Vec3(1.0f, 1.0f, 1.0f);

    HitRecord tmp_rec;
    Vec3 tmp_attenuation;

    for (int i = 0; i < RECURSION_DEPTH; i++) {
        bool hit_anything = false;
        for (int j = 0; j < num_primitives; j++) {
            if (primitive_ptrs[j]->hit(cur_ray, tmp_rec)) {
                hit_anything = true;
                cur_ray.tmax = tmp_rec.t;
            }
        }

        if (hit_anything) {
            if (tmp_rec.mat_ptr->scatter(cur_ray, tmp_rec, rand_state, tmp_attenuation, cur_ray)) {
                cur_attenuation *= tmp_attenuation;
            } else {
                return Vec3(0.0f, 0.0f, 0.0f);
            }
        } else {
            return cur_attenuation;
        }
    }

    return Vec3(0.0f, 0.0f, 0.0f);
}

__global__ void create_world(Primitive **primitive_ptrs, Camera **camera_ptrs, float aspect_ratio) {
    if (threadIdx.x == 0 & blockIdx.x == 0) {
        primitive_ptrs[0] = new Sphere(Vec3(0.0f, -100.0f, -1.0f), 100.0f, new Lambertian(Vec3(0.8f, 0.8f, 0.0f)));
        primitive_ptrs[1] = new Sphere(Vec3(0.5f, 0.5f, -1.0f), 0.5f, new Lambertian(Vec3(0.8f, 0.3f, 0.3f)));
        primitive_ptrs[2] = new Sphere(Vec3(1.5f, 0.5f, -1.0f), 0.5f, new Metal(Vec3(0.8f, 0.6f, 0.2f), 1.0f));
        primitive_ptrs[3] = new Triangle(Vec3(-0.5f, 0.5f, -1.0f), Vec3(0.0f, 0.5f, -1.0f), Vec3(0.0f, 0.0f, -1.0f),
                                         new Lambertian(Vec3(0.8f, 0.8f, 0.8f)));
        camera_ptrs[0] = new Camera(Vec3(0.5f, 0.5f, 1.0f),
                                    Vec3(0.5f, 0.5f, -1.0f),
                                    Vec3(0.0f, 1.0f, 0.0f),
                                    0.92f, aspect_ratio, 0.0f);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_idx = j * max_x + i;
    curand_init(1, pixel_idx, 0, &rand_state[pixel_idx]);
}

__global__ void render(Vec3 *fb, curandState *rand_state, int num_samples,
                       int max_x, int max_y,
                       Primitive **primitive_ptrs, int num_primitives,
                       Camera **camera_ptrs) {
    // TODO: eliminate fp_64 operations
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_idx = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_idx];
    Vec3 color(0.0f, 0.0f, 0.0f);

    float inv_max_x = 1.0f / max_x;
    float inv_max_y = 1.0f / max_y;
    for (int s = 0; s < num_samples; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) * inv_max_x;
        float v = float(j + curand_uniform(&local_rand_state)) * inv_max_y;
        Ray r = camera_ptrs[0]->get_ray(u, v, &local_rand_state);
        color += get_color(r, primitive_ptrs, num_primitives, &local_rand_state);
    }
    rand_state[pixel_idx] = local_rand_state;

    color /= num_samples;
    color.sqrt_inplace();
    fb[pixel_idx] = color;
}

int main() {
    //
    constexpr int WIDTH = 600;
    constexpr int HEIGHT = 600;
    constexpr int WIDTH_PER_BLOCK = 8;
    constexpr int HEIGHT_PER_BLOCK = 8;
    constexpr int NUM_SAMPLES = 100;

    constexpr float ASPECT_RATIO = float(WIDTH) / float(HEIGHT);

    constexpr int NUM_PIXELS = WIDTH * HEIGHT;
    Vec3 *d_fb;
    checkCudaErrors(cudaMalloc(&d_fb, NUM_PIXELS*sizeof(Vec3)));

    constexpr int NUM_PRIMITIVES = 4;
    Primitive **d_primitive_ptrs;
    checkCudaErrors(cudaMalloc(&d_primitive_ptrs, NUM_PRIMITIVES*sizeof(Primitive*)));

    Camera **d_camera_ptrs;
    checkCudaErrors(cudaMalloc(&d_camera_ptrs, sizeof(Camera*)));

    create_world<<<1, 1>>>(d_primitive_ptrs, d_camera_ptrs, ASPECT_RATIO);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc(&d_rand_state, NUM_PIXELS*sizeof(curandState)));

    constexpr dim3 BLOCKS_PER_GRID((WIDTH + WIDTH_PER_BLOCK - 1) / WIDTH_PER_BLOCK,
                                   (HEIGHT + HEIGHT_PER_BLOCK - 1) / HEIGHT_PER_BLOCK);
    constexpr dim3 THREADS_PER_BLOCK(WIDTH_PER_BLOCK, HEIGHT_PER_BLOCK);

    render_init<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(WIDTH, HEIGHT, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_fb, d_rand_state, NUM_SAMPLES,
                                                   WIDTH, HEIGHT,
                                                   d_primitive_ptrs, NUM_PRIMITIVES,
                                                   d_camera_ptrs);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Vec3 *h_fb = new Vec3[NUM_PIXELS];
    checkCudaErrors(cudaMemcpy(h_fb, d_fb, NUM_PIXELS*sizeof(Vec3), cudaMemcpyDeviceToHost));

    std::ofstream file("image.ppm");
    file << "P3\n" << WIDTH << ' ' << HEIGHT << "\n255\n";
    for (int j = HEIGHT - 1; j >= 0; j--) {
        for (int i = 0; i < WIDTH; i++) {
            size_t pixel_idx = j * WIDTH + i;
            float r = h_fb[pixel_idx].x;
            float g = h_fb[pixel_idx].y;
            float b = h_fb[pixel_idx].z;
            int ir = std::clamp(int(256.0f * r), 0, 255);
            int ig = std::clamp(int(256.0f * g), 0, 255);
            int ib = std::clamp(int(256.0f * b), 0, 255);
            file << ir << ' ' << ig << ' ' << ib << "\n";
        }
    }

    checkCudaErrors(cudaDeviceReset());
    return 0;
}
