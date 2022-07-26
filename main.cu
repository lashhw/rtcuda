#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <numeric>
#include <memory>
#include <cfloat>
#include <array>
#include <cassert>
#include <chrono>
#include <algorithm>
#include <stack>

#include <curand_kernel.h>

#include "profiler.hpp"
#include "vec3.cuh"
#include "matrix4x4.hpp"
#include "transform.hpp"
#include "ray.cuh"
#include "bounding_box.cuh"
#include "aabb_intersector.cuh"
#include "utility.cuh"
#include "intersection.hpp"
#include "material.cuh"
#include "primitive.cuh"
#include "bvh.cuh"
#include "device_stack.cuh"
#include "camera.cuh"
#include "happly.h"

template <typename Stack>
__device__ Vec3 get_color(const Ray &ray, const Bvh &bvh, Stack &stack, curandState &rand_state) {
    static constexpr int RECURSION_DEPTH = 10;
    static constexpr Vec3 BACKGROUND_COLOR = Vec3(0.f, 0.f, 0.f);

    Ray cur_ray = ray;
    Vec3 cur_attenuation = Vec3(1.f, 1.f, 1.f);
    Vec3 accumulate_radiance = Vec3(0.f, 0.f, 0.f);

    Intersection isect;
    Vec3 tmp_attenuation;

    for (int i = 0; i < RECURSION_DEPTH; i++) {
        if (bvh.traverse(stack, cur_ray, isect)) {
            Vec3 emit_spectrum;
            Material *d_mat = isect.d_mat;
            if (d_mat->emit(emit_spectrum)) {
                accumulate_radiance += cur_attenuation * emit_spectrum;
            }
            if (d_mat->scatter(cur_ray, isect, rand_state, tmp_attenuation, cur_ray)) {
                cur_attenuation *= tmp_attenuation;
            } else {
                return accumulate_radiance;
            }
        } else {
            return accumulate_radiance + cur_attenuation * BACKGROUND_COLOR;
        }
    }

    return Vec3(0.f, 0.f, 0.f);
}

__global__ void render_init(int width, int height, curandState *d_rand_state) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;

    int pixel_idx = j * width + i;
    curand_init(1, pixel_idx, 0, &d_rand_state[pixel_idx]);
}

// TODO: eliminate fp_64 operations
__global__ void render(Camera<false> camera, Bvh bvh, int num_samples, int width, int height,
                       curandState *d_rand_state, Vec3 *d_framebuffer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;

    int pixel_idx = j * width + i;
    curandState local_rand_state = d_rand_state[pixel_idx];
    Vec3 color(0.f, 0.f, 0.f);
    DeviceStack<Bvh::MAX_DEPTH - 1> stack;

    float inv_width = 1.f / width;
    float inv_height = 1.f / height;
    for (int s = 0; s < num_samples; s++) {
        float x = (i + curand_uniform(&local_rand_state)) * inv_width;
        float y = (j + curand_uniform(&local_rand_state)) * inv_height;
        Ray ray = camera.get_ray(x, y);
        color += get_color(ray, bvh, stack, local_rand_state);
    }

    d_rand_state[pixel_idx] = local_rand_state;
    color /= num_samples;
    color.sqrt_inplace();
    d_framebuffer[pixel_idx] = color;
}

int main() {
    // create materials on host
    constexpr int NUM_MATERIALS = 7;
    std::vector<Material> materials(NUM_MATERIALS);
    materials[0] = Material::create_matte(Vec3(0.65f, 0.05f, 0.05f));
    materials[1] = Material::create_matte(Vec3(0.12f, 0.45f, 0.15f));
    materials[2] = Material::create_matte(Vec3(0.73f, 0.73f, 0.73f));
    materials[3] = Material::create_mirror(Vec3(0.8f, 0.8f, 0.9f));
    materials[4] = Material::create_metal(Vec3(0.9f, 0.73f, 0.05f), 0.5f);
    materials[5] = Material::create_glass(1.5f);
    materials[6] = Material::create_light(Vec3(15.f, 15.f, 15.f));

    // move materials to device
    Material *d_materials;
    CHECK_CUDA(cudaMalloc(&d_materials, NUM_MATERIALS * sizeof(Material)));
    CHECK_CUDA(cudaMemcpy(d_materials, materials.data(), NUM_MATERIALS * sizeof(Material), cudaMemcpyHostToDevice));
    materials.clear();
    Material *d_red    = &d_materials[0];
    Material *d_green  = &d_materials[1];
    Material *d_white  = &d_materials[2];
    Material *d_mirror = &d_materials[3];
    Material *d_gold   = &d_materials[4];
    Material *d_glass  = &d_materials[5];
    Material *d_light  = &d_materials[6];

    // create world on host
    constexpr int NUM_PRIMITIVES = 12;
    std::vector<Triangle> primitives(NUM_PRIMITIVES);
    primitives[0] = Triangle(Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, -1.0f), Vec3(0.0f, 1.0f, -1.0f), d_red);
    primitives[1] = Triangle(Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), Vec3(0.0f, 1.0f, -1.0f), d_red);
    primitives[2] = Triangle(Vec3(1.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, -1.0f), Vec3(1.0f, 1.0f, -1.0f), d_green);
    primitives[3] = Triangle(Vec3(1.0f, 0.0f, 0.0f), Vec3(1.0f, 1.0f, 0.0f), Vec3(1.0f, 1.0f, -1.0f), d_green);
    primitives[4] = Triangle(Vec3(0.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, -1.0f), d_white);
    primitives[5] = Triangle(Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, -1.0f), Vec3(1.0f, 0.0f, -1.0f), d_white);
    primitives[6] = Triangle(Vec3(0.0f, 1.0f, 0.0f), Vec3(1.0f, 1.0f, 0.0f), Vec3(1.0f, 1.0f, -1.0f), d_white);
    primitives[7] = Triangle(Vec3(0.0f, 1.0f, 0.0f), Vec3(0.0f, 1.0f, -1.0f), Vec3(1.0f, 1.0f, -1.0f), d_white);
    primitives[8] = Triangle(Vec3(0.0f, 0.0f, -1.0f), Vec3(1.0f, 0.0f, -1.0f), Vec3(1.0f, 1.0f, -1.0f), d_white);
    primitives[9] = Triangle(Vec3(0.0f, 0.0f, -1.0f), Vec3(0.0f, 1.0f, -1.0f), Vec3(1.0f, 1.0f, -1.0f), d_white);
    primitives[10] = Triangle(Vec3(0.4f, 0.99f, -0.4f), Vec3(0.6f, 0.99f, -0.4f), Vec3(0.6f, 0.99f, -0.6f), d_light);
    primitives[11] = Triangle(Vec3(0.4f, 0.99f, -0.4f), Vec3(0.4f, 0.99f, -0.6f), Vec3(0.6f, 0.99f, -0.6f), d_light);

    // build bvh
    Bvh bvh(primitives);

    // create camera
    constexpr int WIDTH = 600;
    constexpr int HEIGHT = 600;
    constexpr float ASPECT_RATIO = (float)WIDTH / (float)HEIGHT;
    Camera<false> camera(Vec3(0.5f, 0.5f, 1.5f),
                         Vec3(0.5f, 0.5f, 0.0f),
                         Vec3(0.0f, 1.0f, 0.0f),
                         37.8f,
                         ASPECT_RATIO);

    // initialize rand_state
    constexpr int NUM_PIXELS = WIDTH * HEIGHT;
    constexpr int WIDTH_PER_BLOCK = 8;
    constexpr int HEIGHT_PER_BLOCK = 8;
    constexpr dim3 GRID_SIZE((WIDTH + WIDTH_PER_BLOCK - 1) / WIDTH_PER_BLOCK,
                             (HEIGHT + HEIGHT_PER_BLOCK - 1) / HEIGHT_PER_BLOCK);
    constexpr dim3 BLOCK_SIZE(WIDTH_PER_BLOCK, HEIGHT_PER_BLOCK);
    curandState *d_rand_state;
    CHECK_CUDA(cudaMalloc(&d_rand_state, NUM_PIXELS * sizeof(curandState)));
    render_init<<<GRID_SIZE, BLOCK_SIZE>>>(WIDTH, HEIGHT, d_rand_state);
    CHECK_CUDA(cudaGetLastError());

    // allocate framebuffer on device
    Vec3 *d_framebuffer;
    CHECK_CUDA(cudaMalloc(&d_framebuffer, NUM_PIXELS * sizeof(Vec3)));

    // start rendering
    constexpr int NUM_SAMPLES = 1000;
    render<<<GRID_SIZE, BLOCK_SIZE>>>(camera, bvh, NUM_SAMPLES, WIDTH, HEIGHT, d_rand_state, d_framebuffer);
    CHECK_CUDA(cudaGetLastError());

    // copy framebuffer to host
    auto h_framebuffer = std::make_unique<Vec3[]>(NUM_PIXELS);
    CHECK_CUDA(cudaMemcpy(h_framebuffer.get(), d_framebuffer, NUM_PIXELS * sizeof(Vec3), cudaMemcpyDeviceToHost));

    // write image
    std::ofstream file("image.ppm");
    file << "P3\n" << WIDTH << ' ' << HEIGHT << "\n255\n";
    for (int j = 0; j < HEIGHT; j++) {
        for (int i = 0; i < WIDTH; i++) {
            size_t pixel_idx = j * WIDTH + i;
            float r = h_framebuffer[pixel_idx].x;
            float g = h_framebuffer[pixel_idx].y;
            float b = h_framebuffer[pixel_idx].z;
            int ir = clamp(int(256.f * r), 0, 255);
            int ig = clamp(int(256.f * g), 0, 255);
            int ib = clamp(int(256.f * b), 0, 255);
            file << ir << ' ' << ig << ' ' << ib << "\n";
        }
    }

    return 0;
}