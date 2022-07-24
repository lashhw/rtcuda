#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <numeric>
#include <memory>
#include <cfloat>
#include <array>
#include <algorithm>
#include <stack>

#include <curand_kernel.h>

#include "vec3.cuh"
#include "matrix4x4.hpp"
#include "transform.hpp"
#include "ray.cuh"
#include "bounding_box.cuh"
#include "aabb_intersector.cuh"
#include "utility.cuh"
#include "hit_record.cuh"
#include "material.cuh"
#include "primitive.cuh"
#include "bvh.cuh"
#include "camera.cuh"
#include "happly.h"

__device__ Vec3 get_color(const Ray &ray, Primitive **primitive_ptrs, int num_primitives,
                          curandState *rand_state) {
    static constexpr int RECURSION_DEPTH = 10;

    Ray cur_ray = ray;
    Vec3 cur_attenuation = Vec3(1.0f, 1.0f, 1.0f);

    HitRecord rec;
    Vec3 tmp_attenuation;

    Vec3 accumulate_spectrum = Vec3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < RECURSION_DEPTH; i++) {
        bool hit_anything = false;
        for (int j = 0; j < num_primitives; j++) {
            if (primitive_ptrs[j]->hit(cur_ray, rec)) {
                hit_anything = true;
                cur_ray.tmax = rec.t;
            }
        }

        if (hit_anything) {
            Vec3 emit_spectrum;
            if(rec.mat_ptr->emit(emit_spectrum)) {
                accumulate_spectrum += cur_attenuation * emit_spectrum;
            }
            if (rec.mat_ptr->scatter(cur_ray, rec, rand_state, tmp_attenuation, cur_ray)) {
                cur_attenuation *= tmp_attenuation;
            } else {
                return accumulate_spectrum;
            }
        } else {
            return accumulate_spectrum + cur_attenuation * Vec3(0.0f, 0.0f, 0.0f);
        }
    }

    return Vec3(0.0f, 0.0f, 0.0f);
}

__global__ void create_world(Primitive **primitive_ptrs, Camera **camera_ptrs, float aspect_ratio) {
    if (threadIdx.x == 0 & blockIdx.x == 0) {
        Material *red = new Lambertian(Vec3(0.65f, 0.05f, 0.05f));
        Material *green = new Lambertian(Vec3(0.12f, 0.45f, 0.15f));
        Material *white = new Lambertian(Vec3(0.73f, 0.73f, 0.73f));
        Material *mirror = new Metal(Vec3(0.8f, 0.8f, 0.9f), 0.0f);
        Material *gold = new Metal(Vec3(0.9f, 0.73f, 0.05f), 1.0f);
        Material *glass = new Dielectric(1.5f);
        Material *light = new Light(Vec3(15.0f, 15.0f, 15.0f));

        primitive_ptrs[0] = new Triangle(Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, -1.0f), Vec3(0.0f, 1.0f, -1.0f),
                                         red);
        primitive_ptrs[1] = new Triangle(Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), Vec3(0.0f, 1.0f, -1.0f),
                                         red);
        primitive_ptrs[2] = new Triangle(Vec3(1.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, -1.0f), Vec3(1.0f, 1.0f, -1.0f),
                                         green);
        primitive_ptrs[3] = new Triangle(Vec3(1.0f, 0.0f, 0.0f), Vec3(1.0f, 1.0f, 0.0f), Vec3(1.0f, 1.0f, -1.0f),
                                         green);
        primitive_ptrs[4] = new Triangle(Vec3(0.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, -1.0f),
                                         white);
        primitive_ptrs[5] = new Triangle(Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, -1.0f), Vec3(1.0f, 0.0f, -1.0f),
                                         white);
        primitive_ptrs[6] = new Triangle(Vec3(0.0f, 1.0f, 0.0f), Vec3(1.0f, 1.0f, 0.0f), Vec3(1.0f, 1.0f, -1.0f),
                                         white);
        primitive_ptrs[7] = new Triangle(Vec3(0.0f, 1.0f, 0.0f), Vec3(0.0f, 1.0f, -1.0f), Vec3(1.0f, 1.0f, -1.0f),
                                         white);
        primitive_ptrs[8] = new Triangle(Vec3(0.0f, 0.0f, -1.0f), Vec3(1.0f, 0.0f, -1.0f), Vec3(1.0f, 1.0f, -1.0f),
                                         white);
        primitive_ptrs[9] = new Triangle(Vec3(0.0f, 0.0f, -1.0f), Vec3(0.0f, 1.0f, -1.0f), Vec3(1.0f, 1.0f, -1.0f),
                                         white);
        primitive_ptrs[10] = new Triangle(Vec3(0.4f, 0.99f, -0.4f), Vec3(0.6f, 0.99f, -0.4f), Vec3(0.6f, 0.99f, -0.6f),
                                          light);
        primitive_ptrs[11] = new Triangle(Vec3(0.4f, 0.99f, -0.4f), Vec3(0.4f, 0.99f, -0.6f), Vec3(0.6f, 0.99f, -0.6f),
                                          light);
        primitive_ptrs[12] = new Sphere(Vec3(0.75f, 0.15f, -0.55f), 0.15f, mirror);
        primitive_ptrs[13] = new Sphere(Vec3(0.25f, 0.15f, -0.35f), 0.15f, glass);
        primitive_ptrs[14] = new Sphere(Vec3(0.55f, 0.10f, -0.15f), 0.10f, gold);

        camera_ptrs[0] = new Camera(Vec3(0.5f, 0.5f, 1.5f),
                                    Vec3(0.5f, 0.5f, 0.0f),
                                    Vec3(0.0f, 1.0f, 0.0f),
                                    0.66f, aspect_ratio, 0.0f);
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
    happly::PLYData ply_in("../bun_zipper.ply");
    std::vector<std::array<double, 3>> v_pos = ply_in.getVertexPositions();
    std::vector<std::vector<size_t>> f_index = ply_in.getFaceIndices<size_t>();

    Transform transform(Matrix4x4::Translate(0.02f, -0.1f, 0.f));
    transform.composite(Matrix4x4::Rotate(0.f, 1.f, 0.f, 0.1f));
    for (auto &v : v_pos) transform.apply(v);

    std::vector<Triangle> primitives;
    for (int i = 0; i < f_index.size(); i++) {
        const std::vector<size_t> &face = f_index[i];
        primitives.push_back(Triangle(Vec3(v_pos[face[0]][0], v_pos[face[0]][1], v_pos[face[0]][2]),
                                      Vec3(v_pos[face[1]][0], v_pos[face[1]][1], v_pos[face[1]][2]),
                                      Vec3(v_pos[face[2]][0], v_pos[face[2]][1], v_pos[face[2]][2]),
                                      NULL));
    }

    Bvh bvh(primitives);
}

/*
int main() {
    constexpr int WIDTH = 600;
    constexpr int HEIGHT = 600;
    constexpr int WIDTH_PER_BLOCK = 8;
    constexpr int HEIGHT_PER_BLOCK = 8;
    constexpr int NUM_SAMPLES = 10;

    constexpr float ASPECT_RATIO = float(WIDTH) / float(HEIGHT);

    constexpr int NUM_PIXELS = WIDTH * HEIGHT;
    Vec3 *d_fb;
    CHECK_CUDA(cudaMalloc(&d_fb, NUM_PIXELS * sizeof(Vec3)));

    constexpr int NUM_PRIMITIVES = 15;
    Primitive **d_primitive_ptrs;
    CHECK_CUDA(cudaMalloc(&d_primitive_ptrs, NUM_PRIMITIVES * sizeof(Primitive*)));

    Camera **d_camera_ptrs;
    CHECK_CUDA(cudaMalloc(&d_camera_ptrs, sizeof(Camera*)));

    create_world<<<1, 1>>>(d_primitive_ptrs, d_camera_ptrs, ASPECT_RATIO);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    curandState *d_rand_state;
    CHECK_CUDA(cudaMalloc(&d_rand_state, NUM_PIXELS * sizeof(curandState)));

    constexpr dim3 BLOCKS_PER_GRID((WIDTH + WIDTH_PER_BLOCK - 1) / WIDTH_PER_BLOCK,
                                   (HEIGHT + HEIGHT_PER_BLOCK - 1) / HEIGHT_PER_BLOCK);
    constexpr dim3 THREADS_PER_BLOCK(WIDTH_PER_BLOCK, HEIGHT_PER_BLOCK);

    render_init<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(WIDTH, HEIGHT, d_rand_state);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    render<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_fb, d_rand_state, NUM_SAMPLES,
                                                   WIDTH, HEIGHT,
                                                   d_primitive_ptrs, NUM_PRIMITIVES,
                                                   d_camera_ptrs);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    Vec3 *h_fb = new Vec3[NUM_PIXELS];
    CHECK_CUDA(cudaMemcpy(h_fb, d_fb, NUM_PIXELS * sizeof(Vec3), cudaMemcpyDeviceToHost));

    std::ofstream file("image.ppm");
    file << "P3\n" << WIDTH << ' ' << HEIGHT << "\n255\n";
    for (int j = HEIGHT - 1; j >= 0; j--) {
        for (int i = 0; i < WIDTH; i++) {
            size_t pixel_idx = j * WIDTH + i;
            float r = h_fb[pixel_idx].x;
            float g = h_fb[pixel_idx].y;
            float b = h_fb[pixel_idx].z;
            int ir = clamp(int(256.0f * r), 0, 255);
            int ig = clamp(int(256.0f * g), 0, 255);
            int ib = clamp(int(256.0f * b), 0, 255);
            file << ir << ' ' << ig << ' ' << ib << "\n";
        }
    }

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}
 */
