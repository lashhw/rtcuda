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
#include "hit_record.cuh"
#include "material.cuh"
#include "primitive.cuh"
#include "device_stack.cuh"
#include "bvh.cuh"
#include "camera.cuh"
#include "happly.h"

/*
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
            if (primitive_ptrs[j]->intersect(cur_ray, rec)) {
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
 */

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

__global__ void render(Bvh bvh, int width, int height, uchar3 *d_framebuffer,
                       Vec3 origin, Vec3 upper_left, Vec3 horizontal, Vec3 vertical) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;

    Vec3 lookat = upper_left + horizontal * ((float)i / width) + vertical * ((float)j / height);
    Ray ray(origin, lookat - origin);

    DeviceStack<Bvh::MAX_DEPTH> stack;
    Bvh::Intersection intersection;
    bool hit = bvh.traverse(stack, ray, intersection);
    Vec3 color = Vec3(0.f, 0.f, 0.f);
    if (hit) {
        color = bvh.d_primitives[intersection.primitive_index].n;
        color.unit_vector_inplace();
        color += Vec3(1.f, 1.f, 1.f);
        color /= 2;
    }

    d_framebuffer[width * j + i].x = 255.9f * color.x;
    d_framebuffer[width * j + i].y = 255.9f * color.y;
    d_framebuffer[width * j + i].z = 255.9f * color.z;
}

int main() {
    profiler.start("Reading scene");
    happly::PLYData ply_in("../dragon.ply");
    std::vector<std::array<double, 3>> v_pos = ply_in.getVertexPositions();
    std::vector<std::vector<size_t>> f_index = ply_in.getFaceIndices<size_t>();
    profiler.stop();
    std::cout << v_pos.size() << " vertices, " << f_index.size() << " faces" << std::endl;

    profiler.start("Transforming scene");
    Transform transform(Matrix4x4::Scale(0.02f, 0.02f, 0.02f));
    transform.composite(Matrix4x4::Rotate(1.f, 0.f, 0.f, deg_to_rad(90.f)));
    transform.composite(Matrix4x4::Rotate(0.f, 0.f, 1.f, deg_to_rad(-90.f)));
    transform.composite(Matrix4x4::Translate(0.2f, 0.3f, 0.78f));
    for (auto &v : v_pos) transform.apply(v);
    profiler.stop();

    profiler.start("Converting scene to triangles");
    std::vector<Triangle> primitives;
    for (int i = 0; i < f_index.size(); i++) {
        const std::vector<size_t> &face = f_index[i];
        primitives.push_back(Triangle(Vec3(v_pos[face[0]][0], v_pos[face[0]][1], v_pos[face[0]][2]),
                                      Vec3(v_pos[face[1]][0], v_pos[face[1]][1], v_pos[face[1]][2]),
                                      Vec3(v_pos[face[2]][0], v_pos[face[2]][1], v_pos[face[2]][2]),
                                      NULL));
    }
    profiler.stop();

    Bvh bvh(primitives);

    constexpr int WIDTH = 1366;
    constexpr int HEIGHT = 1024;
    constexpr int WIDTH_PER_BLOCK = 16;
    constexpr int HEIGHT_PER_BLOCK = 16;
    constexpr dim3 BLOCK_SIZE(WIDTH_PER_BLOCK, HEIGHT_PER_BLOCK);
    constexpr dim3 GRID_SIZE((WIDTH + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x,
                             (HEIGHT + BLOCK_SIZE.y - 1) / BLOCK_SIZE.y);

    // define camera
    Vec3 lookfrom(3.69558f, -3.46243f, 3.25463f);
    Vec3 lookat(3.04072f, -2.85176f, 2.80939f);
    Vec3 up(-0.317366f, 0.312466f, 0.895346f);
    float vfov = 28.8415038750464f;
    Vec3 w = (lookfrom - lookat).unit_vector();
    Vec3 v = (up - dot(up, w) * w).unit_vector();
    Vec3 u = cross(v, w);
    float viewpoint_height = 2.0f * tanf(deg_to_rad(vfov) / 2);
    float viewpoint_width = viewpoint_height * (float) WIDTH / (float) HEIGHT;
    Vec3 horizontal = viewpoint_width * u;
    Vec3 vertical = -viewpoint_height * v;
    Vec3 upper_left = lookfrom - w - horizontal / 2 - vertical / 2;

    profiler.start("Rendering");
    uchar3 *d_framebuffer;
    CHECK_CUDA(cudaMalloc(&d_framebuffer, WIDTH * HEIGHT * 3 * sizeof(unsigned char)));
    render<<<GRID_SIZE, BLOCK_SIZE>>>(bvh, WIDTH, HEIGHT, d_framebuffer,
                                      lookfrom, upper_left, horizontal, vertical);
    CHECK_CUDA(cudaGetLastError());
    uchar3 *h_framebuffer = new uchar3[WIDTH * HEIGHT];
    CHECK_CUDA(cudaMemcpy(h_framebuffer, d_framebuffer, WIDTH * HEIGHT * sizeof(uchar3), cudaMemcpyDeviceToHost));
    profiler.stop();

    profiler.start("Writing image");
    std::ofstream file("image.ppm");
    file << "P3\n" << WIDTH << " " << HEIGHT << "\n255\n";
    for (int j = 0; j < HEIGHT; j++) {
        for (int i = 0; i < WIDTH; i++) {
            int pixel_idx = j * WIDTH + i;
            file << (int)h_framebuffer[pixel_idx].x << " " << (int)h_framebuffer[pixel_idx].y << " " << (int)h_framebuffer[pixel_idx].z << '\n';
        }
    }
    file.close();
    profiler.stop();
}

