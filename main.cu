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

#include "constant.hpp"
#include "profiler.hpp"
#include "vec3.cuh"
#include "matrix4x4.hpp"
#include "transform.hpp"
#include "utility.cuh"
#include "ray.cuh"
#include "bounding_box.cuh"
#include "aabb_intersector.cuh"
#include "intersection.hpp"
#include "material.cuh"
#include "light.cuh"
#include "primitive.cuh"
#include "device_stack.cuh"
#include "bvh.cuh"
#include "scene.cuh"
#include "camera.cuh"
#include "happly.h"
#include "render.cuh"

int main() {
    // create materials on host
    std::vector<Material> materials;
    materials.push_back(Material::make_matte(Vec3(0.65f, 0.05f, 0.05f)));
    materials.push_back(Material::make_matte(Vec3(0.12f, 0.45f, 0.15f)));
    materials.push_back(Material::make_matte(Vec3(0.73f, 0.73f, 0.73f)));
    materials.push_back(Material::make_matte(Vec3(0.62f, 0.57f, 0.54f)));

    // move materials to device
    int num_materials = materials.size();
    Material *d_materials;
    CHECK_CUDA(cudaMalloc(&d_materials, num_materials * sizeof(Material)));
    CHECK_CUDA(cudaMemcpy(d_materials, materials.data(), num_materials * sizeof(Material), cudaMemcpyHostToDevice));
    materials.clear();
    Material *d_red    = &d_materials[0];
    Material *d_green  = &d_materials[1];
    Material *d_white  = &d_materials[2];
    Material *d_brown  = &d_materials[3];

    // read bunny
    profiler.start("Reading bunny");
    happly::PLYData ply_in("../bun_zipper.ply");
    std::vector<std::array<double, 3>> v_pos = ply_in.getVertexPositions();
    std::vector<std::vector<size_t>> f_index = ply_in.getFaceIndices<size_t>();
    profiler.stop();
    std::cout << v_pos.size() << " vertices, " << f_index.size() << " faces" << std::endl;

    // transform bunny
    profiler.start("Transforming bunny");
    Transform transform(Matrix4x4::Translate(0.0946899f, -0.0329874f, -0.0587997f));
    transform.composite(Matrix4x4::Scale(2.f, 2.f, 2.f));
    transform.composite(Matrix4x4::Translate(0.3f, 0.f, -0.5f));
    for (auto &v : v_pos) transform.apply(v);
    profiler.stop();

    // convert bunny to triangles
    profiler.start("Converting bunny to triangles");
    std::vector<Triangle> primitives;
    for (int i = 0; i < f_index.size(); i++) {
        const std::vector<size_t> &face = f_index[i];
        primitives.push_back(Triangle(Vec3(v_pos[face[0]][0], v_pos[face[0]][1], v_pos[face[0]][2]),
                                      Vec3(v_pos[face[1]][0], v_pos[face[1]][1], v_pos[face[1]][2]),
                                      Vec3(v_pos[face[2]][0], v_pos[face[2]][1], v_pos[face[2]][2]),
                                      d_brown));
    }
    profiler.stop();

    // create walls
    primitives.push_back(Triangle(Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, -1.0f), Vec3(0.0f, 1.0f, -1.0f), d_red));
    primitives.push_back(Triangle(Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), Vec3(0.0f, 1.0f, -1.0f), d_red));
    primitives.push_back(Triangle(Vec3(1.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, -1.0f), Vec3(1.0f, 1.0f, -1.0f), d_green));
    primitives.push_back(Triangle(Vec3(1.0f, 0.0f, 0.0f), Vec3(1.0f, 1.0f, 0.0f), Vec3(1.0f, 1.0f, -1.0f), d_green));
    primitives.push_back(Triangle(Vec3(0.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, -1.0f), d_white));
    primitives.push_back(Triangle(Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, -1.0f), Vec3(1.0f, 0.0f, -1.0f), d_white));
    primitives.push_back(Triangle(Vec3(0.0f, 1.0f, 0.0f), Vec3(1.0f, 1.0f, 0.0f), Vec3(1.0f, 1.0f, -1.0f), d_white));
    primitives.push_back(Triangle(Vec3(0.0f, 1.0f, 0.0f), Vec3(0.0f, 1.0f, -1.0f), Vec3(1.0f, 1.0f, -1.0f), d_white));
    primitives.push_back(Triangle(Vec3(0.0f, 0.0f, -1.0f), Vec3(1.0f, 0.0f, -1.0f), Vec3(1.0f, 1.0f, -1.0f), d_white));
    primitives.push_back(Triangle(Vec3(0.0f, 0.0f, -1.0f), Vec3(0.0f, 1.0f, -1.0f), Vec3(1.0f, 1.0f, -1.0f), d_white));

    // create lights on host
    std::vector<Light> lights;
    lights.push_back(Light::make_point_light(Vec3(0.5, 0.9, -0.5), Vec3(1.f, 1.f, 1.f)));

    // move lights to device
    int num_lights = lights.size();
    Light *d_lights;
    CHECK_CUDA(cudaMalloc(&d_lights, num_lights * sizeof(Light)));
    CHECK_CUDA(cudaMemcpy(d_lights, lights.data(), num_lights * sizeof(Light), cudaMemcpyHostToDevice));
    lights.clear();

    // build bvh
    Bvh bvh(primitives);

    // create scene
    Scene scene = { bvh, num_lights, d_lights };

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
    const dim3 GRID_SIZE((WIDTH + WIDTH_PER_BLOCK - 1) / WIDTH_PER_BLOCK,
                         (HEIGHT + HEIGHT_PER_BLOCK - 1) / HEIGHT_PER_BLOCK);
    const dim3 BLOCK_SIZE(WIDTH_PER_BLOCK, HEIGHT_PER_BLOCK);
    curandState *d_rand_state;
    CHECK_CUDA(cudaMalloc(&d_rand_state, NUM_PIXELS * sizeof(curandState)));
    render_init<<<GRID_SIZE, BLOCK_SIZE>>>(WIDTH, HEIGHT, d_rand_state);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // allocate framebuffer on device
    Vec3 *d_framebuffer;
    CHECK_CUDA(cudaMalloc(&d_framebuffer, NUM_PIXELS * sizeof(Vec3)));

    // start rendering
    profiler.start("Rendering");
    constexpr int NUM_SAMPLES = 100;
    render<<<GRID_SIZE, BLOCK_SIZE>>>(camera, scene, NUM_SAMPLES, WIDTH, HEIGHT, d_rand_state, d_framebuffer);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    profiler.stop();

    // copy framebuffer to host
    auto h_framebuffer = std::make_unique<Vec3[]>(NUM_PIXELS);
    CHECK_CUDA(cudaMemcpy(h_framebuffer.get(), d_framebuffer, NUM_PIXELS * sizeof(Vec3), cudaMemcpyDeviceToHost));

    // write image
    profiler.start("Writing image");
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
    profiler.stop();

    return 0;
}