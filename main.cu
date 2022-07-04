#include <iostream>
#include <fstream>
#include <cfloat>
#include <curand_kernel.h>

#include "Vec3.h"
#include "Ray.h"
#include "utility.h"
#include "HitRecord.h"
#include "Material.h"
#include "Primitive.h"
#include "Camera.h"

__device__ Vec3 get_color(const Ray &r, Primitive **world, curandState *local_rand_state) {
    Ray cur_ray = r;
    Vec3 cur_attenuation = Vec3(1.0, 1.0, 1.0);

    HitRecord tmp_rec;
    Vec3 tmp_attenuation;
    Ray tmp_r_out;

    for (int i = 0; i < 50; i++) {
        if ((*world)->hit(cur_ray, 0.001, FLT_MAX, tmp_rec)) {
            if (tmp_rec.mat_ptr->scatter(cur_ray, tmp_rec, local_rand_state, tmp_attenuation, tmp_r_out)) {
                cur_attenuation *= tmp_attenuation;
                cur_ray = tmp_r_out;
            } else {
                return Vec3(0.0, 0.0, 0.0);
            }
        }
        else {
            return cur_attenuation * Vec3(1.0, 1.0, 1.0);
        }
    }

    return Vec3(0.0, 0.0, 0.0);
}

__global__ void create_world(Primitive **d_list, Primitive **d_world, Camera **d_camera, float aspect_ratio) {
    if (threadIdx.x == 0 & blockIdx.x == 0) {
        *(d_list + 0) = new Sphere(Vec3(0, -100, -1), 100, new Lambertian(Vec3(0.8, 0.8, 0.0)));
        *(d_list + 1) = new Sphere(Vec3(0.5, 0.5, -1), 0.5, new Lambertian(Vec3(0.8, 0.3, 0.3)));
        *(d_list + 2) = new Sphere(Vec3(1.5, 0.5, -1), 0.5, new Metal(Vec3(0.8, 0.6, 0.2), 1.0));
        *(d_list + 3) = new Sphere(Vec3(-0.5, 0.5, -1), 0.5, new Dielectric(1.5));
        *d_world = new PrimitiveList(d_list, 4);
        *d_camera = new Camera(Vec3(0.5, 0.5, 1.0),
                               Vec3(0.5, 0.5, -1.0),
                               Vec3(0.0, 1.0, 0.0),
                               0.92, aspect_ratio, 0.5);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_idx = j * max_x + i;
    curand_init(1, pixel_idx, 0, &rand_state[pixel_idx]);
}

__global__ void render(Vec3 *fb, curandState *rand_state, int num_samples,
                       int max_x, int max_y,
                       Primitive **world, Camera **camera) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_idx = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_idx];
    Vec3 color(0, 0, 0);
    for (int s = 0; s < num_samples; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        Ray r = (*camera)->get_ray(u, v, &local_rand_state);
        color += get_color(r, world, &local_rand_state);
    }
    rand_state[pixel_idx] = local_rand_state;

    color /= num_samples;
    color[0] = sqrt(color[0]);
    color[1] = sqrt(color[1]);
    color[2] = sqrt(color[2]);
    fb[pixel_idx] = color;
}

int main() {
    const int WIDTH = 600;
    const int HEIGHT = 600;
    const int WIDTH_PER_BLOCK = 8;
    const int HEIGHT_PER_BLOCK = 8;
    const int NUM_SAMPLES = 100;

    float aspect_ratio = WIDTH / HEIGHT;

    int num_pixels = WIDTH * HEIGHT;
    size_t fb_size = num_pixels * sizeof(Vec3);
    Vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    Primitive **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 4*sizeof(Primitive*)));
    Primitive **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Primitive*)));
    Camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera*)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera, aspect_ratio);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

    dim3 blocks((WIDTH + WIDTH_PER_BLOCK - 1) / WIDTH_PER_BLOCK,
                (HEIGHT + HEIGHT_PER_BLOCK - 1) / HEIGHT_PER_BLOCK);
    dim3 threads(WIDTH_PER_BLOCK, HEIGHT_PER_BLOCK);

    render_init<<<blocks, threads>>>(WIDTH, HEIGHT, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(fb, d_rand_state, NUM_SAMPLES,
                                WIDTH, HEIGHT,
                                d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::ofstream file("image.ppm");
    file << "P3\n" << WIDTH << ' ' << HEIGHT << "\n255\n";
    for (int j = HEIGHT - 1; j >= 0; j--) {
        for (int i = 0; i < WIDTH; i++) {
            size_t pixel_idx = j * WIDTH + i;
            float r = fb[pixel_idx].x();
            float g = fb[pixel_idx].y();
            float b = fb[pixel_idx].z();
            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);
            file << ir << " " << ig << " " << ib << "\n";
        }
    }

    checkCudaErrors(cudaDeviceReset());
    return 0;
}
