cmake_minimum_required(VERSION 3.21)
project(rtcuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(rtcuda main.cu vec3.cuh ray.cuh primitive.cuh camera.cuh material.cuh utility.cuh transform.hpp matrix4x4.hpp bounding_box.cuh aabb_intersector.cuh bvh.cuh device_stack.cuh profiler.hpp intersection.hpp render.cuh light.cuh scene.cuh constant.hpp)

set_target_properties(rtcuda PROPERTIES
        CUDA_ARCHITECTURES "50;75")