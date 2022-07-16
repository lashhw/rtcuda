cmake_minimum_required(VERSION 3.21)
project(rtcuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(rtcuda main.cu Vec3.cuh Ray.cuh Primitive.cuh Camera.cuh HitRecord.cuh Material.cuh utility.cuh)

set_target_properties(rtcuda PROPERTIES
        CUDA_ARCHITECTURES "50;75")