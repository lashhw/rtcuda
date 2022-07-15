cmake_minimum_required(VERSION 3.21)
project(rtcuda CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(rtcuda main.cu Vec3.h Ray.h Primitive.h Camera.h HitRecord.h Material.h utility.h)

set_target_properties(rtcuda PROPERTIES
        CUDA_ARCHITECTURES "50;75")