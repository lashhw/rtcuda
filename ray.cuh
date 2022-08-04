#ifndef RTCUDA_RAY_CUH
#define RTCUDA_RAY_CUH

struct Ray {
    __device__ Ray() { }
    __device__ Ray(const Vec3 &origin,
                   const Vec3 &unit_d,
                   float tmax = FLT_MAX)
        : origin(origin), unit_d(unit_d), tmax(tmax) { }
    __device__ Vec3 at(float t) const { return origin + t * unit_d; }
    __device__ static Ray spawn_offset_ray(const Vec3 &origin, const Vec3 &unit_n, const Vec3 &unit_d);
    __device__ static Ray spawn_offset_ray(const Vec3 &origin, const Vec3 &unit_n, const Vec3 &unit_d, float tmax);

    Vec3 origin;
    Vec3 unit_d;
    float tmax;
};

__device__ Ray Ray::spawn_offset_ray(const Vec3 &origin, const Vec3 &unit_n, const Vec3 &unit_d) {
    return Ray(offset_ray_origin(origin, unit_n), unit_d);
}

__device__ Ray Ray::spawn_offset_ray(const Vec3 &origin, const Vec3 &unit_n, const Vec3 &unit_d, float tmax) {
    return Ray(offset_ray_origin(origin, unit_n), unit_d, tmax);
}

#endif //RTCUDA_RAY_CUH
