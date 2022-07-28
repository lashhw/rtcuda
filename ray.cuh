#ifndef RTCUDA_RAY_CUH
#define RTCUDA_RAY_CUH

struct Ray {
    __device__ Ray() { }
    __device__ Ray(const Vec3 &origin,
                   const Vec3 &unit_d,
                   float tmax = FLT_MAX)
        : origin(origin), unit_d(unit_d), tmax(tmax) { }
    __device__ Vec3 at(float t) const { return origin + t * unit_d; }

    Vec3 origin;
    Vec3 unit_d;
    float tmax;
};

#endif //RTCUDA_RAY_CUH
