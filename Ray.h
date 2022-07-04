#ifndef RTCUDA_RAY_H
#define RTCUDA_RAY_H

class Ray {
public:
    __device__ Ray() { }
    __device__ Ray(const Vec3 &origin, const Vec3 &direction) : origin(origin), direction(direction) { }
    __device__ Vec3 at(float t) const { return origin + t * direction; }

    Vec3 origin;
    Vec3 direction;
};

#endif //RTCUDA_RAY_H
