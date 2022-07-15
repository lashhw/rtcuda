#ifndef RTCUDA_RAY_H
#define RTCUDA_RAY_H

struct Ray {
    __device__ Ray() { }
    __device__ Ray(const Vec3 &origin,
                   const Vec3 &unit_direction,
                   float tmin = 1e-3f,  // TODO: parameterize this value
                   float tmax = std::numeric_limits<float>::max())
        : origin(origin), unit_direction(unit_direction), tmin(tmin), tmax(tmax) { }
    __device__ Vec3 at(float t) const { return origin + t * unit_direction; }

    Vec3 origin;
    Vec3 unit_direction;
    float tmin;
    float tmax;
};

#endif //RTCUDA_RAY_H
