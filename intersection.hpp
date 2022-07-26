#ifndef RTCUDA_INTERSECTION_HPP
#define RTCUDA_INTERSECTION_HPP

struct Material;
struct Intersection {
    float t, u, v;
    Vec3 n;
    Material *d_mat;
};

#endif //RTCUDA_INTERSECTION_HPP
