#ifndef RTCUDA_LIGHT_CUH
#define RTCUDA_LIGHT_CUH

enum LightType {
    POINT_LIGHT,
    DIFFUSE_AREA_LIGHT
};

struct Light {
    Light() { }
    __device__ bool sample(const Intersection &isect, Vec3 &unit_wi, Vec3 &Li, float &t, float &pdf) const;
    __device__ float pdf(const Intersection &isect, const Vec3 &unit_wi) const;
    __device__ bool get_Le(const Vec3 &w, Vec3 &Le) const;
    __device__ bool is_delta() const { return type == POINT_LIGHT; }

    static Light make_point_light(const Vec3 &pos, const Vec3 &I);

    LightType type;
    Vec3 pos;  // for point light
    union {
        Vec3 I;  // for point light
        Vec3 L;  // for diffuse area light
    };
};

__device__ bool Light::sample(const Intersection &isect, Vec3 &unit_wi, Vec3 &Li, float &t, float &pdf) const {
    if (type == POINT_LIGHT) {
        unit_wi = pos - isect.p;
        t = unit_wi.length();
        Li = I / (t * t);
        unit_wi /= t;
        pdf = 1.f;
        return true;
    } else if (type == DIFFUSE_AREA_LIGHT) {
        // TODO: implement this
    }
}

__device__ float Light::pdf(const Intersection &isect, const Vec3 &unit_wi) const {
    if (type == POINT_LIGHT) {
        return 0.f;
    } else if (type == DIFFUSE_AREA_LIGHT) {
        // TODO: implement this
    }
}

__device__ bool Light::get_Le(const Vec3 &w, Vec3 &Le) const {
    return false;
}

Light Light::make_point_light(const Vec3 &pos, const Vec3 &I) {
    Light l;
    l.type = POINT_LIGHT;
    l.pos = pos;
    l.I = I;
    return l;
}

#endif //RTCUDA_LIGHT_CUH
