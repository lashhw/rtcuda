#ifndef RTCUDA_LIGHT_CUH
#define RTCUDA_LIGHT_CUH

enum LightType {
    POINT_LIGHT,
    AREA_LIGHT
};

struct Light {
    Light() { }
    __device__ bool sample_Li(const Intersection &isect, curandState &rand_state, Vec3 &unit_wi,
                              Vec3 &Li, float &t, float &pdf) const;
    __device__ float pdf_Li(const Intersection &isect, const Vec3 &unit_wi, const Vec3 &unit_n) const;
    __device__ bool get_Le(const Vec3 &w, Vec3 &Le) const;
    __device__ bool is_delta() const { return type == POINT_LIGHT; }

    static Light make_point_light(const Vec3 &pos, const Vec3 &I);
    static Light make_area_light(Triangle *d_shape, const Vec3 &L);

    LightType type;
    Vec3 pos;  // for point light
    Triangle *d_shape;  // for area light
    union {
        Vec3 I;  // for point light
        Vec3 L;  // for area light
    };
};

__device__ bool Light::sample_Li(const Intersection &isect, curandState &rand_state, Vec3 &unit_wi,
                                 Vec3 &Li, float &t, float &pdf) const {
    if (type == POINT_LIGHT) {
        unit_wi = pos - isect.p;
        t = unit_wi.length();
        Li = I / (t * t);
        unit_wi /= t;
        pdf = 1.f;
        return true;
    } else if (type == AREA_LIGHT) {
        Vec3 shape_p = d_shape->sample_p(rand_state, pdf);
        unit_wi = shape_p - isect.p;
        t = unit_wi.length();
        unit_wi /= t;
        Li = L;
    }
}

__device__ float Light::pdf_Li(const Intersection &isect, const Vec3 &unit_wi, const Vec3 &unit_n) const {
    if (type == POINT_LIGHT) {
        return 0.f;
    } else if (type == AREA_LIGHT) {
        Ray light_ray = Ray::spawn_offset_ray(isect.p, unit_n, unit_wi);
        Intersection isect_light;
        if (d_shape->intersect(light_ray, isect_light)) {
            Vec3 isect_light_p = d_shape->p(isect_light.u, isect_light.v);
            // TODO: for sphere, this line should be modified
            return (isect_light_p - isect.p).length_squared() / (d_shape->area() * abs_dot(d_shape->n, unit_wi));
        } else {
            return 0.f;
        }
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

Light Light::make_area_light(Triangle *d_shape, const Vec3 &L) {
    Light l;
    l.type = AREA_LIGHT;
    l.d_shape = d_shape;
    l.L = L;
    return l;
}

#endif //RTCUDA_LIGHT_CUH
