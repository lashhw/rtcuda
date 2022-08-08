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
    __device__ float pdf_Li(const Intersection &isect, const Vec3 &unit_wi) const;
    __device__ bool get_Le(const Vec3 &w, Vec3 &Le) const;
    __device__ bool is_delta() const { return type == POINT_LIGHT; }

    static Light make_point_light(const Vec3 &pos, const Vec3 &I);
    static Light make_area_light(Triangle *d_triangle, const Vec3 &L);

    LightType type;
    Vec3 pos;  // for point light
    Triangle *d_triangle;  // for area light
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
        Vec3 triangle_p = d_triangle->sample_p(rand_state, pdf);
        unit_wi = triangle_p - isect.p;
        t = unit_wi.length();
        unit_wi /= t;
        Li = L;
        // convert pdf from area to solid angle
        pdf *= (triangle_p - isect.p).length_squared() / abs_dot(d_triangle->n.unit_vector(), unit_wi);
        return true;
    }
}

__device__ float Light::pdf_Li(const Intersection &isect, const Vec3 &unit_wi) const {
    if (type == POINT_LIGHT) {
        return 0.f;
    } else if (type == AREA_LIGHT) {
        Ray light_ray = Ray(isect.p, unit_wi);
        Intersection light_isect;
        if (d_triangle->intersect(light_ray, light_isect)) {
            Vec3 light_isect_p = d_triangle->p(light_isect.u, light_isect.v);
            Vec3 light_unit_n = d_triangle->n.unit_vector();
            return (light_isect_p - isect.p).length_squared() / (d_triangle->area() * abs_dot(light_unit_n, unit_wi));
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

Light Light::make_area_light(Triangle *d_triangle, const Vec3 &L) {
    Light l;
    l.type = AREA_LIGHT;
    l.d_triangle = d_triangle;
    l.L = L;
    return l;
}

#endif //RTCUDA_LIGHT_CUH
