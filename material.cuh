#ifndef RTCUDA_MATERIAL_CUH
#define RTCUDA_MATERIAL_CUH

enum MaterialType {
    MATTE,
    MIRROR,
    GLASS
};

struct Material {
    Material() { }
    __device__ bool scatter(const Ray &r_in, const Intersection &isect,
                            const Vec3 &p0, const Vec3 &e1, const Vec3 &e2, const Vec3 &n,
                            curandState &rand_state, Vec3 &attenuation, Ray &r_out);
    __device__ bool get_f(const Vec3 &unit_wo, const Vec3 &unit_wi, const Vec3 &unit_n, Vec3 &f, float &pdf) const;
    __device__ bool sample_f(const Vec3 &unit_wo, curandState &rand_state, Vec3 &unit_n,
                             Vec3 &f, Vec3 &unit_wi, float &pdf) const;
    __device__ bool is_specular() const { return type == MIRROR || type == GLASS; }

    static Material make_matte(const Vec3 &albedo);
    static Material make_mirror(const Vec3 &albedo);
    static Material make_glass(float index_of_refraction);

    Vec3 albedo;  // for matte, mirror
    float index_of_refraction;  // for glass
    MaterialType type;
};

__device__ bool
Material::scatter(const Ray &r_in, const Intersection &isect,
                  const Vec3 &p0, const Vec3 &e1, const Vec3 &e2, const Vec3 &n,
                  curandState &rand_state, Vec3 &attenuation, Ray &r_out) {
    Vec3 p = p0 - isect.u * e1 + isect.v * e2;
    Vec3 out_n = n.unit_vector();
    bool intersect_front_face = dot(r_in.unit_d, out_n) < 0.f;
    if (!intersect_front_face) out_n = -out_n;

    if (type == MATTE || type == MIRROR) {
        attenuation = albedo;
        if (type == MATTE) {
            Vec3 direction = out_n + random_in_unit_sphere(rand_state);
            r_out = Ray(offset_ray_origin(p, out_n), direction.unit_vector());
        } else {
            Vec3 unit_reflected = reflect(r_in.unit_d, out_n);
            r_out = Ray(offset_ray_origin(p, out_n), unit_reflected);
        }
    } else if (type == GLASS) {
        attenuation = Vec3(1.f, 1.f, 1.f);

        float eta_ratio = intersect_front_face ? 1.f / index_of_refraction : index_of_refraction;

        float cos_theta = -dot(r_in.unit_d, out_n);
        float sin_theta = sqrtf(1.f - cos_theta * cos_theta);

        bool cannot_refract = eta_ratio * sin_theta > 1.f;

        // Schlick's approximation
        float r0 = __fdividef(1 - index_of_refraction, 1 + index_of_refraction);
        r0 = r0 * r0;
        float reflectance = r0 + (1 - r0) * __powf((1 - cos_theta), 5);

        Vec3 unit_reflect_direction = reflect(r_in.unit_d, out_n);
        Vec3 unit_refract_direction = refract(r_in.unit_d, out_n, eta_ratio);

        if (cannot_refract || curand_uniform(&rand_state) < reflectance)
            r_out = Ray(offset_ray_origin(p, out_n), unit_reflect_direction);
        else
            r_out = Ray(offset_ray_origin(p, out_n), unit_refract_direction);
    }

    return true;
}

Material Material::make_matte(const Vec3 &albedo) {
    Material m;
    m.albedo = albedo;
    m.type = MATTE;
    return m;
}

Material Material::make_mirror(const Vec3 &albedo) {
    Material m;
    m.albedo = albedo;
    m.type = MIRROR;
    return m;
}

Material Material::make_glass(float index_of_refraction) {
    Material m;
    m.index_of_refraction = index_of_refraction;
    m.type = GLASS;
    return m;
}

// return false if f is all zeros
__device__ bool Material::get_f(const Vec3 &unit_wo, const Vec3 &unit_wi, const Vec3 &unit_n,
                                Vec3 &f, float &pdf) const {
    if (type == MATTE) {
        if (same_hemisphere(unit_wo, unit_wi, unit_n)) {
            f = albedo * INV_PI;
            pdf = dot(unit_wi, unit_n) * INV_PI;
            return true;
        }
    }
    return false;
}

// this function also modify unit_n so that unit_n and unit_wi are in the same hemisphere
__device__ bool Material::sample_f(const Vec3 &unit_wo, curandState &rand_state, Vec3 &unit_n,
                                   Vec3 &f, Vec3 &unit_wi, float &pdf) const {
    if (dot(unit_wo, unit_n) > 0.f) unit_n = -unit_n;
    if (type == MATTE) {
        f = albedo * INV_PI;
        unit_wi = (unit_n + uniform_sample_sphere(rand_state)).unit_vector();
        pdf = dot(unit_wi, unit_n) * INV_PI;
        return true;
    } else if (type == MIRROR) {
        unit_wi = reflect(unit_wo, unit_n);
        f = albedo / dot(unit_wi, unit_n);
        pdf = 1.f;
        return true;
    } else if (type == GLASS) {
        unit_n = -unit_n;
        // TODO: implement this
    }
}

#endif //RTCUDA_MATERIAL_CUH
