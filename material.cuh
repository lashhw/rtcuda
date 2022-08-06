#ifndef RTCUDA_MATERIAL_CUH
#define RTCUDA_MATERIAL_CUH

enum MaterialType {
    MATTE,
    MIRROR,
    GLASS
};

struct Material {
    Material() { }
    __device__ bool get_f(const Vec3 &unit_wo, const Vec3 &unit_wi, const Vec3 &unit_n, Vec3 &f, float &pdf) const;
    __device__ Vec3 sample_f(const Vec3 &unit_wo, curandState &rand_state, Vec3 &unit_n, Vec3 &unit_wi, float &pdf) const;
    __device__ bool is_specular() const { return type == MIRROR || type == GLASS; }

    static Material make_matte(const Vec3 &albedo);
    static Material make_mirror(const Vec3 &albedo);
    static Material make_glass(float index_of_refraction);

    Vec3 albedo;  // for matte, mirror
    float index_of_refraction;  // for glass
    MaterialType type;
};

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
__device__ Vec3 Material::sample_f(const Vec3 &unit_wo, curandState &rand_state, Vec3 &unit_n,
                                   Vec3 &unit_wi, float &pdf) const {
    if (type == MATTE || type == MIRROR) {
        if (dot(unit_wo, unit_n) > 0.f) unit_n = -unit_n;
        if (type == MATTE) {
            unit_wi = (unit_n + uniform_sample_sphere(rand_state)).unit_vector();
            pdf = dot(unit_wi, unit_n) * INV_PI;
            return albedo * INV_PI;
        } else if (type == MIRROR) {
            unit_wi = reflect(unit_wo, unit_n);
            pdf = 1.f;
            return albedo / dot(unit_wi, unit_n);
        }
    } else if (type == GLASS) {
        float cos_theta = dot(unit_wo, unit_n);
        bool intersect_front_face = cos_theta < 0.f;
        if (intersect_front_face) cos_theta = -cos_theta;
        float inv_cos_theta = 1.f / cos_theta;
        float eta_ratio = intersect_front_face ? 1.f / index_of_refraction : index_of_refraction;

        float sin_theta = sqrtf(1.f - cos_theta * cos_theta);
        bool cannot_refract = eta_ratio * sin_theta > 1.f;
        if (cannot_refract) {
            if (!intersect_front_face) unit_n = -unit_n;
            unit_wi = reflect(unit_wo, unit_n);
            pdf = 1.f;
            return Vec3(inv_cos_theta);
        }

        // Schlick's approximation
        float r0 = (1 - index_of_refraction) / (1 + index_of_refraction);
        r0 = r0 * r0;
        float reflectance = r0 + (1 - r0) * powf((1 - cos_theta), 5);

        if (curand_uniform(&rand_state) < reflectance) {
            // reflect
            if (!intersect_front_face) unit_n = -unit_n;
            unit_wi = reflect(unit_wo, unit_n);
            pdf = reflectance;
            return Vec3(pdf * inv_cos_theta);
        } else {
            // refract
            if (!intersect_front_face) unit_n = -unit_n;
            unit_wi = refract(unit_wo, unit_n, eta_ratio, cos_theta);
            unit_n = -unit_n;
            pdf = 1.f - reflectance;
            return Vec3(pdf * eta_ratio * eta_ratio / dot(unit_wi, unit_n));
        }
    }
}

#endif //RTCUDA_MATERIAL_CUH
