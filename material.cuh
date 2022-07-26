#ifndef RTCUDA_MATERIAL_CUH
#define RTCUDA_MATERIAL_CUH

enum MaterialType {
    MATTE,
    MIRROR,
    METAL,
    GLASS,
    LIGHT
};

struct Material {
    Material() { }
    __device__ bool scatter(const Ray &r_in, const Intersection &isect,
                            curandState &rand_state, Vec3 &attenuation, Ray &r_out);
    __device__ bool emit(Vec3 &emitted_radiance);

    static Material create_matte(const Vec3 &albedo);
    static Material create_mirror(const Vec3 &albedo);
    static Material create_metal(const Vec3 &albedo, float fuzz);
    static Material create_glass(float index_of_refraction);
    static Material create_light(const Vec3 &emitted_radiance);

    union {
        Vec3 albedo;  // for matte, mirror, metal
        Vec3 emitted_radiance;  // for light
    };
    union {
        float fuzz;  // for metal
        float index_of_refraction;  // for glass
    };
    MaterialType type;
};

__device__ bool
Material::scatter(const Ray &r_in, const Intersection &isect,
                  curandState &rand_state, Vec3 &attenuation, Ray &r_out) {
    if (type == LIGHT) return false;

    Vec3 p = r_in.origin + isect.t * r_in.unit_d;
    Vec3 n = isect.n.unit_vector();
    bool intersect_front_face = dot(r_in.unit_d, n) < 0.f;
    if (!intersect_front_face) n = -n;

    if (type == MATTE || type == MIRROR || type == METAL) {
        attenuation = albedo;
        if (type == MATTE) {
            Vec3 direction = n + random_in_unit_sphere(rand_state);
            r_out = Ray(p, direction.unit_vector());
        } else {
            Vec3 unit_reflected = reflect(r_in.unit_d, n);
            if (type == MIRROR) {
                r_out = Ray(p, unit_reflected);
            } else {
                Vec3 direction = unit_reflected + fuzz * random_in_unit_sphere(rand_state);
                r_out = Ray(p, direction.unit_vector());
            }
        }
    } else if (type == GLASS) {
        attenuation = Vec3(1.f, 1.f, 1.f);

        float eta_ratio = intersect_front_face ? 1.f / index_of_refraction : index_of_refraction;

        float cos_theta = -dot(r_in.unit_d, n);
        float sin_theta = sqrtf(1.f - cos_theta * cos_theta);

        bool cannot_refract = eta_ratio * sin_theta > 1.f;

        // Schlick's approximation
        float r0 = __fdividef(1 - index_of_refraction, 1 + index_of_refraction);
        r0 = r0 * r0;
        float reflectance = r0 + (1 - r0) * __powf((1 - cos_theta), 5);

        Vec3 unit_reflect_direction = reflect(r_in.unit_d, n);
        Vec3 unit_refract_direction = refract(r_in.unit_d, n, eta_ratio);

        if (cannot_refract || curand_uniform(&rand_state) < reflectance)
            r_out = Ray(p, unit_reflect_direction);
        else
            r_out = Ray(p, unit_refract_direction);
    }

    return true;
}

__device__ bool Material::emit(Vec3 &emitted_radiance) {
    if (type != LIGHT) return false;
    emitted_radiance = this->emitted_radiance;
    return true;
}

Material Material::create_matte(const Vec3 &albedo) {
    Material m;
    m.albedo = albedo;
    m.type = MATTE;
    return m;
}

Material Material::create_mirror(const Vec3 &albedo) {
    Material m;
    m.albedo = albedo;
    m.type = MIRROR;
    return m;
}

Material Material::create_metal(const Vec3 &albedo, float fuzz) {
    Material m;
    m.albedo = albedo;
    m.fuzz = fuzz;
    m.type = METAL;
    return m;
}

Material Material::create_glass(float index_of_refraction) {
    Material m;
    m.index_of_refraction = index_of_refraction;
    m.type = GLASS;
    return m;
}

Material Material::create_light(const Vec3 &emitted_radiance) {
    Material m;
    m.emitted_radiance = emitted_radiance;
    m.type = LIGHT;
    return m;
}

#endif //RTCUDA_MATERIAL_CUH
