#ifndef RTCUDA_MATERIAL_H
#define RTCUDA_MATERIAL_H

class Material {
public:
    __device__ virtual bool scatter(const Ray &r_in, const HitRecord &rec, curandState *rand_state,
                                    Vec3 &attenuation, Ray &r_out) const = 0;
};

class Lambertian : public Material {
public:
    __device__ Lambertian(const Vec3 &albedo) : albedo(albedo) { }
    __device__ virtual bool scatter(const Ray &r_in, const HitRecord &rec, curandState *rand_state,
                                    Vec3 &attenuation, Ray &r_out) const override;

    Vec3 albedo;
};

class Metal : public Material {
public:
    __device__ Metal(const Vec3 &albedo, float fuzz) : albedo(albedo), fuzz(fuzz) { }
    __device__ virtual bool scatter(const Ray &r_in, const HitRecord &rec, curandState *rand_state,
                                    Vec3 &attenuation, Ray &r_out) const override;

    Vec3 albedo;
    float fuzz;
};

class Dielectric : public Material {
public:
    __device__ Dielectric(float index_of_refraction) : index_of_refraction(index_of_refraction) { }
    __device__ virtual bool scatter(const Ray &r_in, const HitRecord &rec, curandState *rand_state,
                                    Vec3 &attenuation, Ray &r_out) const override;

    float index_of_refraction;
};

__device__ bool Lambertian::scatter(const Ray &r_in, const HitRecord &rec, curandState *rand_state,
                                    Vec3 &attenuation, Ray &r_out) const {
    Vec3 n = dot(r_in.unit_direction, rec.unit_outward_normal) < 0.0f ? rec.unit_outward_normal : -rec.unit_outward_normal;
    Vec3 target = rec.p + n + random_in_unit_sphere(rand_state);
    attenuation = albedo;
    Vec3 direction = target - rec.p;
    r_out = Ray(rec.p, direction.unit_vector());
    return true;
}

__device__ bool Metal::scatter(const Ray &r_in, const HitRecord &rec, curandState *rand_state,
                               Vec3 &attenuation, Ray &r_out) const {
    Vec3 n = dot(r_in.unit_direction, rec.unit_outward_normal) < 0.0f ? rec.unit_outward_normal : -rec.unit_outward_normal;
    Vec3 unit_reflected = reflect(r_in.unit_direction, n);
    attenuation = albedo;
    Vec3 direction = unit_reflected + fuzz * random_in_unit_sphere(rand_state);
    r_out = Ray(rec.p, direction.unit_vector());
    return true;
}

__device__ bool Dielectric::scatter(const Ray &r_in, const HitRecord &rec, curandState *rand_state,
                                    Vec3 &attenuation, Ray &r_out) const {
    attenuation = Vec3(1.0f, 1.0f, 1.0f);

    bool from_air_to_dielectric = dot(r_in.unit_direction, rec.unit_outward_normal) < 0.0f;
    Vec3 unit_normal = from_air_to_dielectric ? rec.unit_outward_normal : -rec.unit_outward_normal;
    float eta_ratio = from_air_to_dielectric ? 1.0f / index_of_refraction : index_of_refraction;

    float cos_theta = -dot(r_in.unit_direction, unit_normal);
    float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);

    bool cannot_refract = eta_ratio * sin_theta > 1.0f;

    // Schlick's approximation
    float r0 = __fdividef(1-index_of_refraction, 1+index_of_refraction);
    r0 = r0 * r0;
    float reflectance = r0 + (1-r0) * __powf((1-cos_theta), 5);

    Vec3 unit_reflect_direction = reflect(r_in.unit_direction, unit_normal);
    Vec3 unit_refract_direction = refract(r_in.unit_direction, unit_normal, eta_ratio);

    Vec3 unit_r_out_direction;
    if (cannot_refract || curand_uniform(rand_state) < reflectance)
        unit_r_out_direction = unit_reflect_direction;
    else
        unit_r_out_direction = unit_refract_direction;

    r_out = Ray(rec.p, unit_r_out_direction);

    return true;
}

#endif //RTCUDA_MATERIAL_H
