#ifndef RTCUDA_MATERIAL_H
#define RTCUDA_MATERIAL_H

class Material {
public:
    __device__ virtual bool scatter(const Ray &r_in, const HitRecord &rec, curandState *local_rand_state,
                                    Vec3 &attenuation, Ray &r_out) const = 0;
};

class Lambertian : public Material {
public:
    __device__ Lambertian(const Vec3 &albedo) : albedo(albedo) { }
    __device__ virtual bool scatter(const Ray &r_in, const HitRecord &rec, curandState *local_rand_state,
                                    Vec3 &attenuation, Ray &r_out) const override;

    Vec3 albedo;
};

class Metal : public Material {
public:
    __device__ Metal(const Vec3 &albedo, float fuzz) : albedo(albedo), fuzz(fuzz) { }
    __device__ virtual bool scatter(const Ray &r_in, const HitRecord &rec, curandState *local_rand_state,
                                    Vec3 &attenuation, Ray &r_out) const override;

    Vec3 albedo;
    float fuzz;
};

class Dielectric : public Material {
public:
    __device__ Dielectric(float index_of_refraction) : index_of_refraction(index_of_refraction) { }
    __device__ virtual bool scatter(const Ray &r_in, const HitRecord &rec, curandState *local_rand_state,
                                    Vec3 &attenuation, Ray &r_out) const override;

    float index_of_refraction;
};

__device__ bool Lambertian::scatter(const Ray &r_in, const HitRecord &rec, curandState *local_rand_state,
                                    Vec3 &attenuation, Ray &r_out) const {
    Vec3 target = rec.p + rec.outward_unit_normal + random_in_unit_sphere(local_rand_state);
    attenuation = albedo;
    r_out = Ray(rec.p, target-rec.p);
    return true;
}

__device__ bool Metal::scatter(const Ray &r_in, const HitRecord &rec, curandState *local_rand_state,
                               Vec3 &attenuation, Ray &r_out) const {
    Vec3 unit_reflected = reflect(r_in.direction.unit_vector(), rec.outward_unit_normal);
    attenuation = albedo;
    r_out = Ray(rec.p, unit_reflected + fuzz * random_in_unit_sphere(local_rand_state));
    return true;
}

__device__ bool Dielectric::scatter(const Ray &r_in, const HitRecord &rec, curandState *local_rand_state,
                                    Vec3 &attenuation, Ray &r_out) const {
    attenuation = Vec3(1.0, 1.0, 1.0);

    Vec3 unit_normal;
    float eta_ratio;
    if (dot(r_in.direction, rec.outward_unit_normal) > 0.0f) {
        unit_normal = -rec.outward_unit_normal;
        eta_ratio = index_of_refraction;
    } else {
        unit_normal = rec.outward_unit_normal;
        eta_ratio = 1.0f / index_of_refraction;
    }

    Vec3 unit_r_in_direction = r_in.direction.unit_vector();
    float cos_theta = -dot(unit_r_in_direction, unit_normal);
    float sin_theta = sqrt(1.0f - cos_theta*cos_theta);

    bool cannot_refract = eta_ratio * sin_theta > 1.0f;

    // Schlick's approximation
    float r0 = (1-index_of_refraction) / (1+index_of_refraction);
    r0 = r0 * r0;
    float reflectance = r0 + (1-r0)*pow((1-cos_theta), 5);

    Vec3 reflect_direction = reflect(unit_r_in_direction, unit_normal);
    Vec3 refract_direction = refract(unit_r_in_direction, unit_normal, eta_ratio);

    Vec3 r_out_direction;
    if (cannot_refract || curand_uniform(local_rand_state) < reflectance)
        r_out_direction = reflect_direction;
    else
        r_out_direction = refract_direction;

    r_out = Ray(rec.p, r_out_direction);

    return true;
}

#endif //RTCUDA_MATERIAL_H
