#ifndef RTCUDA_RENDER_CUH
#define RTCUDA_RENDER_CUH

// direct lighting "Ld" estimator
__device__ Vec3 Ld(const Scene &scene, const Intersection &isect, const Vec3 &unit_wo,
                   const Material* d_mat, curandState &rand_state, DeviceStack &stack) {
    // uniformly choose one light source
    if (scene.num_lights == 0) return Vec3::make_zeros();
    int light_idx = min((int)(curand_uniform(&rand_state) * scene.num_lights),
                        scene.num_lights - 1);
    const Light &light = scene.d_lights[light_idx];

    Vec3 L = Vec3::make_zeros();

    // sample light source with MIS
    {
        Vec3 unit_wi;
        Vec3 Li;
        float light_t;
        float light_pdf;
        if (light.sample(isect, unit_wi, Li, light_t, light_pdf)) {
            Vec3 unit_n = dot(isect.unit_n, unit_wi) > 0.f ? isect.unit_n : -isect.unit_n;
            Vec3 f;
            if (d_mat->get_f(unit_wo, unit_wi, unit_n, f)) {
                f *= dot(unit_wi, unit_n);
                float scattering_pdf = d_mat->pdf(unit_wo, unit_wi, unit_n);

                // test whether the ray is occluded
                Ray shadow_ray = Ray::spawn_offset_ray(isect.p, unit_n, unit_wi, light_t);
                stack.reset();
                if (!scene.bvh.traverse(stack, shadow_ray)) {
                    if (light.is_delta()) {
                        L += f * Li / light_pdf;
                    } else {
                        float weight = power_heuristic(light_pdf, scattering_pdf);
                        L += f * Li * weight / light_pdf;
                    }
                }
            }
        }
    }

    // sample BSDF with MIS
    {
        if (!light.is_delta()) {
            Vec3 unit_n = isect.unit_n;
            Vec3 f, unit_wi;
            float scattering_pdf;
            if (d_mat->sample(unit_wo, rand_state, unit_n, f, unit_wi, scattering_pdf)) {
                f *= dot(unit_wi, unit_n);
                float weight = 1.f;

                if (!d_mat->is_specular()) {
                    float light_pdf = light.pdf(isect, unit_wi);
                    if (light_pdf == 0.f) return L;
                    weight = power_heuristic(scattering_pdf, light_pdf);
                }

                Vec3 Li;
                bool Li_is_valid = false;
                Ray light_ray = Ray::spawn_offset_ray(isect.p, unit_n, unit_wi);
                Intersection light_isect;
                Triangle *d_light_isect_primitive;
                if (scene.bvh.traverse(stack, light_ray, light_isect, d_light_isect_primitive)) {
                    Light *d_isect_area_light = d_light_isect_primitive->d_area_light;
                    if (d_isect_area_light == &light) {
                        Li = d_isect_area_light->L;
                        Li_is_valid = true;
                    }
                } else {
                    if (light.get_Le(unit_wi, Li)) Li_is_valid = true;
                }
                if (Li_is_valid) L += f * Li * weight / scattering_pdf;
            }
        }
    }

    return L * scene.num_lights;
}

// outgoing radiance "Lo" estimator
__device__ Vec3 Lo(const Ray &ray, const Scene &scene, curandState &rand_state, DeviceStack &stack) {
    // TODO: change to russian roulette
    static constexpr int MAX_BOUNCES = 10;

    Vec3 L = Vec3::make_zeros();
    Vec3 beta = Vec3::make_ones();
    Ray cur_ray = ray;
    bool last_bounce_is_specular = false;

    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        stack.reset();
        Intersection isect;
        Triangle *d_isect_primitive;
        bool hit_anything = scene.bvh.traverse(stack, cur_ray, isect, d_isect_primitive);

        // possibily add "Le" at intersection point
        if (bounce == 0 || last_bounce_is_specular) {
            if (hit_anything) {
                if (d_isect_primitive->d_area_light) {
                    L += beta * d_isect_primitive->d_area_light->L;
                }
            } else {
                // TODO: add infinite lights
            }
        }

        if (!hit_anything) break;

        // sample "Ld" (direct lighting) only if intersected material is not specular,
        // since it is a waste to track specular ray twice (once here, another is in next loop)
        Material *d_mat = d_isect_primitive->d_mat;
        if (!d_mat->is_specular()) {
            L += beta * Ld(scene, isect, cur_ray.unit_d, d_mat, rand_state, stack);
        }

        // sample BSDF to get new path direction
        Vec3 unit_n = isect.unit_n;
        Vec3 f, unit_wi;
        float pdf;
        bool scattered = d_mat->sample(cur_ray.unit_d, rand_state, unit_n, f, unit_wi, pdf);
        if (!scattered) break;
        beta *= f * dot(unit_wi, unit_n) / pdf;
        last_bounce_is_specular = d_mat->is_specular();
    }

    return L;
}

__global__ void render_init(int width, int height, curandState *d_rand_state) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;

    int pixel_idx = j * width + i;
    curand_init(1, pixel_idx, 0, &d_rand_state[pixel_idx]);
}

// TODO: eliminate fp_64 operations
__global__ void render(Camera<false> camera, Scene scene, int num_samples, int width, int height,
                       curandState *d_rand_state, Vec3 *d_framebuffer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;

    int pixel_idx = j * width + i;
    curandState local_rand_state = d_rand_state[pixel_idx];
    Vec3 L(0.f, 0.f, 0.f);
    DeviceStack stack;

    float inv_width = 1.f / width;
    float inv_height = 1.f / height;
    for (int s = 0; s < num_samples; s++) {
        float x = (i + curand_uniform(&local_rand_state)) * inv_width;
        float y = (j + curand_uniform(&local_rand_state)) * inv_height;
        Ray ray = camera.get_ray(x, y);
        L += Lo(ray, scene, local_rand_state, stack);
    }

    d_rand_state[pixel_idx] = local_rand_state;
    L /= num_samples;
    L.sqrt_inplace();
    d_framebuffer[pixel_idx] = L;
}

#endif //RTCUDA_RENDER_CUH
