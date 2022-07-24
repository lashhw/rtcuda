#ifndef RTCUDA_PRIMITIVE_CUH
#define RTCUDA_PRIMITIVE_CUH

struct Primitive {
    __device__ virtual bool hit(const Ray &ray, HitRecord &rec) const = 0;
};

struct Sphere : public Primitive {
    __device__ Sphere() { }
    __device__ Sphere(const Vec3 &center, float radius, Material *mat_ptr)
        : center(center), radius(radius), mat_ptr(mat_ptr) { }

    __device__ virtual bool hit(const Ray &ray, HitRecord &rec) const override;

    Vec3 center;
    float radius;
    Material *mat_ptr;
};

struct Triangle : public Primitive {
    __host__ __device__ Triangle() { }
    __host__ __device__ Triangle(const Vec3 &p0, const Vec3 &p1, const Vec3 &p2, Material *mat_ptr)
        : p0(p0), e1(p0-p1), e2(p2-p0), n(cross(e1, e2)), mat_ptr(mat_ptr) { }

    __device__ Vec3 p1() const { return p0 - e1; }
    __device__ Vec3 p2() const { return p0 + e2; }
    __device__ Vec3 center() const { return (p0 + p1() + p2()) * (1.f / 3.f); }
    __device__ BoundingBox bounding_box() const;
    __device__ virtual bool hit(const Ray &ray, HitRecord &rec) const override;

    Vec3 p0, e1, e2, n;
    Material *mat_ptr;
};

__device__ bool Sphere::hit(const Ray& ray, HitRecord &rec) const {
    Vec3 oc = ray.origin - center;
    float a = dot(ray.unit_direction, ray.unit_direction);
    float b = 2.0f * dot(ray.unit_direction, oc);
    float c = dot(oc, oc) - radius * radius;
    float delta = b * b - 4.0f * a * c;

    if (delta >= 0.0f) {
        float inv = -0.5f / a;
        float sqrt_delta = sqrtf(delta);
        float root = (b + sqrt_delta) * inv;
        if (ray.tmin <= root && root <= ray.tmax) {
            rec.t = root;
            rec.p = ray.at(rec.t);
            // TODO: change divide to multiply?
            rec.unit_outward_normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        root = (b - sqrt_delta) * inv;
        if (ray.tmin <= root && root <= ray.tmax) {
            rec.t = root;
            rec.p = ray.at(rec.t);
            // TODO: change divide to multiply?
            rec.unit_outward_normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }

    return false;
}

__device__ BoundingBox Triangle::bounding_box() const {
    BoundingBox bbox;

    Vec3 p1_ = p1();
    Vec3 p2_ = p2();

    bbox.bounds[0] = fminf(p0.x, fminf(p1_.x, p2_.x));
    bbox.bounds[2] = fminf(p0.y, fminf(p1_.y, p2_.y));
    bbox.bounds[4] = fminf(p0.z, fminf(p1_.z, p2_.z));

    bbox.bounds[1] = fmaxf(p0.x, fmaxf(p1_.x, p2_.x));
    bbox.bounds[3] = fmaxf(p0.y, fmaxf(p1_.y, p2_.y));
    bbox.bounds[5] = fmaxf(p0.z, fmaxf(p1_.z, p2_.z));

    return bbox;
}

__device__ bool Triangle::hit(const Ray& ray, HitRecord &rec) const {
    Vec3 c = p0 - ray.origin;
    Vec3 r = cross(ray.unit_direction, c);
    float inv_det = 1.0f / dot(ray.unit_direction, n);

    float u = inv_det * dot(e2, r);
    float v = inv_det * dot(e1, r);

    if (u >= 0.0f && v >= 0.0f && (u + v) <= 1.0f) {
        float t = inv_det * dot(c, n);
        if (ray.tmin <= t && t <= ray.tmax) {
            rec.t = t;
            rec.p = ray.at(rec.t);
            // TODO: maybe -n is correct?
            rec.unit_outward_normal = n;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }

    return false;
}

#endif //RTCUDA_PRIMITIVE_CUH
