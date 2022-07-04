#ifndef RTCUDA_PRIMITIVE_H
#define RTCUDA_PRIMITIVE_H

class Primitive {
public:
    __device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const = 0;
};

class Sphere : public Primitive {
public:
    __device__ Sphere() { }
    __device__ Sphere(const Vec3 &center, float radius, Material *mat_ptr) : center(center), radius(radius), mat_ptr(mat_ptr) { }
    __device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const;

    Vec3 center;
    float radius;
    Material *mat_ptr;
};

class PrimitiveList : public Primitive {
public:
    __device__ PrimitiveList() { }
    __device__ PrimitiveList(Primitive **list, int list_size) : list(list), list_size(list_size) { }
    __device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const;

    Primitive **list;
    int list_size;
};

__device__ bool Sphere::hit(const Ray& r, float t_min, float t_max, HitRecord &rec) const {
    Vec3 amc = r.origin - center;
    float a = dot(r.direction, r.direction);
    float b = 2.0f * dot(amc, r.direction);
    float c = dot(amc, amc) - radius * radius;
    float discriminant = b*b - 4.0f*a*c;

    if (discriminant > 0) {
        float root = (-b-sqrt(discriminant)) / (2*a);
        if (t_min < root && root < t_max) {
            rec.t = root;
            rec.p = r.at(rec.t);
            rec.outward_unit_normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        root = (-b+sqrt(discriminant)) / (2*a);
        if (t_min < root && root < t_max) {
            rec.t = root;
            rec.p = r.at(rec.t);
            rec.outward_unit_normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }

    return false;
}

__device__ bool PrimitiveList::hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const {
    bool hit_anything = false;
    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, t_min, t_max, rec)) {
            hit_anything = true;
            t_max = rec.t;
        }
    }
    return hit_anything;
}

#endif //RTCUDA_PRIMITIVE_H
