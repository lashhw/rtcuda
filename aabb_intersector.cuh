#ifndef RTCUDA_AABB_INTERSECTOR_CUH
#define RTCUDA_AABB_INTERSECTOR_CUH

struct AABBIntersector {
    __device__ AABBIntersector(const Ray &ray);

    __device__ bool intersect(const BoundingBox &bbox, float &entry) const;

    int3 octant;
    Vec3 inverse_direction;
    Vec3 scaled_origin;
};

__device__ AABBIntersector::AABBIntersector(const Ray &ray) : octant { ray.unit_d.x < 0 ? 1 : 0,
                                                                       ray.unit_d.y < 0 ? 1 : 0,
                                                                       ray.unit_d.z < 0 ? 1 : 0 } {
    float inv_x = 1.f / ((fabsf(ray.unit_d.x) < FLT_EPSILON) ? copysignf(FLT_EPSILON, ray.unit_d.x) : ray.unit_d.x);
    float inv_y = 1.f / ((fabsf(ray.unit_d.y) < FLT_EPSILON) ? copysignf(FLT_EPSILON, ray.unit_d.y) : ray.unit_d.y);
    float inv_z = 1.f / ((fabsf(ray.unit_d.z) < FLT_EPSILON) ? copysignf(FLT_EPSILON, ray.unit_d.z) : ray.unit_d.z);
    inverse_direction = Vec3(inv_x, inv_y, inv_z);
    scaled_origin = -ray.origin * inverse_direction;
}

__device__ bool AABBIntersector::intersect(const BoundingBox &bbox, float &entry) const {
    float entry_x = inverse_direction.x * bbox.bounds[0 + octant.x] + scaled_origin.x;
    float entry_y = inverse_direction.y * bbox.bounds[2 + octant.y] + scaled_origin.y;
    float entry_z = inverse_direction.z * bbox.bounds[4 + octant.z] + scaled_origin.z;
    entry = fmaxf(entry_x, fmaxf(entry_y, entry_z));

    float exit_x = inverse_direction.x * bbox.bounds[1 - octant.x] + scaled_origin.x;
    float exit_y = inverse_direction.y * bbox.bounds[3 - octant.y] + scaled_origin.y;
    float exit_z = inverse_direction.z * bbox.bounds[5 - octant.z] + scaled_origin.z;
    float exit = fminf(exit_x, fminf(exit_y, exit_z));

    return entry <= exit;
}

#endif //RTCUDA_AABB_INTERSECTOR_CUH
