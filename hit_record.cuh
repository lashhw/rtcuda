#ifndef RTCUDA_HIT_RECORD_CUH
#define RTCUDA_HIT_RECORD_CUH

struct Material;  // forward declaration

struct HitRecord {
    float t;
    Vec3 p;
    Vec3 unit_outward_normal;
    Material *mat_ptr;
};

#endif //RTCUDA_HIT_RECORD_CUH
