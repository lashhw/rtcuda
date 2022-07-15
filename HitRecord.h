#ifndef RTCUDA_HITRECORD_H
#define RTCUDA_HITRECORD_H

struct Material;  // forward declaration

struct HitRecord {
    float t;
    Vec3 p;
    Vec3 unit_outward_normal;
    Material *mat_ptr;
};

#endif //RTCUDA_HITRECORD_H
