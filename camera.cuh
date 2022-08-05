#ifndef RTCUDA_CAMERA_CUH
#define RTCUDA_CAMERA_CUH

// TODO: implement depth of field
template <bool enable_depth_of_field>
struct Camera;

template <>
struct Camera<false> {
    Camera(Vec3 lookfrom, Vec3 lookat, Vec3 up, float vfov, float aspect_ratio);
    __device__ Ray get_ray(float x, float y) const;

    Vec3 lookfrom;
    Vec3 upper_left;
    Vec3 horizontal;
    Vec3 vertical;
};

Camera<false>::Camera(Vec3 lookfrom, Vec3 lookat, Vec3 up, float vfov, float aspect_ratio)
    : lookfrom(lookfrom) {
    float vfov_rad = deg_to_rad(vfov);

    float viewpoint_height = 2.f * tanf(vfov_rad * 0.5f);
    float viewpoint_width = viewpoint_height * aspect_ratio;

    Vec3 w = (lookfrom-lookat).unit_vector();
    Vec3 v = (up-dot(up, w)*w).unit_vector();
    Vec3 u = cross(v, w);

    horizontal = viewpoint_width * u;
    vertical = -viewpoint_height * v;
    upper_left = lookfrom - w - 0.5f * horizontal - 0.5f * vertical;
}

__device__ Ray Camera<false>::get_ray(float x, float y) const {
    Vec3 direction = upper_left + x * horizontal + y * vertical - lookfrom;
    return Ray(lookfrom, direction.unit_vector());
}

#endif //RTCUDA_CAMERA_CUH
