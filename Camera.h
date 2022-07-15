#ifndef RTCUDA_CAMERA_H
#define RTCUDA_CAMERA_H

class Camera {
public:
    __device__ Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov_rad,
                      float aspect_ratio, float aperture_radius) : aperture_radius(aperture_radius) {
        float viewpoint_height = 2.0f * __tanf(vfov_rad/2);
        float viewpoint_width = viewpoint_height * aspect_ratio;

        Vec3 w = (lookfrom-lookat).unit_vector();
        v = (vup-dot(vup, w)*w).unit_vector();
        u = cross(v, w);

        camera_origin = lookfrom;
        float focus_dist = (lookat-lookfrom).length();
        horizontal = focus_dist * viewpoint_width * u;
        vertical = focus_dist * viewpoint_height * v;
        lower_left_corner = camera_origin - focus_dist * w - horizontal / 2 - vertical / 2;
    }

    __device__ Ray get_ray(float s, float t, curandState *rand_state) {
        float x, y;
        random_in_unit_disk(rand_state, x, y);
        x *= aperture_radius;
        y *= aperture_radius;

        Vec3 origin = camera_origin + u*x + v*y;
        Vec3 direction = lower_left_corner + s*horizontal + t*vertical - origin;
        return Ray(origin, direction.unit_vector());
    }

    Vec3 u;
    Vec3 v;
    Vec3 camera_origin;
    Vec3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    float aperture_radius;
};

#endif //RTCUDA_CAMERA_H
