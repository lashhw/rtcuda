#ifndef RTCUDA_PRIMITIVE_CUH
#define RTCUDA_PRIMITIVE_CUH

struct Primitive {
    Primitive() { }
    Primitive(Triangle *d_triangle, Material *d_mat, Light *d_area_light = NULL)
        : d_triangle(d_triangle), d_mat(d_mat), d_area_light(d_area_light) { }

    Triangle *d_triangle;
    Material *d_mat;
    Light *d_area_light;
};

#endif //RTCUDA_PRIMITIVE_CUH
