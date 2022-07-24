#ifndef RTCUDA_MATRIX4X4_HPP
#define RTCUDA_MATRIX4X4_HPP

struct Matrix4x4 {
    Matrix4x4() { }
    Matrix4x4(float e00, float e01, float e02, float e03,
              float e10, float e11, float e12, float e13,
              float e20, float e21, float e22, float e23,
              float e30, float e31, float e32, float e33)
        : data { { e00, e01, e02, e03 },
                 { e10, e11, e12, e13 },
                 { e20, e21, e22, e23 },
                 { e30, e31, e32, e33 } } { }

    static Matrix4x4 Translate(float dx, float dy, float dz);
    static Matrix4x4 Scale(float sx, float sy, float sz);
    static Matrix4x4 Rotate(float axis_x, float axis_y, float axis_z, float theta);

    float data[4][4];
};

Matrix4x4 Matrix4x4::Translate(float dx, float dy, float dz) {
    return Matrix4x4(1.f, 0.f, 0.f, dx,
                     0.f, 1.f, 0.f, dy,
                     0.f, 0.f, 1.f, dz,
                     0.f, 0.f, 0.f, 1.f);
}

Matrix4x4 Matrix4x4::Scale(float sx, float sy, float sz) {
    return Matrix4x4(sx, 0.f, 0.f, 0.f,
                     0.f, sy, 0.f, 0.f,
                     0.f, 0.f, sz, 0.f,
                     0.f, 0.f, 0.f, 1.f);
}

Matrix4x4 Matrix4x4::Rotate(float axis_x, float axis_y, float axis_z, float theta) {
    const float &x = axis_x;
    const float &y = axis_y;
    const float &z = axis_z;
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);
    float cos_theta_1 = 1.f - cos_theta;
    return Matrix4x4(cos_theta + x * x * cos_theta_1,
                     x * y * cos_theta_1 - z * sin_theta,
                     x * z * cos_theta_1 + y * sin_theta,
                     0.f,
                     x * y * cos_theta_1 + z * sin_theta,
                     cos_theta + y * y * cos_theta_1,
                     y * z * cos_theta_1 - x * sin_theta,
                     0.f,
                     x * z * cos_theta_1 - y * sin_theta,
                     y * z * cos_theta_1 + x * sin_theta,
                     cos_theta + z * z * cos_theta_1,
                     0.f,
                     0.f, 0.f, 0.f, 1.f);
}

#endif //RTCUDA_MATRIX4X4_HPP
