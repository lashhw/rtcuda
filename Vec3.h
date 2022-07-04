#ifndef RTCUDA_VEC3_H
#define RTCUDA_VEC3_H

#include <cmath>

class Vec3 {
public:
    __host__ __device__ Vec3() { }
    __host__ __device__ Vec3(float e0, float e1, float e2) : e{e0, e1, e2} { }

    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }

    __host__ __device__ inline Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) { return e[i]; }

    __host__ __device__ inline Vec3& operator+=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator-=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator*=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator/=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator*=(float t);
    __host__ __device__ inline Vec3& operator/=(float t);

    __host__ __device__ inline float length_squared() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
    __host__ __device__ inline float length() const { return sqrt(length_squared()); }
    __host__ __device__ inline Vec3 unit_vector() const;
    __host__ __device__ inline void to_unit_vector();

    float e[3];
};

__host__ __device__ inline Vec3 operator+(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline Vec3 operator-(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline Vec3 operator*(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline Vec3 operator/(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline Vec3 operator*(const Vec3 &v, float t) {
    return Vec3(v.e[0] * t, v.e[1] * t, v.e[2] * t);
}

__host__ __device__ inline Vec3 operator*(float t, const Vec3 &v) {
    return Vec3(v.e[0] * t, v.e[1] * t, v.e[2] * t);
}

__host__ __device__ inline Vec3 operator/(const Vec3 &v, float t) {
    return Vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ inline float dot (const Vec3 &v1, const Vec3 &v2) {
    return v1.e[0]*v2.e[0] + v1.e[1]*v2.e[1] + v1.e[2]*v2.e[2];
}

__host__ __device__ inline Vec3 cross(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1],
                v1.e[2]*v2.e[0] - v1.e[0]*v2.e[2],
                v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]);
}

__host__ __device__ inline Vec3 reflect(const Vec3 &v, const Vec3 &unit_normal) {
    return v - 2.0f*dot(v, unit_normal)*unit_normal;
}

__host__ __device__ inline Vec3 refract(const Vec3 &unit_v, const Vec3 &unit_normal, double eta_ratio) {
    double cos_theta = fmin(dot(-unit_v, unit_normal), 1.0f);
    Vec3 v_parallel = eta_ratio * (unit_v+cos_theta*unit_normal);
    Vec3 v_perp = -sqrt(1.0f-v_parallel.length_squared()) * unit_normal;
    return v_parallel + v_perp;
}

__host__ __device__ inline Vec3& Vec3::operator+=(const Vec3 &v2) {
    e[0] += v2.e[0];
    e[1] += v2.e[1];
    e[2] += v2.e[2];
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator-=(const Vec3 &v2) {
    e[0] -= v2.e[0];
    e[1] -= v2.e[1];
    e[2] -= v2.e[2];
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const Vec3 &v2) {
    e[0] *= v2.e[0];
    e[1] *= v2.e[1];
    e[2] *= v2.e[2];
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const Vec3 &v2) {
    e[0] /= v2.e[0];
    e[1] /= v2.e[1];
    e[2] /= v2.e[2];
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(float t) {
    e[0] /= t;
    e[1] /= t;
    e[2] /= t;
    return *this;
}

__host__ __device__ inline Vec3 Vec3::unit_vector() const {
    float k = 1.0f / this->length();
    return Vec3(e[0] * k, e[1] * k, e[2] * k);
}

__host__ __device__ inline void Vec3::to_unit_vector() {
    float k = 1.0f / this->length();
    e[0] = e[0] * k;
    e[1] = e[1] * k;
    e[2] = e[2] * k;
}

#endif //RTCUDA_VEC3_H
