#ifndef RTCUDA_VEC3_CUH
#define RTCUDA_VEC3_CUH

struct Vec3 {
    __host__ __device__ Vec3() { }
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) { }

    __host__ __device__ Vec3 operator-() const { return Vec3(-x, -y, -z); }

    __host__ __device__ Vec3& operator+=(const Vec3 &v2);
    __host__ __device__ Vec3& operator-=(const Vec3 &v2);
    __host__ __device__ Vec3& operator*=(const Vec3 &v2);
             __device__ Vec3& operator/=(const Vec3 &v2);
    __host__ __device__ Vec3& operator*=(float t);
    __host__ __device__ Vec3& operator/=(float t);

    __host__ __device__ float length_squared() const { return x * x + y * y + z * z; }
    __host__ __device__ float length() const { return sqrtf(x * x + y * y + z * z); }
    __host__ __device__ Vec3 unit_vector() const;
    __host__ __device__ void unit_vector_inplace();
    __host__ __device__ void sqrt_inplace();

    float x, y, z;
};

__host__ __device__ Vec3 operator+(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z);
}

__host__ __device__ Vec3 operator-(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z);
}

__host__ __device__ Vec3 operator*(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z);
}

         __device__ Vec3 operator/(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(__fdividef(v1.x, v2.x), __fdividef(v1.y, v2.y), __fdividef(v1.z, v2.z));
}

__host__ __device__ Vec3 operator*(const Vec3 &v, float t) {
    return Vec3(v.x * t, v.y * t, v.z * t);
}

__host__ __device__ Vec3 operator*(float t, const Vec3 &v) {
    return Vec3(v.x * t, v.y * t, v.z * t);
}

__host__ __device__ Vec3 operator/(const Vec3 &v, float t) {
    float inv_t = 1.f / t;
    return Vec3(v.x * inv_t, v.y * inv_t, v.z * inv_t);
}

__host__ __device__ float dot(const Vec3 &v1, const Vec3 &v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__ Vec3 cross(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1.y * v2.z - v1.z * v2.y,
                v1.z * v2.x - v1.x * v2.z,
                v1.x * v2.y - v1.y * v2.x);
}

__host__ __device__ Vec3 reflect(const Vec3 &v, const Vec3 &unit_normal) {
    return v - 2.f * dot(v, unit_normal) * unit_normal;
}

__host__ __device__ Vec3 refract(const Vec3 &unit_v, const Vec3 &unit_normal, double eta_ratio) {
    double cos_theta = -dot(unit_v, unit_normal);
    Vec3 v_parallel = eta_ratio * (unit_v+cos_theta*unit_normal);
    Vec3 v_perp = -sqrtf(1.f-v_parallel.length_squared()) * unit_normal;
    return v_parallel + v_perp;
}

__host__ __device__ Vec3& Vec3::operator+=(const Vec3 &v2) {
    x += v2.x;
    y += v2.y;
    z += v2.z;
    return *this;
}

__host__ __device__ Vec3& Vec3::operator-=(const Vec3 &v2) {
    x -= v2.x;
    y -= v2.y;
    z -= v2.z;
    return *this;
}

__host__ __device__ Vec3& Vec3::operator*=(const Vec3 &v2) {
    x *= v2.x;
    y *= v2.y;
    z *= v2.z;
    return *this;
}

         __device__ Vec3& Vec3::operator/=(const Vec3 &v2) {
    x = __fdividef(x, v2.x);
    y = __fdividef(y, v2.y);
    z = __fdividef(z, v2.z);
    return *this;
}

__host__ __device__ Vec3& Vec3::operator*=(float t) {
    x *= t;
    y *= t;
    z *= t;
    return *this;
}

__host__ __device__ Vec3& Vec3::operator/=(float t) {
    float inv_t = 1.f / t;
    x *= inv_t;
    y *= inv_t;
    z *= inv_t;
    return *this;
}

__host__ __device__ Vec3 Vec3::unit_vector() const {
    float inv_len = 1.f / this->length();
    return Vec3(x * inv_len, y * inv_len, z * inv_len);
}

__host__ __device__ void Vec3::unit_vector_inplace() {
    float inv_len = 1.f / this->length();
    x *= inv_len;
    y *= inv_len;
    z *= inv_len;
}

__host__ __device__ void Vec3::sqrt_inplace() {
    x = sqrtf(x);
    y = sqrtf(y);
    z = sqrtf(z);
}

#endif //RTCUDA_VEC3_CUH
