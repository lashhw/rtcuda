#ifndef RTCUDA_TRANSFORM_HPP
#define RTCUDA_TRANSFORM_HPP

struct Transform {
    Transform(const Matrix4x4 &matrix) : matrix(matrix) { }

    void composite(const Matrix4x4 &other);
    void apply(std::array<double, 3> &v);

    Matrix4x4 matrix;
};

void Transform::composite(const Matrix4x4 &other) {
    Matrix4x4 result;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result.data[i][j] = 0;
            for (int k = 0; k < 4; ++k) {
                result.data[i][j] += other.data[i][k] * matrix.data[k][j];
            }
        }
    }
    matrix = result;
}

void Transform::apply(std::array<double, 3> &v) {
    float new_x = matrix.data[0][0] * v[0] + matrix.data[0][1] * v[1] + matrix.data[0][2] * v[2] + matrix.data[0][3];
    float new_y = matrix.data[1][0] * v[0] + matrix.data[1][1] * v[1] + matrix.data[1][2] * v[2] + matrix.data[1][3];
           v[2] = matrix.data[2][0] * v[0] + matrix.data[2][1] * v[1] + matrix.data[2][2] * v[2] + matrix.data[2][3];

    v[0] = new_x;
    v[1] = new_y;
}

#endif //RTCUDA_TRANSFORM_HPP
