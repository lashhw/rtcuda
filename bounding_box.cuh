#ifndef RTCUDA_BOUNDING_BOX_CUH
#define RTCUDA_BOUNDING_BOX_CUH

struct BoundingBox {
    __host__ __device__ BoundingBox() { }
    __host__ __device__ BoundingBox(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax)
        : bounds { xmin, xmax, ymin, ymax, zmin, zmax } { }

    __host__ __device__ void extend(const BoundingBox &other);
    __host__ __device__ float half_area() const;
    __host__ __device__ void reset();

    __host__ __device__ static BoundingBox Empty() { return BoundingBox(FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX); }

    float bounds[6];  // [xmin, xmax, ymin, ymax, zmin, zmax]
};

__host__ __device__ void BoundingBox::extend(const BoundingBox &other) {
    bounds[0] = fminf(bounds[0], other.bounds[0]);
    bounds[1] = fmaxf(bounds[1], other.bounds[1]);
    bounds[2] = fminf(bounds[2], other.bounds[2]);
    bounds[3] = fmaxf(bounds[3], other.bounds[3]);
    bounds[4] = fminf(bounds[4], other.bounds[4]);
    bounds[5] = fmaxf(bounds[5], other.bounds[5]);
}

__host__ __device__ float BoundingBox::half_area() const {
    float e1 = bounds[1] - bounds[0];
    float e2 = bounds[3] - bounds[2];
    float e3 = bounds[5] - bounds[4];
    return (e1 + e2) * e3 + e1 * e2;
}

__host__ __device__ void BoundingBox::reset() {
    bounds[0] = bounds[2] = bounds[4] = FLT_MAX;
    bounds[1] = bounds[3] = bounds[5] = -FLT_MAX;
}

#endif //RTCUDA_BOUNDING_BOX_CUH
