#ifndef RTCUDA_DEVICE_STACK_CUH
#define RTCUDA_DEVICE_STACK_CUH

struct DeviceStack {
    __device__ void push(int value) { data[size++] = value; }
    __device__ int pop() { return data[--size]; }
    __device__ bool empty() const { return size == 0; }
    __device__ void reset() { size = 0; }

    int data[BVH_MAX_DEPTH - 1];
    int size = 0;
};

#endif //RTCUDA_DEVICE_STACK_CUH
