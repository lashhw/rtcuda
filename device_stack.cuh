#ifndef RTCUDA_DEVICE_STACK_CUH
#define RTCUDA_DEVICE_STACK_CUH

template <int STACK_SIZE>
struct DeviceStack {
    __device__ void push(int value) { data[size++] = value; }
    __device__ int pop() { return data[--size]; }
    __device__ bool empty() const { return size == 0; }

    int data[STACK_SIZE];
    int size = 0;
};

#endif //RTCUDA_DEVICE_STACK_CUH
