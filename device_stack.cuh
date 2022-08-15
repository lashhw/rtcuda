#ifndef RTCUDA_DEVICE_STACK_CUH
#define RTCUDA_DEVICE_STACK_CUH

struct DeviceStack {
    __device__ ~DeviceStack() { if (d_global_data != NULL) free(d_global_data); }
    __device__ void push(int value);
    __device__ int pop();
    __device__ bool empty() const { return size == 0; }
    __device__ void reset() { size = 0; }

    int data[BVH_STACK_RF];
    int *d_global_data = NULL;
    int size = 0;
};

__device__ void DeviceStack::push(int value) {
    if (size < BVH_STACK_RF) data[size] = value;
    else {
        if (d_global_data == NULL) d_global_data = (int*)malloc(sizeof(int) * (BVH_MAX_DEPTH - 1 - BVH_STACK_RF));
        d_global_data[size - BVH_STACK_RF] = value;
    }
    size++;
}

__device__ int DeviceStack::pop() {
    size--;
    if (size < BVH_STACK_RF) return data[size];
    else return d_global_data[size - BVH_STACK_RF];
}

#endif //RTCUDA_DEVICE_STACK_CUH
