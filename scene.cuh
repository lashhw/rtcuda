#ifndef RTCUDA_SCENE_CUH
#define RTCUDA_SCENE_CUH

struct Scene {
    Bvh bvh;
    int num_lights;
    Light *d_lights;
};

#endif //RTCUDA_SCENE_CUH
