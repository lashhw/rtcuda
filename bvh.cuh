#ifndef RTCUDA_BVH_CUH
#define RTCUDA_BVH_CUH

struct Bvh {
    struct Node {
        __device__ bool is_leaf() const { return num_primitives > 0; }

        BoundingBox bbox;
        int num_primitives;  // 0 for non-leaf node
        union {
            int left_node_index;  // used when node != leaf
            int first_primitive_index;  // used when node == leaf
        };
    };

    Bvh(int num_primitives, Triangle *d_primitives);

    static constexpr int MAX_DEPTH = 64;

    int num_primitives;
    Triangle *d_primitives;
    int num_nodes;
    Node *d_nodes;
};

struct MergeBoundingBox {
    __device__ BoundingBox operator()(const BoundingBox &bbox1, const BoundingBox &bbox2) const {
        return BoundingBox::merge(bbox1, bbox2);
    }
};

struct MarksPredicate {
    __device__ bool operator()(const int &x) const {
        return d_marks[x];
    }
    const bool *d_marks;
};

__global__ void fill_bboxes_and_centers(const Triangle *d_primitives, int num_primitives,
                                        BoundingBox *d_bboxes, Vec3 *d_centers) {
    int primitive_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (primitive_index < num_primitives) {
        d_bboxes[primitive_index] = d_primitives[primitive_index].bounding_box();
        d_centers[primitive_index] = d_primitives[primitive_index].center();
    }
}

__global__ void fill_keys_by_x(const Vec3 *d_centers, int num_primitives, float *d_keys) {
    int primitive_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (primitive_index < num_primitives)
        d_keys[primitive_index] = d_centers[primitive_index].x;
}

__global__ void fill_keys_by_y(const Vec3 *d_centers, int num_primitives, float *d_keys) {
    int primitive_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (primitive_index < num_primitives)
        d_keys[primitive_index] = d_centers[primitive_index].y;
}

__global__ void fill_keys_by_z(const Vec3 *d_centers, int num_primitives, float *d_keys) {
    int primitive_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (primitive_index < num_primitives)
        d_keys[primitive_index] = d_centers[primitive_index].z;
}

__global__ void make_leaf(int node_index, int curr_num_primitives, int begin, Bvh::Node *d_nodes) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        d_nodes[node_index].num_primitives = curr_num_primitives;
        d_nodes[node_index].first_primitive_index = begin;
    }
}

__global__ void make_internal_node(int node_index, int left_node_index, Bvh::Node *d_nodes) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        d_nodes[node_index].num_primitives = 0;
        d_nodes[node_index].left_node_index = left_node_index;
    }
}

__global__ void fill_costs(const BoundingBox *d_bboxes_tmp, int begin, int end, float *d_costs) {
    int primitive_index = blockIdx.x * blockDim.x + threadIdx.x + begin;
    if (primitive_index < end - 1)
        d_costs[primitive_index] = d_bboxes_tmp[primitive_index].half_area() * (primitive_index + 1 - begin);
}

__global__ void update_costs(const BoundingBox *d_bboxes_tmp, int begin, int end, float *d_costs) {
    int primitive_index = blockIdx.x * blockDim.x + threadIdx.x + begin;
    if (primitive_index < end - 1)
        d_costs[primitive_index] += d_bboxes_tmp[primitive_index + 1].half_area() * (end - primitive_index - 1);
}

__global__ void fill_curr_half_area(const Bvh::Node *d_nodes, int node_index, float *d_curr_half_area) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *d_curr_half_area = d_nodes[node_index].bbox.half_area();
}

__global__ void fill_bboxes_tmp(const BoundingBox *d_bboxes, const int *d_sorted_references_best_axis,
                                int begin, int end, BoundingBox *d_bboxes_tmp) {
    int primitive_index = blockIdx.x * blockDim.x + threadIdx.x + begin;
    if (primitive_index < end)
        d_bboxes_tmp[primitive_index] = d_bboxes[d_sorted_references_best_axis[primitive_index]];
}

__global__ void fill_marks_left(const int *d_sorted_references_best_axis, int begin, int best_split_index,
                                bool *d_marks) {
    int primitive_index = blockIdx.x * blockDim.x + threadIdx.x + begin;
    if (primitive_index < best_split_index)
        d_marks[d_sorted_references_best_axis[primitive_index]] = true;
}

__global__ void fill_marks_right(const int *d_sorted_references_best_axis, int best_split_index, int end,
                                 bool *d_marks) {
    int primitive_index = blockIdx.x * blockDim.x + threadIdx.x + best_split_index;
    if (primitive_index < end)
        d_marks[d_sorted_references_best_axis[primitive_index]] = false;
}

__global__ void rearrange_primitives(const int *d_sorted_references_0, const Triangle *d_primitives_tmp,
                                     int num_primitives, Triangle *d_primitives) {
    int primitive_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (primitive_index < num_primitives) {
        d_primitives[primitive_index] = d_primitives_tmp[d_sorted_references_0[primitive_index]];
    }
}

// TODO: add CHECK_CUDA for cub
Bvh::Bvh(int num_primitives, Triangle *d_primitives) : num_primitives(num_primitives), d_primitives(d_primitives) {
    // TODO: try different parameters
    // kernel execution configuration
    const int THREADS_PER_BLOCK = 256;
    auto blocks_per_grid = [](int num_threads) -> int { return (num_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; };

    // allocate temporary memory for BVH construction
    BoundingBox *d_bboxes;
    BoundingBox *d_bboxes_tmp;
    Vec3 *d_centers;
    float *d_costs;
    float *d_unsorted_keys;
    float *d_sorted_keys;
    int *d_unsorted_references;
    int *d_sorted_references[3];
    bool *d_marks;
    Triangle *d_primitives_tmp;
    void *d_temp_storage;
    CHECK_CUDA(cudaMalloc(&d_bboxes, num_primitives * sizeof(BoundingBox)));
    CHECK_CUDA(cudaMalloc(&d_bboxes_tmp, num_primitives * sizeof(BoundingBox)));
    CHECK_CUDA(cudaMalloc(&d_centers, num_primitives * sizeof(Vec3)));
    CHECK_CUDA(cudaMalloc(&d_costs, num_primitives * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_unsorted_keys, num_primitives * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sorted_keys, num_primitives * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_unsorted_references, num_primitives * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_sorted_references[0], num_primitives * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_sorted_references[1], num_primitives * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_sorted_references[2], num_primitives * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_marks, num_primitives * sizeof(bool)));
    CHECK_CUDA(cudaMalloc(&d_primitives_tmp, num_primitives * sizeof(Triangle)));

    // allocate memory for BVH d_nodes
    num_nodes = 1;
    CHECK_CUDA(cudaMalloc(&d_nodes, 2 * num_primitives * sizeof(Node)));  // TODO: free unused node after construction

    // fill d_bboxes and d_centers
    fill_bboxes_and_centers<<<blocks_per_grid(num_primitives), THREADS_PER_BLOCK>>>(d_primitives, num_primitives,
                                                                                    d_bboxes, d_centers);
    CHECK_CUDA(cudaGetLastError());

    // fill d_nodes[0].bbox by merging all d_bboxes
    d_temp_storage = NULL;
    size_t temp_storage_bytes;
    MergeBoundingBox merge_bounding_box;
    CHECK_CUDA(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_bboxes, &d_nodes[0].bbox,
                                         num_primitives, merge_bounding_box, BoundingBox::Empty()));
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CHECK_CUDA(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_bboxes, &d_nodes[0].bbox,
                                         num_primitives, merge_bounding_box, BoundingBox::Empty()));

    thrust::device_ptr<int> unsorted_references_dev_ptr(d_unsorted_references);

    // fill d_sorted_references[0]
    thrust::sequence(unsorted_references_dev_ptr, unsorted_references_dev_ptr + num_primitives);
    fill_keys_by_x<<<blocks_per_grid(num_primitives), THREADS_PER_BLOCK>>>(d_centers, num_primitives, d_unsorted_keys);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaFree(d_temp_storage));
    d_temp_storage = NULL;
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_unsorted_keys, d_sorted_keys,
                                               d_unsorted_references, d_sorted_references[0], num_primitives));
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_unsorted_keys, d_sorted_keys,
                                               d_unsorted_references, d_sorted_references[0], num_primitives));

    // fill d_sorted_references[1]
    thrust::sequence(unsorted_references_dev_ptr, unsorted_references_dev_ptr + num_primitives);
    fill_keys_by_y<<<blocks_per_grid(num_primitives), THREADS_PER_BLOCK>>>(d_centers, num_primitives, d_unsorted_keys);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaFree(d_temp_storage));
    d_temp_storage = NULL;
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_unsorted_keys, d_sorted_keys,
                                               d_unsorted_references, d_sorted_references[1], num_primitives));
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_unsorted_keys, d_sorted_keys,
                                               d_unsorted_references, d_sorted_references[1], num_primitives));

    // fill d_sorted_references[2]
    thrust::sequence(unsorted_references_dev_ptr, unsorted_references_dev_ptr + num_primitives);
    fill_keys_by_z<<<blocks_per_grid(num_primitives), THREADS_PER_BLOCK>>>(d_centers, num_primitives, d_unsorted_keys);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaFree(d_temp_storage));
    d_temp_storage = NULL;
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_unsorted_keys, d_sorted_keys,
                                               d_unsorted_references, d_sorted_references[2], num_primitives));
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_unsorted_keys, d_sorted_keys,
                                               d_unsorted_references, d_sorted_references[2], num_primitives));

    // declare stack used for constructing BVH
    std::stack<std::array<int, 4>> stack;
    int node_index = 0;
    int begin = 0;
    int end = num_primitives;
    int depth = 0;
    auto check_and_update_and_pop_stack = [&]() -> bool {
        if (stack.empty()) return false;
        node_index = stack.top()[0];
        begin = stack.top()[1];
        end = stack.top()[2];
        depth = stack.top()[3];
        stack.pop();
        return true;
    };

    // recursively build BVH (recursion is implemented by stack)
    while (true) {
        int curr_num_primitives = end - begin;

        if (curr_num_primitives <= 1 || depth >= MAX_DEPTH) {
            make_leaf<<<1, 1>>>(node_index, curr_num_primitives, begin, d_nodes);
            CHECK_CUDA(cudaGetLastError());
            if (check_and_update_and_pop_stack()) continue;
            else break;
        }

        float best_cost = FLT_MAX;
        int best_axis = -1;
        int best_split_index = -1;

        thrust::device_ptr<BoundingBox> bboxes_dev_ptr(d_bboxes);
        thrust::device_ptr<BoundingBox> bboxes_tmp_dev_ptr(d_bboxes_tmp);
        thrust::reverse_iterator<thrust::device_ptr<BoundingBox>> bboxes_tmp_rev_iter(bboxes_tmp_dev_ptr + end);
        for (int axis = 0; axis < 3; axis++) {
            thrust::device_ptr<int> sorted_references_axis_dev_ptr(d_sorted_references[axis]);
            thrust::reverse_iterator<thrust::device_ptr<int>> sorted_references_axis_rev_iter(sorted_references_axis_dev_ptr + end);

            thrust::inclusive_scan(thrust::make_permutation_iterator(bboxes_dev_ptr, sorted_references_axis_dev_ptr + begin),
                                   thrust::make_permutation_iterator(bboxes_dev_ptr, sorted_references_axis_dev_ptr + end - 1),
                                   bboxes_tmp_dev_ptr + begin,
                                   merge_bounding_box);
            fill_costs<<<blocks_per_grid(curr_num_primitives - 1), THREADS_PER_BLOCK>>>(d_bboxes_tmp, begin, end, d_costs);
            CHECK_CUDA(cudaGetLastError());

            thrust::inclusive_scan(thrust::make_permutation_iterator(bboxes_dev_ptr, sorted_references_axis_rev_iter),
                                   thrust::make_permutation_iterator(bboxes_dev_ptr, sorted_references_axis_rev_iter + curr_num_primitives - 1),
                                   bboxes_tmp_rev_iter,
                                   merge_bounding_box);
            update_costs<<<blocks_per_grid(curr_num_primitives - 1), THREADS_PER_BLOCK>>>(d_bboxes_tmp, begin, end, d_costs);
            CHECK_CUDA(cudaGetLastError());

            CHECK_CUDA(cudaFree(d_temp_storage));
            d_temp_storage = NULL;
            cub::KeyValuePair<int, float> *d_min_pair;
            CHECK_CUDA(cudaMalloc(&d_min_pair, sizeof(cub::KeyValuePair<int, float>)));
            CHECK_CUDA(cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_costs + begin, d_min_pair, curr_num_primitives - 1));
            CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
            CHECK_CUDA(cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_costs + begin, d_min_pair, curr_num_primitives - 1));

            cub::KeyValuePair<int, float> min_pair;
            CHECK_CUDA(cudaMemcpy(&min_pair, d_min_pair, sizeof(cub::KeyValuePair<int, float>), cudaMemcpyDeviceToHost));

            if (min_pair.value < best_cost) {
                best_cost = min_pair.value;
                best_axis = axis;
                best_split_index = begin + min_pair.key + 1;
            }
        }
        assert(best_axis != -1 && best_split_index != -1);

        float curr_half_area;
        fill_curr_half_area<<<1, 1>>>(d_nodes, node_index, &d_costs[0]);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpy(&curr_half_area, &d_costs[0], sizeof(float), cudaMemcpyDeviceToHost));
        float max_split_cost = curr_half_area * (curr_num_primitives - 1);

        if (best_cost >= max_split_cost) {
            make_leaf<<<1, 1>>>(node_index, curr_num_primitives, begin, d_nodes);
            CHECK_CUDA(cudaGetLastError());
            if (check_and_update_and_pop_stack()) continue;
            else break;
        }

        int left_node_index = num_nodes;
        int right_node_index = left_node_index + 1;

        fill_bboxes_tmp<<<blocks_per_grid(curr_num_primitives), THREADS_PER_BLOCK>>>(d_bboxes, d_sorted_references[best_axis],
                                                                                     begin, end, d_bboxes_tmp);
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaFree(d_temp_storage));
        d_temp_storage = NULL;
        CHECK_CUDA(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_bboxes_tmp + begin,
                                             &d_nodes[left_node_index].bbox, best_split_index - begin,
                                             merge_bounding_box, BoundingBox::Empty()));
        CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        CHECK_CUDA(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_bboxes_tmp + begin,
                                             &d_nodes[left_node_index].bbox, best_split_index - begin,
                                             merge_bounding_box, BoundingBox::Empty()));

        CHECK_CUDA(cudaFree(d_temp_storage));
        d_temp_storage = NULL;
        CHECK_CUDA(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_bboxes_tmp + best_split_index,
                                             &d_nodes[right_node_index].bbox, end - best_split_index,
                                             merge_bounding_box, BoundingBox::Empty()));
        CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        CHECK_CUDA(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_bboxes_tmp + best_split_index,
                                             &d_nodes[right_node_index].bbox, end - best_split_index,
                                             merge_bounding_box, BoundingBox::Empty()));

        fill_marks_left<<<blocks_per_grid(best_split_index-begin), THREADS_PER_BLOCK>>>(d_sorted_references[best_axis],
                                                                                        begin, best_split_index, d_marks);
        fill_marks_right<<<blocks_per_grid(end-best_split_index), THREADS_PER_BLOCK>>>(d_sorted_references[best_axis],
                                                                                       best_split_index, end, d_marks);
        CHECK_CUDA(cudaGetLastError());

        int other_axis[2] = { (best_axis + 1) % 3, (best_axis + 2) % 3 };
        MarksPredicate marks_predicate = { d_marks };

        thrust::device_ptr<int> sorted_references_other_axis_0_dev_ptr(d_sorted_references[other_axis[0]]);
        thrust::device_ptr<int> sorted_references_other_axis_1_dev_ptr(d_sorted_references[other_axis[1]]);
        thrust::stable_partition(sorted_references_other_axis_0_dev_ptr + begin,
                                 sorted_references_other_axis_0_dev_ptr + end, marks_predicate);
        thrust::stable_partition(sorted_references_other_axis_1_dev_ptr + begin,
                                 sorted_references_other_axis_1_dev_ptr + end, marks_predicate);

        num_nodes += 2;
        make_internal_node<<<1, 1>>>(node_index, left_node_index, d_nodes);

        int left_size = best_split_index - begin;
        int right_size = end - best_split_index;

        if (left_size < right_size) {
            stack.push( { right_node_index, best_split_index, end, depth + 1 } );
            node_index = left_node_index;
            begin = begin;
            end = best_split_index;
            depth = depth + 1;
        } else {
            stack.push( { left_node_index, begin, best_split_index, depth + 1 } );
            node_index = right_node_index;
            begin = best_split_index;
            end = end;
            depth = depth + 1;
        }
    }

    // rearrange triangles based on sorted_references
    CHECK_CUDA(cudaMemcpy(d_primitives_tmp, d_primitives, num_primitives * sizeof(Triangle), cudaMemcpyDeviceToDevice));
    rearrange_primitives<<<blocks_per_grid(num_primitives), THREADS_PER_BLOCK>>>(d_sorted_references[0],
                                                                                 d_primitives_tmp,
                                                                                 num_primitives, d_primitives);
    CHECK_CUDA(cudaGetLastError());

    // free temporary memory
    CHECK_CUDA(cudaFree(d_bboxes));
    CHECK_CUDA(cudaFree(d_bboxes_tmp));
    CHECK_CUDA(cudaFree(d_centers));
    CHECK_CUDA(cudaFree(d_costs));
    CHECK_CUDA(cudaFree(d_unsorted_keys));
    CHECK_CUDA(cudaFree(d_sorted_keys));
    CHECK_CUDA(cudaFree(d_unsorted_references));
    CHECK_CUDA(cudaFree(d_sorted_references[0]));
    CHECK_CUDA(cudaFree(d_sorted_references[1]));
    CHECK_CUDA(cudaFree(d_sorted_references[2]));
    CHECK_CUDA(cudaFree(d_marks));
    CHECK_CUDA(cudaFree(d_primitives_tmp));
    CHECK_CUDA(cudaFree(d_temp_storage));

    // debug
    Node *h_nodes = new Node[2 * num_primitives];
    CHECK_CUDA(cudaMemcpy(h_nodes, d_nodes, 2 * num_primitives * sizeof(Node), cudaMemcpyDeviceToHost));

    Triangle *h_primitives = new Triangle[num_primitives];
    CHECK_CUDA(cudaMemcpy(h_primitives, d_primitives, num_primitives * sizeof(Triangle), cudaMemcpyDeviceToHost));

    int *dummy = new int;
}

#endif //RTCUDA_BVH_CUH
