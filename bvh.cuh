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

    struct Intersection {
        int primitive_index;
        Triangle::Intersection triangle_intersection;
    };

    Bvh(const std::vector<Triangle> &primitives);
    __device__ bool intersect_leaf(const Node* d_node_ptr, Ray &ray, Intersection &intersection) const;
    template <typename Stack> __device__ bool traverse(Stack &stack, Ray &ray, Bvh::Intersection &intersection) const;

    static constexpr int MAX_DEPTH = 64;  // depth restriction

    int num_primitives;
    Triangle *d_primitives;
    int num_nodes;
    Node *d_nodes;
};

Bvh::Bvh(const std::vector<Triangle> &primitives)
    : num_primitives(primitives.size()) {
    // allocate temporary memory for BVH construction
    profiler.start("Allocating temporary memory for BVH construction");
    auto h_bboxes = std::make_unique<BoundingBox[]>(num_primitives);
    auto h_centers = std::make_unique<Vec3[]>(num_primitives);
    auto h_costs = std::make_unique<float[]>(num_primitives);
    auto h_marks = std::make_unique<bool[]>(num_primitives);
    auto h_sorted_references_data = std::make_unique<int[]>(3 * num_primitives);
    int *h_sorted_references[3] = { h_sorted_references_data.get(),
                                    h_sorted_references_data.get() + num_primitives,
                                    h_sorted_references_data.get() + 2 * num_primitives };
    auto h_nodes = std::make_unique<Node[]>(2 * num_primitives);
    auto h_primitives_tmp = std::make_unique<Triangle[]>(num_primitives);
    profiler.stop();

    // initially, there is only one node
    num_nodes = 1;

    // initialize h_bboxes, h_centers, and h_nodes[0].bbox
    profiler.start("Initializing bounding boxes and centers");
    h_nodes[0].bbox.reset();
    for (int i = 0; i < num_primitives; i++) {
        h_bboxes[i] = primitives[i].bounding_box();
        h_nodes[0].bbox.extend(h_bboxes[i]);
        h_centers[i] = primitives[i].center();
    }
    profiler.stop();

    // TODO: change to radix sort (maybe on GPU?)
    profiler.start("Sorting primitives");
    // sort on x-coordinate
    std::iota(h_sorted_references[0], h_sorted_references[0] + num_primitives, 0);
    std::sort(h_sorted_references[0], h_sorted_references[0] + num_primitives,
              [&](int i, int j) { return h_centers[i].x < h_centers[j].x; });

    // sort on y-coordinate
    std::iota(h_sorted_references[1], h_sorted_references[1] + num_primitives, 0);
    std::sort(h_sorted_references[1], h_sorted_references[1] + num_primitives,
              [&](int i, int j) { return h_centers[i].y < h_centers[j].y; });

    // sort on z-coordinate
    std::iota(h_sorted_references[2], h_sorted_references[2] + num_primitives, 0);
    std::sort(h_sorted_references[2], h_sorted_references[2] + num_primitives,
              [&](int i, int j) { return h_centers[i].z < h_centers[j].z; });
    profiler.stop();

    // initialize stack for BVH construction
    std::stack<std::array<int, 4>> stack;  // node_index, begin, end, depth
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

    // recursion step for BVH construction (implemented by stack)
    profiler.start("Recursively constructing BVH");
    while (true) {
        Node &curr_node = h_nodes[node_index];
        int curr_num_primitives = end - begin;

        // this node should be a leaf node
        if (curr_num_primitives <= 1 || depth >= MAX_DEPTH) {
            curr_node.num_primitives = curr_num_primitives;
            curr_node.first_primitive_index = begin;
            if (check_and_update_and_pop_stack()) continue;
            else break;
        }

        float best_cost = FLT_MAX;
        int best_axis = -1;
        int best_split_index = -1;

        // find best split axis and split index
        for (int axis = 0; axis < 3; axis++) {
            BoundingBox tmp_bbox = BoundingBox::Empty();
            for (int i = end - 1; i > begin; i--) {
                tmp_bbox.extend(h_bboxes[h_sorted_references[axis][i]]);
                h_costs[i] = tmp_bbox.half_area() * (end - i);
            }

            tmp_bbox.reset();
            for (int i = begin; i < end - 1; i++) {
                tmp_bbox.extend(h_bboxes[h_sorted_references[axis][i]]);
                float cost = tmp_bbox.half_area() * (i + 1 - begin) + h_costs[i + 1];
                if (cost < best_cost) {
                    best_cost = cost;
                    best_axis = axis;
                    best_split_index = i + 1;
                }
            }
        }

        // if best_cost >= max_split_cost, this node should be a leaf node
        float max_split_cost = curr_node.bbox.half_area() * (curr_num_primitives - 1);
        if (best_cost >= max_split_cost) {
            curr_node.num_primitives = curr_num_primitives;
            curr_node.first_primitive_index = begin;
            if (check_and_update_and_pop_stack()) continue;
            else break;
        }

        // set bbox of left and right nodes
        int left_node_index = num_nodes;
        int right_node_index = num_nodes + 1;
        Node &left_node = h_nodes[left_node_index];
        Node &right_node = h_nodes[right_node_index];
        left_node.bbox.reset();
        right_node.bbox.reset();
        for (int i = begin; i < best_split_index; i++) {
            left_node.bbox.extend(h_bboxes[h_sorted_references[best_axis][i]]);
            h_marks[h_sorted_references[best_axis][i]] = true;
        }
        for (int i = best_split_index; i < end; i++) {
            right_node.bbox.extend(h_bboxes[h_sorted_references[best_axis][i]]);
            h_marks[h_sorted_references[best_axis][i]] = false;
        }

        // partition sorted_references of other axes and ensure their relative order
        int other_axis[2] = { (best_axis + 1) % 3, (best_axis + 2) % 3 };
        std::stable_partition(h_sorted_references[other_axis[0]] + begin,
                              h_sorted_references[other_axis[0]] + end,
                              [&](int i) { return h_marks[i]; });
        std::stable_partition(h_sorted_references[other_axis[1]] + begin,
                              h_sorted_references[other_axis[1]] + end,
                              [&](int i) { return h_marks[i]; });

        // now we are sure that this node is an internal node
        num_nodes += 2;
        curr_node.num_primitives = 0;
        curr_node.left_node_index = left_node_index;

        int left_size = best_split_index - begin;
        int right_size = end - best_split_index;

        // process smaller subtree first
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
    profiler.stop();

    profiler.start("Copying constructed BVH to device");
    // rearrange primitives based on h_sorted_references
    std::copy(primitives.begin(), primitives.end(), h_primitives_tmp.get());
    for (int i = 0; i < num_primitives; i++) h_primitives_tmp[i] = primitives[h_sorted_references[0][i]];

    // copy primitives to device
    CHECK_CUDA(cudaMalloc(&d_primitives, num_primitives * sizeof(Triangle)));
    CHECK_CUDA(cudaMemcpy(d_primitives, h_primitives_tmp.get(),
                          num_primitives * sizeof(Triangle), cudaMemcpyHostToDevice));

    // copy nodes to device
    CHECK_CUDA(cudaMalloc(&d_nodes, num_nodes * sizeof(Node)));
    CHECK_CUDA(cudaMemcpy(d_nodes, h_nodes.get(), num_nodes * sizeof(Node), cudaMemcpyHostToDevice));
    profiler.stop();
}

__device__ bool Bvh::intersect_leaf(const Node* d_node_ptr, Ray &ray, Bvh::Intersection &intersection) const {
    bool hit_anything = false;
    for (int i = d_node_ptr->first_primitive_index;
         i < d_node_ptr->first_primitive_index + d_node_ptr->num_primitives;
         i++) {
        if (d_primitives[i].intersect(ray, intersection.triangle_intersection)) {
            intersection.primitive_index = i;
            ray.tmax = intersection.triangle_intersection.t;
            hit_anything = true;
        }
    }
    return hit_anything;
}

template <typename Stack>
__device__ bool Bvh::traverse(Stack &stack, Ray &ray, Bvh::Intersection &intersection) const {
    if (d_nodes[0].is_leaf()) return intersect_leaf(&d_nodes[0], ray, intersection);

    bool hit_anything = false;
    AABBIntersector aabb_intersector(ray);

    Node *left_node_ptr = &d_nodes[d_nodes[0].left_node_index];
    while (true) {
        Node *right_node_ptr = left_node_ptr + 1;

        float entry_left;
        if (aabb_intersector.intersect(left_node_ptr->bbox, entry_left)) {
            if (left_node_ptr->is_leaf()) {
                hit_anything |= intersect_leaf(left_node_ptr, ray, intersection);
                left_node_ptr = nullptr;
            }
        } else {
            left_node_ptr = nullptr;
        }

        float entry_right;
        if (aabb_intersector.intersect(right_node_ptr->bbox, entry_right)) {
            if (right_node_ptr->is_leaf()) {
                hit_anything |= intersect_leaf(right_node_ptr, ray, intersection);
                right_node_ptr = nullptr;
            }
        } else {
            right_node_ptr = nullptr;
        }

        // TODO: maybe eliminate branch divergence can boost performance?
        if (left_node_ptr) {
            // TODO: check which node is nearer
            if (right_node_ptr) stack.push(right_node_ptr->left_node_index);
            left_node_ptr = &d_nodes[left_node_ptr->left_node_index];
        } else if (right_node_ptr) {
            left_node_ptr = &d_nodes[right_node_ptr->left_node_index];
        } else {
            if (stack.empty()) break;
            left_node_ptr = &d_nodes[stack.pop()];
        }
    }

    return hit_anything;
}

#endif //RTCUDA_BVH_CUH
