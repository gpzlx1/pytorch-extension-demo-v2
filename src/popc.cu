#include <ATen/cuda/CUDAContext.h>
#include "operator.h"

__device__ int32_t popc(int32_t value) { return __popc(value); }

__device__ int32_t popc(int64_t value) { return __popcll(value); }

template <typename T>
__global__ void popc_kernel(T *__restrict__ query, T *__restrict__ key,
                            int32_t *__restrict__ output,
                            uint32_t num_key_group, uint32_t query_head_stride,
                            uint32_t key_head_stride,
                            uint32_t output_head_stride, uint32_t numel,
                            uint32_t dim) {
  // one block for one head
  uint32_t tid = threadIdx.x;
  uint32_t head_id = blockIdx.y;
  uint32_t start_m = blockIdx.x;
  uint32_t BLOCK_M = blockDim.x;

  uint32_t query_head_id = head_id;
  uint32_t key_head_id = head_id / num_key_group;

  T *part_query = query + query_head_id * query_head_stride;
  T *part_key = key + key_head_id * key_head_stride + start_m * BLOCK_M * dim;
  int32_t *part_output =
      output + query_head_id * output_head_stride + start_m * BLOCK_M;

  uint32_t start = start_m * BLOCK_M;
  uint32_t end = min(BLOCK_M, numel - start);
  // printf("start: %d, end: %d\n", tid, end);
  for (uint32_t i = tid; i < end; i += BLOCK_M) {
    int32_t dis = 0;

    for (uint32_t j = 0; j < dim; j++) {
      T tmp = (part_query[j] ^ part_key[i * dim + j]);
      dis += popc(tmp);
    }

    part_output[i] = dis;
  }
}

torch::Tensor PopcCUDA(torch::Tensor query, torch::Tensor key) {
  CHECK(query.size(2) == 1);

  uint32_t bsz = key.size(0);
  uint32_t key_head = key.size(1);
  uint32_t seq_len = key.size(2);
  uint32_t dim = key.size(3);

  uint32_t query_head = query.size(1);
  uint32_t num_key_group = query_head / key_head;

  torch::Tensor output = torch::empty(
      {bsz, query_head, seq_len, 1},
      torch::TensorOptions().dtype(torch::kInt32).device(query.device()));

  uint32_t key_head_stride = key.stride(1);
  uint32_t query_head_stride = query.stride(1);
  uint32_t output_head_stride = output.stride(1);

  int BLOCK_M = 256;
  uint32_t total_head = bsz * query_head;
  dim3 grid((seq_len + BLOCK_M - 1) / BLOCK_M, total_head);
  dim3 block(BLOCK_M);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(query.device().index());

  if (query.dtype() == torch::kInt32) {
    popc_kernel<int32_t><<<grid, block, 0, stream>>>(
        query.data_ptr<int32_t>(), key.data_ptr<int32_t>(),
        output.data_ptr<int32_t>(), num_key_group, query_head_stride,
        key_head_stride, output_head_stride, seq_len, dim);

  } else if (query.dtype() == torch::kInt64) {
    popc_kernel<int64_t><<<grid, block, 0, stream>>>(
        query.data_ptr<int64_t>(), key.data_ptr<int64_t>(),
        output.data_ptr<int32_t>(), num_key_group, query_head_stride,
        key_head_stride, output_head_stride, seq_len, dim);

  } else {
    std::cout << "unsupported dtype for PopcCUDA" << std::endl;
  }

  return output;
}