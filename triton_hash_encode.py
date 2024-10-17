from typing import Tuple
import torch

import triton
import triton.language as tl

configs = [
    triton.Config({"BLOCK_M": BM}, num_stages=s, num_warps=w)
    for BM in [32, 64, 128]
    for s in ([3, 4, 7])
    for w in [4, 8]
]

# configs = [
#     triton.Config({"BLOCK_M": BM}, num_stages=s, num_warps=w)
#     for BM in [16]
#     for s in ([1])
#     for w in [1]
# ]


@triton.autotune(configs=configs, key=["HEAD_DIM", "RBIT"])
@triton.jit
def _hash_encode(
    k_ptr,
    hash_weight_ptr,
    packbit_tensor_ptr,
    output,
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_wd: tl.constexpr,
    stride_wr: tl.constexpr,
    stride_oz: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_on: tl.constexpr,
    stride_oc: tl.constexpr,
    Z: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RBIT: tl.constexpr,
    NUM_CHUNK: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(BLOCK_N == CHUNK_SIZE)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    out_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    # load pack tensor
    packbit_tensor = tl.load(packbit_tensor_ptr + tl.arange(0, CHUNK_SIZE))

    K_block_ptr = tl.make_block_ptr(
        base=k_ptr + k_offset,
        shape=(N, HEAD_DIM),
        strides=(stride_kn, stride_kd),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    Output_block_ptr = tl.make_block_ptr(
        base=output + out_offset,
        shape=(N, NUM_CHUNK),
        strides=(stride_on, stride_oc),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )
    Weight_ptr = tl.make_block_ptr(
        base=hash_weight_ptr,
        shape=(HEAD_DIM, RBIT),
        strides=(stride_wd, stride_wr),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(1, 0),
    )

    # load K
    K = tl.load(K_block_ptr, boundary_check=(1, 0), padding_option="zero")

    for start_n in range(0, RBIT, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # load weight
        Weight = tl.load(
            Weight_ptr, boundary_check=(0, 1), padding_option="zero"
        )  # [HEAD_DIM, BLOCK_N]

        # compute
        tmp_results = tl.dot(K, Weight)  # [BLOCK_M, BLOCK_N] (BLOCK_N == CHUNK_SIZE)
        tmp_results = tmp_results > 0
        tmp_results = tmp_results.to(packbit_tensor.type.element_ty)
        tmp_results = tmp_results * packbit_tensor  # [BLOCK_M, BLOCK_N] * [BLOCK_N]
        tmp_results = tl.sum(tmp_results, axis=1)
        tmp_results = tmp_results.reshape(BLOCK_M, 1)

        tl.store(Output_block_ptr, tmp_results, boundary_check=(1, 0))

        Weight_ptr = tl.advance(Weight_ptr, (0, BLOCK_N))
        Output_block_ptr = tl.advance(Output_block_ptr, (0, 1))


@torch.jit.ignore
def hash_encode(
    k: torch.Tensor, hash_weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    RBIT = hash_weight.shape[-1]
    bsz, head, nctx, dim = k.shape

    assert dim in {16, 32, 64, 128, 256}

    assert RBIT % 32 == 0

    if RBIT % 64 == 0:
        output_dtype = torch.int64
        num_chunk = RBIT // 64
        chunk_size = 64
    else:
        output_dtype = torch.int32
        num_chunk = RBIT // 32
        chunk_size = 32

    extra_kern_args = {}

    packbit_aux_tensor = torch.pow(
        2, torch.arange(0, chunk_size, 1, dtype=output_dtype, device=k.device)
    )
    output = torch.empty(
        (bsz, head, nctx, num_chunk), dtype=output_dtype, device=k.device
    )

    grid = lambda args: (
        triton.cdiv(nctx, args["BLOCK_M"]),
        bsz * head,
        1,
    )
    _hash_encode[grid](
        k,
        hash_weight,
        packbit_aux_tensor,
        output,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        hash_weight.stride(0),
        hash_weight.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        bsz,
        head,
        nctx,
        dim,
        RBIT,
        num_chunk,
        chunk_size,
        BLOCK_N=chunk_size,
        **extra_kern_args,
    )

    return output
