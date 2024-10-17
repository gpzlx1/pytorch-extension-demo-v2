from triton_hash_encode import hash_encode
import torch
from functools import partial

def bench(func):
    import time
    import numpy as np
    for i in range(5):
        func()

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        func()
    torch.cuda.synchronize()
    t1 = time.time()
    print((t1 - t0) * 1000 /100)

def old_hash_encode(
    k: torch.Tensor, hash_weight: torch.Tensor
):
    HEAD_DIM = hash_weight.shape[0]
    RBIT = hash_weight.shape[1]

    if RBIT % 64 == 0:
        output_dtype = torch.int64
        chunk_size = 64
    else:
        output_dtype = torch.int32
        chunk_size = 32
    
    key_code = torch.matmul(key_states, hash_weight) > 0
    special_tensor = torch.pow(
        2, torch.arange(0, chunk_size, 1, dtype=output_dtype, device=key_states.device)
    )
    chunk_num = int(RBIT / chunk_size)
    packbit_key_code = key_code.reshape(BSZ, HEAD, KEY_SIZE, chunk_num, chunk_size)
    packbit_key_code = packbit_key_code * special_tensor
    packbit_key_code = packbit_key_code.sum(dim=-1, dtype=output_dtype)

    return packbit_key_code

torch.cuda.set_device(7)


KEY_SIZE = 31231
BSZ, HEAD, HIDDEN_SIZE = 1, 32, 128
RBIT = 256

key_states = torch.randn(
    (BSZ, HEAD, KEY_SIZE, HIDDEN_SIZE), dtype=torch.float16, device=torch.device("cuda")
)


hash_weight = torch.normal(
    0,
    2,
    size=(HIDDEN_SIZE, RBIT),
    device=key_states.device,
    dtype=key_states.dtype,
)

packbit_key_code = old_hash_encode(key_states, hash_weight)
print(packbit_key_code.shape, packbit_key_code.dtype)
out = hash_encode(key_states, hash_weight)


# assert (out == packbit_key_code).all()
print("begin bench")

bench(partial(old_hash_encode, key_states, hash_weight))
bench(partial(hash_encode, key_states, hash_weight))

