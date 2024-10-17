import torch
import TorchEXTLib


KEY_SIZE = 32000
Q_SIZE = 1
BSZ, HEAD, HIDDEN_SIZE = 1, 32, 128
RBIT = 256

query_states = torch.randn(
    (BSZ, HEAD, Q_SIZE, HIDDEN_SIZE), dtype=torch.float16, device=torch.device("cuda")
)
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

key_code = torch.matmul(key_states, hash_weight) > 0
query_code = torch.matmul(query_states, hash_weight) > 0


hamming_distance = (
    (query_code.to(torch.float16) - key_code.to(torch.float16)).abs().sum(dim=-1)
)


DISTANCE_TYPE = torch.int64
CHUNK_SIZE = 64

special_tensor = torch.pow(
    2, torch.arange(0, CHUNK_SIZE, 1, dtype=DISTANCE_TYPE, device=key_states.device)
)
chunk_num = int(RBIT / CHUNK_SIZE)
key_code = key_code.reshape(BSZ, HEAD, KEY_SIZE, chunk_num, CHUNK_SIZE)
key_code = (key_code * special_tensor).sum(dim=-1).to(DISTANCE_TYPE)
query_code = query_code.reshape(BSZ, HEAD, Q_SIZE, chunk_num, CHUNK_SIZE)
query_code = (query_code * special_tensor).sum(dim=-1).to(DISTANCE_TYPE)
my_hamming_distance = TorchEXTLib.my_popc(query_code, key_code).squeeze(-1)


assert (my_hamming_distance == hamming_distance).all()
print(my_hamming_distance.shape)


# bench mark
import time
import numpy as np

for i in range(10):
    TorchEXTLib.my_popc(query_code, key_code)


time_list = []
torch.cuda.synchronize()
start = time.time()
for i in range(100):
    TorchEXTLib.my_popc(query_code, key_code)
torch.cuda.synchronize()
end = time.time()
  
print((end - start) * 1000 / 100)