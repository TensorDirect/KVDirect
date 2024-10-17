import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import utils
from rdma_transport import (
    TensorBlock,
    TensorBlocks,
    VllmRdmaClient,
    VllmRdmaServer,
)

KV_TIMEOUT = 30


def coalesce_indices(local_indices, remote_indices):
    sorted_indices = np.argsort(local_indices)
    local_indices = local_indices[sorted_indices]
    remote_indices = remote_indices[sorted_indices]
    local_diff = np.diff(local_indices)
    remote_diff = np.diff(remote_indices)
    breaks = np.where((local_diff != 1) | (remote_diff != 1))[0] + 1
    segment_starts = np.concatenate(([0], breaks))
    segment_ends = np.concatenate((breaks, [len(local_indices)]))
    local_start = local_indices[segment_starts]
    remote_start = remote_indices[segment_starts]
    num_blocks = segment_ends - segment_starts
    return local_start, remote_start, num_blocks


class KVComm:
    def __init__(
        self,
        cache: List[torch.Tensor],
        local_addr: str,
        role: str,
    ):
        self.cache = cache
        self.rank = cache[0].device.index
        self.device_id = utils.get_real_device_id(self.rank)
        self.local_addr = local_addr
        self.num_layer = len(cache)
        self.shape = tuple(cache[0].shape)
        self.block_size = (
            self.shape[-1]
            * self.shape[-2]
            * self.shape[-3]
            * cache[0].dtype.itemsize
        )  # in bytes
        self.local_base = [t.data_ptr() for t in cache]
        self.kv_stride = cache[0].stride()[0]

        self.tensor_blocks = TensorBlocks()
        for c in cache:
            self.tensor_blocks.add(
                TensorBlock(c.data_ptr(), 0, c.numel() * c.element_size())
            )
        self.role = role
        if role == "server":
            self.server = VllmRdmaServer(
                self.local_addr, self.device_id, self.tensor_blocks
            )
            self.server.listen()
        elif role == "client":
            self.clients = {}
            self.remote_base_ptr = {}
        self.pending_requests = {}
        self.remaining_blocks = {}

    def get_client(self, server_addr):
        if server_addr not in self.clients:
            client = VllmRdmaClient(self.device_id, self.tensor_blocks)
            remote_tb: TensorBlocks = client.connect(server_addr)
            self.clients[server_addr] = client
            remote_base_ptr = remote_tb.get_base_ptrs()
            self.remote_base_ptr[server_addr] = remote_base_ptr
        else:
            client = self.clients[server_addr]
            remote_base_ptr = self.remote_base_ptr[server_addr]
        return client, remote_base_ptr

    def add_request(
        self, request_id: str, server_addr: str, block_ids: List[int]
    ):
        assert request_id not in self.pending_requests
        self.pending_requests[request_id] = (server_addr, block_ids)
        self.remaining_blocks[request_id] = len(block_ids) * self.num_layer * 2
        _ = self.get_client(server_addr)

    def transfer_kv(
        self,
        request_ids: List[str],
        local_block_ids: List[List[int]],
        layers: List[int],
        action: str,
    ):
        requests = defaultdict(list)
        for rid, local_bid in zip(request_ids, local_block_ids):
            addr, remote_bid = self.pending_requests[rid]
            requests[addr].append((rid, local_bid, remote_bid))

        for addr, request_info in requests.items():
            client, remote_base = self.get_client(addr)
            func = client.send if action == "push" else client.recv
            for rid, local_bid, remote_bid in request_info:
                remaining = self.remaining_blocks[rid]
                if remaining == 0:
                    client.complete(rid)
                local_start, remote_start, num_blocks = coalesce_indices(
                    np.array(local_bid), np.array(remote_bid)
                )
                for l_start, r_start, num_blk in zip(
                    local_start, remote_start, num_blocks
                ):
                    for layer in layers:
                        remote_base_ptr = remote_base[layer]
                        local_base_ptr = self.local_base[layer]
                        remaining -= 2 * num_blk
                        func(
                            TensorBlock(
                                local_base_ptr,
                                l_start * self.block_size,
                                self.block_size * num_blk,
                            ),
                            TensorBlock(
                                remote_base_ptr,
                                r_start * self.block_size,
                                self.block_size * num_blk,
                            ),
                        )
                        func(
                            TensorBlock(
                                local_base_ptr,
                                self.kv_stride + l_start * self.block_size,
                                self.block_size * num_blk,
                            ),
                            TensorBlock(
                                remote_base_ptr,
                                self.kv_stride + r_start * self.block_size,
                                self.block_size * num_blk,
                            ),
                        )
                self.remaining_blocks[rid] = remaining

    def wait_kv(self, request_id):
        if self.role == "server":
            handle = self.server
        elif self.role == "client":
            addr, _ = self.pending_requests[request_id]
            handle = self.clients[addr]
        start = time.time()
        while handle.is_complete(utils.to_byte(request_id)) is None:
            if time.time() - start > KV_TIMEOUT:
                raise TimeoutError(f"KV cache pull for {request_id} timeout")
            time.sleep(1 / 1000)
