import random
from itertools import accumulate
import os

import torch
import utils
from kv_comm import KVComm

from vllm.worker.worker import Worker


class SplitWorkerBase(Worker):
    def setup(self, role, port=40000):
        # Assume all tensors are the same
        cache = self.gpu_cache[0][0] # CSY: Accomodate vllm v0.5
        assert self.local_rank == cache.device.index
        info = utils.detect_NIC(self.local_rank)
        addr = f'{info["address"]}:{port+self.local_rank}'
        if os.getenv("SERVER_LISTEN_ADDR"):
            listen_addr = f"{os.getenv('SERVER_LISTEN_ADDR')}:{port+self.local_rank}"
        else:
            listen_addr = addr
        self.kv_comm = KVComm(self.gpu_cache[0], listen_addr, role)
        return addr

    def wait_kv(self, request_id: str):
        """Wait for kv comm to finish."""
        self.kv_comm.wait_kv(request_id)  # block here

    def add_request(self, request_id, kv_addr, block_ids):
        info = next(
            info for info in kv_addr if info["device"] == self.local_rank
        )
        self.kv_comm.add_request(request_id, info["addr"], block_ids)


class WorkerPush(SplitWorkerBase):
    def decode_kv_init(self):
        """Initialize the KV cache communicator as the decode worker"""
        addr = self.setup("server")
        return {"device": self.local_rank, "addr": addr}

    def prefill_kv_init(self, layer_wise=-1):
        """Initialize the KV cache communicator as the prefill worker"""
        addr = self.setup("client")

        layers = self.model_runner.model.model.layers
        block_size = self.model_runner.block_size
        num_layer = len(self.gpu_cache)
        assert num_layer == len(layers)

        if layer_wise == -1:
            layer_wise = len(self.gpu_cache)

        assert layer_wise <= num_layer and num_layer % layer_wise == 0

        def forward_wrapper(i, layer):
            finished = i + 1
            is_push = finished % layer_wise == 0
            last_push = finished - layer_wise
            layers = list(range(last_push, finished))

            old_forward = layer.forward

            def new_forward(*args, **kwargs):
                output = old_forward(*args, **kwargs)
                # TODO: If there is chunked prefill, we need to get the actual computed block ids for this iteration.
                # Besides, we need to figure out the remote block id for this iteration as well.
                # For now, the chunked prefill has not benefits, so we just push the whole layer.
                slot_mapping = args[3].slot_mapping
                indices = list(accumulate(args[3].seq_lens))[:-1]
                # This ensures the computation of this layer finish
                block_ids = (slot_mapping // block_size).cpu()
                block_ids = block_ids.tensor_split(indices)
                block_ids = [
                    torch.unique_consecutive(bid).tolist() for bid in block_ids
                ]
                self.kv_comm.transfer_kv(
                    args[3].request_ids, block_ids, layers, "push"
                )

                return output

            return new_forward if is_push else old_forward

        for i, layer in enumerate(layers):
            layer.forward = forward_wrapper(i, layer)

        return {"device": self.local_rank, "addr": addr}


class WorkerPull(SplitWorkerBase):
    def decode_kv_init(self):
        addr = self.setup("client")
        return {"device": self.local_rank, "addr": addr}

    def prefill_kv_init(self, layer_wise=-1):
        addr = self.setup("server")
        return {"device": self.local_rank, "addr": addr}

    def pull_kv(self, request_id, local_block_ids):
        """ This function is non-blocking
        """
        self.kv_comm.transfer_kv(
            [request_id],
            [local_block_ids],
            list(range(len(self.gpu_cache))),
            "pull"
        )
