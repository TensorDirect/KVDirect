import asyncio
import base64
import fcntl
import json
import os
import pickle
import socket
import struct
import subprocess
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Callable, Set

import torch

from vllm.sequence import SequenceGroup
from vllm.utils import make_async


@lru_cache()
def to_byte(s):
    return s.encode("ascii")


class Recorder:
    def __init__(self, desc):
        self.desc = desc
        self.tagged_time = {}
        self.tagged_count = {}

    @contextmanager
    def record(self, tag):
        start = time.time()
        yield
        end = time.time()
        elapsed_time = end - start
        if tag not in self.tagged_time:
            self.tagged_time[tag] = elapsed_time
            self.tagged_count[tag] = 1
        else:
            self.tagged_time[tag] += elapsed_time
            self.tagged_count[tag] += 1

    def show(self):
        for tag, t in self.tagged_time.items():
            print(f"{self.desc}:{tag} {t}")


@contextmanager
def timer(desc, print_time = True, result = None):
    t = Recorder(desc)
    start = time.time()
    yield t
    end = time.time()
    if print_time:
        print(f"{desc}: {end - start}")
        t.show()
    if result is not None:
        result['total'] = end - start
        for tag in t.tagged_time.keys():
            result[tag] = {'time': t.tagged_time[tag], "count": t.tagged_count[tag]}

def deserialize_seq_group(data: str) -> SequenceGroup:
    data = base64.b64decode(data)
    return pickle.loads(data)


def serialize_seq_group(seq_group: SequenceGroup) -> str:
    data = pickle.dumps(seq_group)
    return base64.b64encode(data).decode("utf-8")


def get_real_device_id(device_id):
    if device_id is None:
        device_id = torch.cuda.current_device()
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
    if visible_devices:
        visible_devices = visible_devices.split(",")
        device_id = int(visible_devices[device_id])
    return device_id


def get_address(ifname):
    import json
    ip_addr = subprocess.Popen(
        ["ip", "-j", "addr", "show", ifname],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ip_addr = json.loads(ip_addr.communicate()[0])
    # print(f"ip_addr: {ip_addr}")
    ip_addr = ip_addr[0]['addr_info'][0]['local']
    return f"{ip_addr}"
    # ifname = ifname.encode()
    # with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    #     return socket.inet_ntoa(
    #         fcntl.ioctl(
    #             s.fileno(),
    #             0x8915,
    #             struct.pack("256s", ifname[:15]),  # SIOCGIFADDR
    #         )[20:24]
    #     )


def dump_NIC():
    num_gpu = torch.cuda.device_count()
    topo_proc = subprocess.Popen(
        ["nvidia-smi", "topo", "-m"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    rdma_proc = subprocess.Popen(
        ["rdma", "dev"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    rdma_out = rdma_proc.communicate()[0]
    active_flag = "00.00.0000"
    rdma_link = {}
    for link in rdma_out.split("\n"):
        if active_flag in link:
            mlx = link.split(" ")[1].split(":")[0]
            eth = "ib" + link.split(" ")[0].split(":")[0]
            rdma_link[mlx] = eth

    data = {}

    # print(rdma_link)
    nic_lengends = {
        "NIC0": "mlx5_0",
        "NIC1": "mlx5_1",
        "NIC2": "mlx5_2",
        "NIC3": "mlx5_3",
        "NIC4": "mlx5_4",
        "NIC5": "mlx5_5",
        "NIC6": "mlx5_6",
        "NIC7": "mlx5_7",
        "NIC8": "mlx5_8",
    }

    topo_out = topo_proc.communicate()[0]
    topo_lines = topo_out.split("\n")
    devices = topo_lines[0][5:].split()[:-4]
    for i in range(num_gpu):
        device = devices[i]
        conns = topo_lines[i + 1].split()[:-2]
        assert conns[0] == device, "Pasing error"
        # assert len(conns[1:]) == len(devices), "Parsing error"
        mlxs = [nic_lengends[devices[i]] for i, conn in enumerate(conns[1:]) if conn == "NODE"]
        # print(f"mlxs: {mlxs}")
        assert (
            len(mlxs) > 0
        ), f"Unable to find NIC connected to GPU {i} with PXB"
        nics = [
            (mlx, rdma_link[mlx])
            for mlx in mlxs
            if mlx in rdma_link
        ]
        # print(f"nics: {nics}")
        mlx_nic, eth_nic = nics[i % 4]
        data[i] = {
            "mlx": mlx_nic,
            "eth": eth_nic,
            "address": get_address(eth_nic),
        }

    return data


def detect_NIC(device_id):
    device_id = get_real_device_id(device_id)
    with open("/tmp/rdma_link.json", "r") as f:
        data = json.load(f)
    return data[str(device_id)]



async def call_kv_method(engine, method: str, *args, lock=False, **kwargs):
    assert engine and engine.is_running, "Engine is not running"
    driver = engine.engine.model_executor.driver_worker
    coros = [make_async(driver.execute_method)(method, *args, **kwargs)]
    if hasattr(engine.engine.model_executor, "workers"):
        workers = engine.engine.model_executor.workers
        group = "lock" if lock else "kv"
        for worker in workers:
            future = asyncio.wrap_future(
                worker.execute_method.options(concurrency_group=group)
                .remote(method, *args, **kwargs)
                .future()
            )
            coros.append(future)

    return await asyncio.gather(*coros)


def make_done_callback(
    tasks: Set[asyncio.Task], call_back: Callable, *args, **kwargs
):
    def _wrapper(task: asyncio.Task):
        tasks.discard(task)
        call_back(*args, **kwargs)

    return _wrapper


def wrap_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Wrap a tensor with a view when their dtype is not compatible."""
    supported = [
        torch.complex64,
        torch.complex128,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ]
    if tensor.dtype not in supported:
        new_dtype = next(
            dtype
            for dtype in supported
            if tensor.dtype.itemsize == dtype.itemsize
        )
        return tensor.view(new_dtype)
    return tensor
