import argparse
import asyncio
import os
import time
from typing import Dict, Set

import aiohttp
import uvicorn
import uvloop
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from test_stub import get_prefill_worker
from utils import call_kv_method, deserialize_seq_group, timer

from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.core.scheduler import Scheduler
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncStream
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceGroup, SequenceStatus
from vllm.usage.usage_lib import UsageContext
from vllm.utils import make_async, random_uuid
from vllm.logger import init_logger

from Allocator import BlockAllocator

logger = init_logger(__name__)

# global states for this perfill worker
engine: AsyncLLMEngine
block_manager: BlockSpaceManager
scheduler: Scheduler
kv_addr = None
app = FastAPI()
http_session: aiohttp.ClientSession
background_tasks: Set[asyncio.Task] = set()


class PrefillWorker:
    def __init__(self, prefill_addr: str, prefill_port: int):
        self.prefill_addr = prefill_addr
        self.prefill_port = prefill_port

    async def start_prefill(
        self,
        http_session: aiohttp.ClientSession,
        request,
        request_id,
        block_ids,
    ):
        prefill_request = {
            "request_id": request_id,
            "kv_addr": kv_addr,
            "block_ids": block_ids,
            **request,
        }
        url = f"http://{self.prefill_addr}:{self.prefill_port}/prefill"
        async with http_session.post(url, json=prefill_request) as response:
            response.raise_for_status()
            data = await response.json()
            return data


def resume_request(request_id: str, seq_group: SequenceGroup):
    scheduler.block_manager.mark_blocks_as_computed(seq_group)
    stream = AsyncStream(request_id)
    if request_id in engine._request_tracker._request_streams:
        raise KeyError(f"Request {request_id} already exists.")
    engine._request_tracker._request_streams[request_id] = stream
    scheduler.running.append(seq_group)
    engine._request_tracker.new_requests_event.set()
    return stream


async def create_seq_group(
    request_id: str, prompt: str, params: SamplingParams
):
    inputs = await engine.engine.process_model_inputs_async(request_id, prompt)
    arrival_time = time.time()
    processed_inputs = await engine.engine.process_model_inputs_async(
        request_id=request_id, inputs=inputs
    )
    engine.engine._add_processed_request(
        request_id, processed_inputs, params, arrival_time, None, None
    )
    # remove it from the scheduler
    seq_group = scheduler.waiting.pop()
    # reserve cache blocks
    if args.force_fifo:
        await asyncio.wait_for(
            allocator.acquire(arrival_time, seq_group), args.alloc_timeout
        )
    else:
        start_time = time.time()
        while time.time() - start_time < args.alloc_timeout:
            can_allocate = block_manager.can_allocate(seq_group)
            if can_allocate == AllocStatus.OK:
                scheduler._allocate_and_set_running(seq_group)
                break
            elif can_allocate == AllocStatus.LATER:
                await asyncio.sleep(0.1)
            else:
                raise RuntimeError(f"Prompt is too long for {request_id}")
        else:
            raise TimeoutError(f"No enough cache blocks for {request_id}")
    return seq_group


def map_prefilled_seq_group(prefilled, dummy):
    # We assume there is one sequence inside, the prompt
    dummy_seq = dummy.get_seqs()[0]
    real_seq = prefilled.get_seqs()[0]
    real_seq.seq_id = dummy_seq.seq_id
    real_seq.status = SequenceStatus.RUNNING


def get_block_id(seq_group: SequenceGroup):
    seq = seq_group.get_seqs()[0]
    return block_manager.get_block_table(seq)


async def prefill_push(comm, request_info, request_id):
    # copy the request
    request_dict = request_info.copy()
    prompt = request_dict.pop("prompt")
    sampling_params = SamplingParams(**request_dict)
    # create a dummy sequence group to reserve cache
    seq_group = await create_seq_group(request_id, prompt, sampling_params)
    block_ids = get_block_id(seq_group)
    data = await comm.start_prefill(
        http_session, request_info, request_id, block_ids
    )
    prefilled_seq = await make_async(deserialize_seq_group)(data["seq_group"])
    map_prefilled_seq_group(prefilled_seq, seq_group)
    return prefilled_seq


async def prefill_pull(comm, request_info, request_id):
    # copy the request
    request_dict = request_info.copy()
    prompt = request_dict.pop("prompt")
    sampling_params = SamplingParams(**request_dict)
    data = await comm.start_prefill(http_session, request_info, request_id, [])
    prefilled_seq = await make_async(deserialize_seq_group)(data["seq_group"])
    remote_block_id = data["block_id"]
    # create a dummy sequence group to reserve cache
    seq_group = await create_seq_group(request_id, prompt, sampling_params)
    seq = seq_group.get_seqs()[0]
    local_block_ids = block_manager.get_block_table(seq)
    map_prefilled_seq_group(prefilled_seq, seq_group)
    # start to pull KV cache
    await call_kv_method(
        engine,
        "add_request",
        request_id,
        data["kv_addr"],
        remote_block_id,
        lock=True,
    )
    await call_kv_method(engine, "pull_kv", request_id, local_block_ids)
    return prefilled_seq


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_info = await request.json()
    message = request_info.pop("message", "")
    request_id = random_uuid()
    assert "request_id" not in request_info, "The request contains request ID"

    # Get a prefill worker
    i = hash(request_id)
    addr, host = get_prefill_worker(i)
    logger.info("Request %s: is sent to %s.", request_id, addr)
    comm = PrefillWorker(addr, host)

    metrics = {}
    with timer(f"[test{message}] [{request_id}]", result=metrics) as t:
        prefill_cm = t.record("Prefill")
        decode_cm = t.record("Decode")
        prefill_cm.__enter__()

        if args.transfer_mode == "push":
            seq_group = await prefill_push(comm, request_info, request_id)
        elif args.transfer_mode == "pull":
            seq_group = await prefill_pull(comm, request_info, request_id)

        # wait for KV cache ready
        await call_kv_method(engine, "wait_kv", request_id)
        prefill_cm.__exit__(None, None, None)


        # resume inference
        stream = resume_request(request_id, seq_group)
        final_output = None
        decode_cm.__enter__()
        async for request_output in stream:
            decode_cm.__exit__(None, None, None)
            final_output = request_output
            decode_cm = t.record("Decode")
            decode_cm.__enter__()
        decode_cm.__exit__(None, None, None)
        # t.tagged_time["Decode"] /= t.tagged_count["Decode"] - 1

    assert final_output is not None
    prompt = final_output.prompt
    text = [prompt + output.text for output in final_output.outputs]
    return JSONResponse({"request_id": request_id, "text": text, "metric": metrics})


async def run(config: uvicorn.Config):
    global http_session, engine, block_manager, scheduler, kv_addr, allocator
    engine.start_background_loop()
    scheduler = engine.engine.scheduler[0]
    assert len(engine.engine.scheduler) == 1, "Dose not support PP"
    block_manager = scheduler.block_manager
    kv_addr = await call_kv_method(engine, "decode_kv_init")
    http_session = aiohttp.ClientSession()
    server = uvicorn.Server(config=config)
    if args.force_fifo:
        allocator = BlockAllocator(block_manager, scheduler)
    await server.serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", type=str, default="info")
    parser.add_argument(
        "--model-path", type=str, default="/data/mistral-7b-instruct-v0_2"
    )
    parser.add_argument("--alloc-timeout", type=int, default=600)
    parser.add_argument(
        "--transfer-mode", type=str, choices=["pull", "push"], default="pull"
    )
    parser.add_argument("--force-fifo", action="store_true")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.model = args.model_path
    args.enforce_eager = True
    args.disable_custom_all_reduce = True
    args.engine_use_ray = False
    args.worker_use_ray = True
    args.enable_chunked_prefill = False
    engine_args = AsyncEngineArgs.from_cli_args(args)

    os.environ["RAY_NUM_CPUS"] = "64"
    os.environ["WORKER_MODULE"] = "worker"
    os.environ["WORKER_CLASS"] = (
        "WorkerPull" if args.transfer_mode == "pull" else "WorkerPush"
    )
    os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "600"

    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.API_SERVER
    )

    config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    uvloop.run(run(config))
