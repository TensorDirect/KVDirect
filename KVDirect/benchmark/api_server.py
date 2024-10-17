"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""

import argparse
import json
import ssl
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid
from vllm.logger import init_logger

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import timer

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None

logger = init_logger(__name__)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    message = request_dict.pop("message", "")
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    assert engine is not None
    results_generator = engine.generate(prompt, sampling_params, request_id)

    with timer(f"[test{message}] [{request_id}]") as t:
        # Non-streaming case
        final_output = None
        prefill_cm = t.record("Prefill")
        decode_cm = t.record("Decode")
        prefill_cm.__enter__()
        async for request_output in results_generator:
            if final_output is None:
                prefill_cm.__exit__(None, None, None)
                decode_cm.__enter__()
            else:
                decode_cm.__exit__(None, None, None)
                decode_cm = t.record("Decode")
                decode_cm.__enter__()
            final_output = request_output
        decode_cm.__exit__(None, None, None)
        t.tagged_time["Decode"] /= max(1, t.tagged_count["Decode"] - 1)

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    sys.stdout.flush()
    return JSONResponse(ret)


async def run(config: uvicorn.Config):
    server = uvicorn.Server(config=config)
    await server.serve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument("--log-level", type=str, default="debug")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.API_SERVER)
    logger.setLevel(30)
    app.root_path = args.root_path

        
    # from vllm.logger import enable_trace_function_call
    # log_file = f"/dev/shm/vllm_trace.log"
    # with open(log_file, "w+") as f:
    #     f.write("")
    # # enable_trace_function_call(log_file, "/data/vllm/vllm/worker")
    # enable_trace_function_call(log_file, ["/data/vllm/vllm/executor", "/data/vllm/vllm/model_executor", "/data/vllm/vllm/engine"])

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)
    # import asyncio
    # config = uvicorn.Config(
    #     app=app,
    #     host=args.host,
    #     port=args.port,
    #     log_level=args.log_level,
    # )
    # asyncio.run(run(config))