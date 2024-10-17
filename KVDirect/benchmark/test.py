import aiohttp
import asyncio
from transformers import AutoTokenizer
import argparse
import os
import time


async def benchmark(host, payload, qps, num_request):
    interval = 1.0 / qps
    hosts = host.split(',')

    async def send_request(session, i, url, payload):
        try:
            start_time = time.time()
            async with session.post(url, json=payload) as response:
                elapsed = time.time() - start_time
                print(i, response.status, elapsed)
                data = await response.json()
                return (i, response.status, elapsed, data)
        except Exception as e:
            print(e)
            return (i, None, None, str(e))

    async def producer():
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(num_request):
                url = f"http://{hosts[i%len(hosts)]}:{args.port}/generate"
                tasks.append(asyncio.create_task(send_request(session, i, url, payload)))
                await asyncio.sleep(interval)
            results = await asyncio.gather(*tasks)
            return [r[-1] for r in results]

    return await producer()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--text-file", type=str, required=True)
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--response-length", type=int, required=True)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--qps", type=float, default=1)
    parser.add_argument("--num-request", type=int, default=10)
    parser.add_argument("--message", type=str, default="")
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    with open(args.text_file, "r") as f:
        text = f.read()
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"][0]

    if len(input_ids) >= args.prompt_length:
        prompt_id = input_ids[: args.prompt_length - 1]
    assert len(input_ids) >= args.prompt_length
    # else:
    #     repeat_count = (args.prompt_length + len(input_ids) - 1) // len(input_ids)
    #     prompt_id = (input_ids * repeat_count)[:args.prompt_length]
    
    prompt = tokenizer.decode(prompt_id)
    payload = {
        "message": args.message,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": args.response_length,
    }

    data = asyncio.run(benchmark(args.host, payload, args.qps, args.num_request))
    model_name = args.model.split("/")[-1]
    path = os.path.join(args.out_dir, f"{model_name}_prompt{args.prompt_length}_resp{args.response_length}_qps{args.qps}.txt")
    print(path)
    with open(path, "w+") as f:
        for res in data:
            E2E = res['metric']['total']
            TTFT = res['metric']['Prefill']['time']
            TBT = res['metric']['Decode']['time'] / max(1, res['metric']['Decode']['count'] - 1)
            f.writelines([
                f"[{res['request_id']}]: {E2E}\n",
                f"[{res['request_id']}]:Prefill {TTFT}\n",
                f"[{res['request_id']}]:Decode {TBT}\n"
            ])
