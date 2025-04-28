
import aiohttp
import asyncio
from transformers import AutoTokenizer
from datasets import load_dataset, disable_progress_bar
import argparse
import os
import time
import json
import numpy as np

disable_progress_bar()


def arxiv_dataset():
    payloads = []
    dataset = load_dataset("ccdv/arxiv-summarization")
    lengths = np.array([len(article) for article in dataset['test']['article']])
    threshold = np.percentile(lengths, 99)
    indices = np.array([i for i, l in enumerate(lengths) if l >= threshold])
    dataset = dataset['test'].select(indices)
    tokenized_abstract = dataset.map(lambda x: tokenizer(x['abstract']), batched=True)
    for i, abstract in enumerate(tokenized_abstract):
        payloads.append({
            "message": args.message,
            "prompt": dataset['article'][i],
            "temperature": 0.0,
            "max_tokens": len(tokenized_abstract['input_ids'][i]),
        })
    return payloads

def share_gpt():
    payloads = []
    def parse(data):
        conversations = data['conversations']
        context = conversations[:-1]
        context = ', '.join(json.dumps(c) for c in context)
        answer = conversations[-1]
        answer = json.dumps(answer)
        return {'context': context, 'answer': answer}
    dataset = load_dataset("theblackcat102/sharegpt-english")
    parsed_dataset = dataset['train'].map(parse, num_proc=16)
    contexts = parsed_dataset['context']
    responses = parsed_dataset['answer']
    c_len = np.array([len(context) for context in contexts])
    r_len = np.array([len(response) for response in responses])

    context_left_threshold = np.percentile(c_len, 100)
    context_right_threshold = np.percentile(c_len, 90)
    response_left_threshold = np.percentile(r_len, 100)
    response_right_threshold = np.percentile(r_len, 99)

    indices = np.array([i for i, (cl, rl) in enumerate(zip(c_len, r_len)) 
                        if context_right_threshold < cl < context_left_threshold and 
                        response_right_threshold < rl < response_left_threshold])

    sampled_dataset = parsed_dataset.select(indices)
    tokenized_response = sampled_dataset.map(lambda x: tokenizer(x['answer']), batched=True)
    for i in range(len(sampled_dataset)):
        payloads.append({
            "message": args.message,
            "prompt": sampled_dataset['context'][i],
            "temperature": 0.0,
            "max_tokens": len(tokenized_response['input_ids'][i]),
        })
    return payloads


async def benchmark(host, payloads, qps, num_request):
    intervals = np.random.exponential(1.0 / qps, num_request)
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
                tasks.append(asyncio.create_task(send_request(session, i, url, payloads[i])))
                if i < num_request - 1:
                    await asyncio.sleep(intervals[i])
            results = await asyncio.gather(*tasks)
            return [r[-1] for r in results]

    return await producer()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    # parser.add_argument("--prompt-length", type=int, required=True)
    # parser.add_argument("--response-length", type=int, required=True)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--qps", type=float, default=1)
    parser.add_argument("--num-request", type=int, default=-1)
    parser.add_argument("--message", type=str, default="")
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    payloads = []

    if args.dataset == "arxiv":
        arxiv_dataset()
    if args.dataset == "sharegpt":
        share_gpt()
        
    num_request = len(payloads) if args.num_request == -1 else args.num_request
    data = asyncio.run(benchmark(args.host, payloads, args.qps, num_request))
    model_name = args.model.split("/")[-1]
    path = os.path.join(args.out_dir, f"{model_name}_{args.dataset}_qps{args.qps}.txt")
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
