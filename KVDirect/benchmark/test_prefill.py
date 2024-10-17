import requests
from transformers import AutoTokenizer
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import time
from vllm.utils import random_uuid
import copy


def benchmark(url, payload, qps, num_request):
    task_queue = Queue()
    result_queue = Queue()

    def producer():
        interval = 1.0 / qps
        for i in range(num_request):
            data = copy.deepcopy(payload)
            data["request_id"] = random_uuid()
            task_queue.put((url, data))
            time.sleep(interval)

        # Signal the end of production
        for _ in range(qps):
            task_queue.put(None)

    def consumer():
        while True:
            task = task_queue.get()
            if task is None:
                break
            url, payload = task
            try:
                response = requests.post(url, json=payload)
                print(response.status_code, response.elapsed.total_seconds())
                result_queue.put(
                    (response.status_code, response.elapsed.total_seconds())
                )
            except Exception as e:
                print(e)
                result_queue.put((None, None, str(e)))

    with ThreadPoolExecutor() as executor:
        producer_thread = executor.submit(producer)
        consumer_threads = [executor.submit(consumer) for _ in range(qps)]
        producer_thread.result()
        # Wait for all consumer threads to finish
        for future in as_completed(consumer_threads):
            future.result()

    # total_requests = 0
    # total_time = 0
    # error_count = 0

    # while not result_queue.empty():
    #     status_code, elapsed_time, *error = result_queue.get()
    #     if error:
    #         error_count += 1
    #     else:
    #         total_requests += 1
    #         total_time += elapsed_time

    # avg_response_time = total_time / total_requests if total_requests else float('inf')
    # error_rate = error_count / (total_requests + error_count) if (total_requests + error_count) else 1

    # print(f"Total Requests: {total_requests}")
    # print(f"Total Errors: {error_count}")
    # print(f"Average Response Time: {avg_response_time:.2f} seconds")
    # print(f"Error Rate: {error_rate:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--text-file", type=str, required=True)
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--response-length", type=int, required=True)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--qps", type=int, default=1)
    parser.add_argument("--num-request", type=int, default=10)
    args = parser.parse_args()
    url = f"http://{args.host}:{args.port}/prefill"

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    with open(args.text_file, "r") as f:
        text = f.read()
    inputs = tokenizer(text, return_tensors="pt")
    assert len(inputs["input_ids"][0]) >= args.prompt_length, "Prompt is short"
    input_ids = inputs["input_ids"][0]
    prompt_id = input_ids[: args.prompt_length]
    prompt = tokenizer.decode(prompt_id)
    payload = {
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": args.response_length,
    }

    benchmark(url, payload, args.qps, args.num_request)
