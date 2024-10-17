# def connect():
#     pass

# _prefill_addr = [
#     ("127.0.0.1", 8001)
#     # "10.121.14.226"
# ]


candidate = ("10.117.208.36", "10.117.208.39")

def get_prefill_worker(i):
    return (candidate[i % len(candidate)], 8001)
