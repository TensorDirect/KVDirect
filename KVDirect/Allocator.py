import asyncio

from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.core.scheduler import Scheduler


class BlockAllocator:
    def __init__(self, block_manager: BlockSpaceManager, scheduler: Scheduler):
        self.lock = asyncio.Lock()
        self._queue = asyncio.PriorityQueue()
        self.manager = block_manager
        self.loop = asyncio.get_running_loop()
        self._task = self.loop.create_task(self._process_queue())
        self.scheduler = scheduler

    async def acquire(self, arrival_time, seq_group):
        future = self.loop.create_future()
        await self._queue.put((arrival_time, future, seq_group))
        return await future

    async def _process_queue(self):
        while True:
            arrival_time, future, seq_group = await self._queue.get()
            while True:
                can_allocate = self.manager.can_allocate(seq_group)
                # reserve cache blocks
                if can_allocate == AllocStatus.OK:
                    self.scheduler._allocate_and_set_running(seq_group)
                    future.set_result(None)
                    break
                elif can_allocate == AllocStatus.LATER:
                    await asyncio.sleep(0.1)
                else:
                    future.set_exception(RuntimeError(
                        f"Prompt is too long for {seq_group.request_id}"
                    ))
                    break
