import sys
import threading
import queue
import random
import collections

import torch
import torch.multiprocessing as multiprocessing
import itertools

from torch._C import _set_worker_signal_handlers
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter
from torch.utils.data import _utils

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue

def _ms_loop(dataset, index_queue, data_queue, collate_fn, scale, seed, init_fn, worker_id):
    global _use_shared_memory
    _use_shared_memory = True
    _set_worker_signal_handlers()

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    while True:
        r = index_queue.get()
        if r is None:
            break
        idx, batch_indices = r
        try:
            idx_scale = 0

            # train on integer scale factors (x2, x3, x4) for 1 epoch to maintain stability   #realrandom zhushidiao
            if dataset.first_epoch and len(scale) > 1 and dataset.train:
                idx_integer_scale_list = [9, 19, 29]
                rand_idx = random.randrange(0, len(idx_integer_scale_list))
                dataset.set_scale(idx_integer_scale_list[rand_idx])

            # train on all scale factors for remaining epochs
            if not dataset.first_epoch and len(scale) > 1 and dataset.train:
                idx_scale = random.randrange(0, len(scale))
                dataset.set_scale(idx_scale)
            # if len(scale) > 1 and dataset.train:
            #     idx_scale = random.randrange(0, len(scale))
            #     dataset.set_scale(idx_scale)

            samples = collate_fn([dataset[i] for i in batch_indices])
            samples.append(idx_scale)

        except Exception:
            data_queue.put((idx, _utils.ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))


class _MSDataLoaderIter(_MultiProcessingDataLoaderIter):
    def __init__(self, loader):
        super(_MultiProcessingDataLoaderIter, self).__init__(loader)

        assert self._num_workers > 0
        assert self._prefetch_factor > 0
        base_seed = torch.LongTensor(1).random_()[0]
        self._scale = loader.scale
        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context

        self._worker_init_fn = loader.worker_init_fn
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        # No certainty which module multiprocessing_context is
        self._worker_result_queue = multiprocessing_context.Queue()  # type: ignore
        self._worker_pids_set = False
        self._shutdown = False
        self._workers_done_event = multiprocessing_context.Event()

        self._index_queues = []
        self._workers = []
        for i in range(self._num_workers):
            # No certainty which module multiprocessing_context is
            index_queue = multiprocessing_context.Queue()  # type: ignore
            # Need to `cancel_join_thread` here!
            # See sections (2) and (3b) above.
            index_queue.cancel_join_thread()
            w = multiprocessing_context.Process(
                target=_ms_loop,
                    args=(
                        self._dataset,
                        index_queue,
                        self._worker_result_queue,
                        self._collate_fn,
                        self._scale,
                        base_seed + i,
                        self._worker_init_fn,
                        i
                    )
                )

                      
            w.daemon = True
            # NB: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self._workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)

        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()

            # Queue is not type-annotated
            self._data_queue = queue.Queue()  # type: ignore[var-annotated]
            pin_memory_thread = threading.Thread(
                target=_utils.pin_memory._pin_memory_loop,
                args=(self._worker_result_queue, self._data_queue,
                      torch.cuda.current_device(),
                      self._pin_memory_thread_done_event, self._pin_memory_device))
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            # Similar to workers (see comment above), we only register
            # pin_memory_thread once it is started.
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue

        # .pid can be None only before process is spawned (not the case, so ignore)
        _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self._workers))  # type: ignore
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True
        self._reset(loader, first_iter=True)


class MSDataLoader(DataLoader):
    def __init__(
        self, args, dataset, batch_size=1, shuffle=False,
        sampler=None, batch_sampler=None,
        collate_fn=_utils.collate.default_collate, pin_memory=False, drop_last=False,
        timeout=0, worker_init_fn=None):
        super(MSDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle,
            sampler=sampler, batch_sampler=batch_sampler,
            num_workers=args.n_threads, collate_fn=collate_fn,
            pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn)

        self.scale = args.scale

    def __iter__(self):
        return _MSDataLoaderIter(self)
