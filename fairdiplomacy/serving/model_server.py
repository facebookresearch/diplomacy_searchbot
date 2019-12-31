import asyncio
import logging
import os
import pickle
import struct
import time
import torch

from collections import defaultdict

logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG)


class ModelServer:
    DEFAULT_PORT = 24565
    DEFAULT_MAX_BATCH_LATENCY = 0.1

    def __init__(
        self,
        load_model_fn,
        max_batch_size,
        max_batch_latency=DEFAULT_MAX_BATCH_LATENCY,
        port=DEFAULT_PORT,
        output_transform=None,
        seed=None,
        log_stats_every=5,
        start=False,
    ):
        """A minimal TCP server serving pytorch models via a simple pickle protocol.
        Handles batching and single-GPU usage.

        Protocol: All payloads are pickled python objects preceded by the
        payload length as a 64-bit unsigned int in python-default endianness,
        e.g. struct.pack("Q", ...)

        Request payload: a tuple or list of input pytorch tensors to pass to model
        Response payload: output_transform(model(*x))

        Server will accept payloads for `max_batch_latency` seconds, then cat
        the inputs, pass them to the model, and return the outputs.

        Arguments:
        - load_model_fn: function called with no args, must return a pytorch model
        - max_batch_size: flush buffer if sum of request batch sizes >= max_batch_size
        - max_batch_latency: flush buffer if any request has waited this long
        - port: to listen for TCP connections
        - output_transform: Optionally, a function applied to the model output before returning
        - seed: if not None, call torch.manual_seed(seed)
        - log_stats_every: period in seconds to log perf stats
        - if start is True, call start() after __init__()
        """
        logging.info(f"ModelServer __init__. batch= {max_batch_size}, latency= {max_batch_latency}")
        self.model = load_model_fn()
        # logging.info(f"Model: {self.model}")
        self.port = port
        self.max_batch_size = max_batch_size
        self.max_batch_latency = max_batch_latency
        self.output_transform = output_transform
        self.log_stats_every = log_stats_every

        self.buf_batch = []
        self.buf_batch_sizes = []
        self.buf_batch_futures = []
        self.timeout_flush_task = None

        self.stat_throughput_numerator = 0
        self.stat_batch_count = 0
        self.timings = defaultdict(float)

        if seed is not None:
            torch.manual_seed(seed)

        if start:
            self.start()

    def start(self):
        asyncio.run(self.serve_forever())

    async def serve_forever(self):
        asyncio.create_task(self.do_log_stats_every(self.log_stats_every))
        logging.info("Listening on port {}".format(self.port))
        server = await asyncio.start_server(self.handle_conn, "localhost", self.port)
        async with server:
            await server.serve_forever()

    async def handle_conn(self, reader, writer):
        while True:
            try:
                raw_size_enc = await reader.readexactly(8)
                if raw_size_enc == b"":
                    return
            except ConnectionResetError:
                return
            except asyncio.streams.IncompleteReadError:
                return

            raw_size = struct.unpack("Q", raw_size_enc)[0]
            raw = await reader.readexactly(raw_size)
            batch = pickle.loads(raw)
            batch_size = batch[0].shape[0]

            if batch_size > self.max_batch_size:
                raise RuntimeError(
                    "{} is bigger than max size of {}".format(batch_size, self.max_batch_size)
                )

            if sum(self.buf_batch_sizes) + batch_size > self.max_batch_size:
                self.flush_buf_batch()

            future = self.append_to_buf_batch(batch)

            if sum(self.buf_batch_sizes) == self.max_batch_size:
                self.flush_buf_batch()

            result = await future

            result_pickled = pickle.dumps(result)
            writer.write(struct.pack("Q", len(result_pickled)))
            writer.write(result_pickled)
            await writer.drain()

    def schedule_timeout(self):
        if self.timeout_flush_task is None and len(self.buf_batch) > 0:
            self.timeout_flush_task = asyncio.create_task(self.scheduled_timeout_flush())

    def append_to_buf_batch(self, batch) -> asyncio.Future:
        self.buf_batch.append(batch)
        self.buf_batch_sizes.append(batch[0].shape[0])

        self.schedule_timeout()

        future = asyncio.get_running_loop().create_future()
        self.buf_batch_futures.append(future)
        return future

    def flush_buf_batch(self):
        tic = time.time()

        with torch.no_grad():
            xs = [torch.cat(ts).to("cuda") for ts in zip(*self.buf_batch)]
            self.timings['to_cuda'] += time.time() - tic; tic = time.time()
            tic = time.time()

            y = self.model(*xs)
            self.timings['forward'] += time.time() - tic; tic = time.time()


        if self.output_transform is not None:
            y = self.output_transform(y)
        self.timings['transform'] += time.time() - tic; tic = time.time()

        y = tuple(t.to("cpu") for t in y)
        self.timings['to_cpu'] += time.time() - tic; tic = time.time()

        i = 0
        for size, future in zip(self.buf_batch_sizes, self.buf_batch_futures):
            batch_y = tuple(t[i : (i + size)] for t in y)
            future.set_result(batch_y)
            i += size
        self.timings['send'] += time.time() - tic; tic = time.time()

        self.stat_throughput_numerator += i
        self.stat_batch_count += 1
        self.buf_batch = []
        self.buf_batch_sizes = []
        self.buf_batch_futures = []

    async def scheduled_timeout_flush(self):
        try:
            await asyncio.sleep(self.max_batch_latency)

            self.flush_buf_batch()
            self.timeout_flush_task = None  # allow a new timeout to be scheduled
            self.schedule_timeout()
        except asyncio.CancelledError:
            pass

    async def do_log_stats_every(self, period):
        last_logged_time = time.time()
        while True:
            await asyncio.sleep(period)
            now = time.time()
            delta = now - last_logged_time
            stat_batch_count = self.stat_batch_count + 1e-8

            logging.info(
                "Throughput: {} evals / {:.5} s = {:.5} evals/s.   (PID {})".format(
                    self.stat_throughput_numerator,
                    delta,
                    self.stat_throughput_numerator / delta,
                    os.getpid()
                )
            )
            logging.info(
                "            {} batches of avg size {:.5}; {:.5} s/batch".format(
                    self.stat_batch_count,
                    self.stat_throughput_numerator / stat_batch_count,
                    delta / stat_batch_count,
                )
            )
            self.timings['wait'] = delta - sum(self.timings.values())
            logging.info({k : "{:.3}".format(v / (self.stat_batch_count + 1e-8)) for k, v in self.timings.items()})
            self.timings.clear()
            self.stat_throughput_numerator = 0
            self.stat_batch_count = 0
            last_logged_time = now


if __name__ == "__main__":
    import argparse
    from fairdiplomacy.models.dipnet.load_model import load_dipnet_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=ModelServer.DEFAULT_PORT)
    parser.add_argument("--max-batch-latency", type=float, default=0.01)
    args = parser.parse_args()

    MODEL_PTH = "/checkpoint/jsgray/dipnet.pth"
    MAX_BATCH_SIZE = 1000

    def load_model():
        return load_dipnet_model(MODEL_PTH, map_location="cuda", eval=True).cuda()

    model_server = ModelServer(
        load_model,
        MAX_BATCH_SIZE,
        args.max_batch_latency,
        port=args.port,
        output_transform=lambda y: y[:1],  # return only order_idxs, not order_scores
        seed=0,
    )
    model_server.start()
