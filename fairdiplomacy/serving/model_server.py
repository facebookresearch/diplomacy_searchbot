import asyncio
import logging
import pickle
import struct
import torch

logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG)


class ModelServer:
    DEFAULT_PORT = 24565
    DEFAULT_MAX_BATCH_LATENCY = 0.1

    def __init__(
        self,
        model,
        max_batch_size,
        max_batch_latency=DEFAULT_MAX_BATCH_LATENCY,
        port=DEFAULT_PORT,
        output_transform=None,
        seed=None,
    ):
        """A minimal TCP server serving pytorch models via a simple pickle protocol.
        Handles batching and single-GPU usage.

        Protocol: All payloads are pickled python objects preceded by the
        payload length as a 64-bit unsigned int in python-default endianness,
        e.g. struct.pack("Q", ...)

        Request payload: a tuple or list of input pytorch tensors to pass to model
        Response payload: output_transform(model(*x))

        Arguments:
        - model: a pytorch model
        - max_batch_size: flush buffer if sum of request batch sizes >= max_batch_size
        - max_batch_latency: flush buffer if any request has waited this long
        - port: to listen for TCP connections
        - output_transform: Optionally, a function applied to the model output before returning
        - seed: if not None, call torch.manual_seed(seed)
        """
        self.model = model
        self.port = port
        self.max_batch_size = max_batch_size
        self.max_batch_latency = max_batch_latency
        self.output_transform = output_transform

        self.buf_batch = None
        self.buf_batch_sizes = []
        self.buf_batch_futures = []
        self.timeout_flush_task = None

        if seed is not None:
            torch.manual_seed(seed)

    def start(self):
        asyncio.run(self.serve_forever())

    async def serve_forever(self):
        logging.info("Listening on port {}".format(self.port))
        server = await asyncio.start_server(self.handle_conn, "localhost", self.port)
        async with server:
            await server.serve_forever()

    async def handle_conn(self, reader, writer):
        logging.debug("New incoming conn")
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
            logging.debug("Got batch of size {}".format(batch_size))

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

    def append_to_buf_batch(self, batch) -> asyncio.Future:
        if self.buf_batch is None:
            self.buf_batch = batch
            assert self.timeout_flush_task is None
            self.timeout_flush_task = asyncio.create_task(self.scheduled_timeout_flush())
        else:
            try:
                self.buf_batch = tuple(
                    torch.cat([cur, x]) for cur, x in zip(self.buf_batch, batch)
                )
            except:
                import ipdb

                ipdb.set_trace()

        self.buf_batch_sizes.append(batch[0].shape[0])

        future = asyncio.get_running_loop().create_future()
        self.buf_batch_futures.append(future)
        return future

    def flush_buf_batch(self):
        logging.debug("Flushing buf batch, sizes {}".format(self.buf_batch_sizes))

        if self.timeout_flush_task is not None:
            self.timeout_flush_task.cancel()
            self.timeout_flush_task = None

        with torch.no_grad():
            y = self.model(*[t.to("cuda") for t in self.buf_batch])  # TODO: possible to do async?

        if self.output_transform is not None:
            y = self.output_transform(y)

        y = tuple(t.to("cpu") for t in y)

        i = 0
        for size, future in zip(self.buf_batch_sizes, self.buf_batch_futures):
            batch_y = tuple(t[i : (i + size)] for t in y)
            future.set_result(batch_y)
            i += size

        self.buf_batch = None
        self.buf_batch_sizes = []
        self.buf_batch_futures = []

    async def scheduled_timeout_flush(self):
        try:
            await asyncio.sleep(self.max_batch_latency)
            logging.debug("Scheduled flush!")
            self.timeout_flush_task = None
            self.flush_buf_batch()
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    from fairdiplomacy.models.dipnet.load_model import load_dipnet_model

    MODEL_PTH = "/checkpoint/jsgray/dipnet.20103672.pth"
    PORT = 24565
    MAX_BATCH_SIZE = 1000
    MAX_BATCH_LATENCY = 0.05

    model = load_dipnet_model(MODEL_PTH, map_location="cuda", eval=True).cuda()

    # return only order_idxs, not order_scores
    output_transform = lambda y: y[:1]

    model_server = ModelServer(
        model, MAX_BATCH_SIZE, MAX_BATCH_LATENCY, output_transform=output_transform, seed=0
    )
    model_server.start()
