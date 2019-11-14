import asyncio
import logging
import pickle
import struct
import torch

logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG)


class ModelServer:
    def __init__(self, model, port, max_batch_size, max_batch_latency):
        self.model = model
        self.port = port
        self.max_batch_size = max_batch_size
        self.max_batch_latency = max_batch_latency

        self.current_batch = None
        self.current_batch_sizes = []
        self.current_batch_futures = []
        self.timeout_flush_task = None

    def start(self):
        asyncio.run(self.serve_forever())

    async def serve_forever(self):
        logging.info("Listening on port {}".format(self.port))
        server = await asyncio.start_server(self.handle_conn, "localhost", self.port)
        async with server:
            await server.serve_forever()

    async def handle_conn(self, reader, writer):
        raw_size = struct.unpack("Q", await reader.read(8))[0]
        raw = await reader.read(raw_size)
        batch = pickle.loads(raw)
        batch_size = batch[0].shape[0]
        logging.debug("Got batch of size {}".format(batch_size))

        if batch_size > self.max_batch_size:
            raise RuntimeError(
                "{} is bigger than max size of {}".format(batch_size, self.max_batch_size)
            )

        if sum(self.current_batch_sizes) + batch_size > self.max_batch_size:
            self.flush_current_batch()

        future = self.append_to_current_batch(batch)

        if sum(self.current_batch_sizes) == self.max_batch_size:
            self.flush_current_batch()

        result = await future

        result_pickled = pickle.dumps(result)
        writer.write(struct.pack("Q", len(result_pickled)))
        writer.write(result_pickled)
        await writer.drain()
        writer.close()

    def append_to_current_batch(self, batch) -> asyncio.Future:
        if self.current_batch is None:
            self.current_batch = batch
            assert self.timeout_flush_task is None
            self.timeout_flush_task = asyncio.create_task(self.scheduled_timeout_flush())
        else:
            self.current_batch = tuple(
                torch.cat([cur, x]) for cur, x in zip(self.current_batch, batch)
            )

        self.current_batch_sizes.append(batch[0].shape[0])

        future = asyncio.get_running_loop().create_future()
        self.current_batch_futures.append(future)
        return future

    def flush_current_batch(self):
        logging.debug("Flushing current batch, sizes {}".format(self.current_batch_sizes))

        if self.timeout_flush_task is not None:
            self.timeout_flush_task.cancel()
            self.timeout_flush_task = None

        y = self.model(*self.current_batch)

        i = 0
        for size, future in zip(self.current_batch_sizes, self.current_batch_futures):
            batch_y = tuple(t[i : (i + size)] for t in y)
            future.set_result(batch_y)
            i += size

        self.current_batch = None
        self.current_batch_sizes = []
        self.current_batch_futures = []

    async def scheduled_timeout_flush(self):
        try:
            await asyncio.sleep(self.max_batch_latency)
            logging.debug("Scheduled flush!")
            self.timeout_flush_task = None
            self.flush_current_batch()
        except asyncio.CancelledError:
            pass


class SquareModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tensor = torch.zeros(1)

    def forward(self, x, y):
        return torch.pow(x + self.tensor, 2), torch.pow(y + self.tensor, 2)


if __name__ == "__main__":
    # from fairdiplomacy.models.dipnet.train_sl import new_model

    MODEL_PTH = "/checkpoint/jsgray/dipnet.20103672.pth"
    PORT = 24565
    MAX_BATCH_SIZE = 8
    MAX_BATCH_LATENCY = 0.1

    # model = new_model()
    # model.load_state_dict(torch.load(MODEL_PTH, map_location="cuda")["model"])
    # model.eval()
    model = SquareModel().to("cuda")

    model_server = ModelServer(model, PORT, MAX_BATCH_SIZE, MAX_BATCH_LATENCY)
    model_server.start()
