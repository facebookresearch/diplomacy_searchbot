import diplomacy
import pickle
import socket
import struct
import time
from collections import Counter

from fairdiplomacy.agents.dipnet_agent import encode_inputs, ORDER_VOCABULARY
from .model_server import ModelServer


class ModelClient:
    def __init__(self, port=ModelServer.DEFAULT_PORT, connect_timeout=5):
        self.s = socket.socket()

        # try to connect for up to "connect_timeout" seconds
        for _ in range(int(connect_timeout / 0.05)):
            try:
                self.s.connect(("localhost", port))
                break
            except ConnectionRefusedError as e:
                err = e
                time.sleep(0.05)
        else:
            raise err

    def synchronous_request(self, x):
        enc = pickle.dumps(x)
        self.s.send(struct.pack("Q", len(enc)))
        self.s.send(enc)
        resp_size = struct.unpack("Q", self.recv_exactly(8))[0]
        resp_enc = self.recv_exactly(resp_size)
        return pickle.loads(resp_enc)

    def recv_exactly(self, size):
        r = b""
        while len(r) < size:
            r += self.s.recv(min(4096, size - len(r)))
        return r


class DipnetModelClient(ModelClient):
    def get_orders(self, game, power, temperature=1.0):
        x = encode_inputs(game, power)
        order_idxs, = self.synchronous_request(x + [temperature])
        return [ORDER_VOCABULARY[idx] for idx in order_idxs[0, :]]

    def get_repeat_orders(self, game, power, n=100, temperature=1.0):
        x = encode_inputs(game, power)
        x = [t.repeat([n] + ([1] * (len(t.shape) - 1))) for t in x]
        order_idxs, = self.synchronous_request(x + [temperature])
        return Counter(tuple(ORDER_VOCABULARY[idx] for idx in order_idxs[i, :]) for i in range(n))


if __name__ == "__main__":
    game = diplomacy.Game()
    client = DipnetModelClient(24565)
    orders = client.get_repeat_orders(game, "ITALY", temperature=0.1)
    print(orders)
