import diplomacy
import pickle
import socket
import struct
import time
from collections import Counter

from fairdiplomacy.agents.dipnet_agent import encode_inputs, ORDER_VOCABULARY
from .model_server import ModelServer
from .recv_exactly import recv_exactly


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
        return recv_exactly(self.s, size)
