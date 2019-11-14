import pickle
import socket
import struct
import torch

if __name__ == "__main__":
    x = (torch.ones(4, 10) + 1, torch.ones(4, 10) + 2)

    s = socket.socket()
    s.connect(("localhost", 24565))
    data = pickle.dumps(x)

    print("Sending data")
    s.send(struct.pack("Q", len(data)))
    s.send(data)
    print("Sent:", x)

    size = struct.unpack("Q", s.recv(8))[0]
    print("Awaiting {} bytes".format(size))
    raw_result = s.recv(size)
    print("Read {} bytes".format(size))

    result = pickle.loads(raw_result)
    print(result)
