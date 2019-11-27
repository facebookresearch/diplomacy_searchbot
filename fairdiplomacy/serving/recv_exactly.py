def recv_exactly(sock, size):
    r = b""
    while len(r) < size:
        r += sock.recv(min(4096, size - len(r)))
    return r
