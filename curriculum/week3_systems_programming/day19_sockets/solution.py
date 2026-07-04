"""Day 19 reference solution — framed TCP echo server & client."""
from __future__ import annotations

import socket
import struct


def make_listen_socket(host: str = "127.0.0.1", port: int = 0) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)
    return s


def recv_exactly(sock: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def send_message(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def recv_message(sock: socket.socket) -> bytes | None:
    header = recv_exactly(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    return recv_exactly(sock, length)


def serve_echo(listen_sock: socket.socket) -> None:
    conn, _addr = listen_sock.accept()
    try:
        while True:
            msg = recv_message(conn)
            if msg is None:
                break
            send_message(conn, msg)
    finally:
        conn.close()


def echo_client(host: str, port: int, messages: list[bytes]) -> list[bytes]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    replies: list[bytes] = []
    try:
        for m in messages:
            send_message(s, m)
            replies.append(recv_message(s))
    finally:
        s.close()
    return replies
