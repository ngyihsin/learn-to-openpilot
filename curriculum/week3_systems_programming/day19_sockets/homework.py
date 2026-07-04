"""Day 19 homework — TCP sockets: a framed echo server & client.

A socket is a file descriptor you can send bytes through to another program — on the same
machine or across the world. TCP gives you a reliable, ordered **byte stream**, which means (just
like the pipe on Day 14) `recv` can return a partial message or several at once. So you reuse the
same idea: length-prefix each message and read *exactly* that many bytes.

You'll build the framing helpers, a server that echoes framed messages back, and a client. The
`make_listen_socket` helper (bind + listen) is provided.

Fill in every ``TODO`` and run ``pytest -q``.
"""
from __future__ import annotations

import socket
import struct


def make_listen_socket(host: str = "127.0.0.1", port: int = 0) -> socket.socket:
    """Create a listening TCP socket. port=0 lets the OS pick a free port; read the real port
    back with ``sock.getsockname()[1]``."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)
    return s


def recv_exactly(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly `n` bytes from `sock`, looping because recv may return fewer. Return None if
    the peer closes before `n` bytes arrive."""
    # TODO: accumulate chunks from sock.recv(n - len(buf)) until you have n bytes; if recv
    #       returns b"" (peer closed), return None
    raise NotImplementedError


def send_message(sock: socket.socket, payload: bytes) -> None:
    """Send a length-prefixed frame: 4-byte big-endian length, then the payload. Use sendall."""
    # TODO: sock.sendall(struct.pack(">I", len(payload)) + payload)
    raise NotImplementedError


def recv_message(sock: socket.socket) -> bytes | None:
    """Receive one framed message (or None if the connection closed cleanly first)."""
    # TODO: recv_exactly 4 bytes for the length; if None return None; unpack; recv_exactly length
    raise NotImplementedError


def serve_echo(listen_sock: socket.socket) -> None:
    """Accept ONE client and echo every framed message back until the client disconnects."""
    # TODO:
    #   conn, _addr = listen_sock.accept()
    #   loop: msg = recv_message(conn); if msg is None: break; send_message(conn, msg)
    #   close conn at the end
    raise NotImplementedError


def echo_client(host: str, port: int, messages: list[bytes]) -> list[bytes]:
    """Connect, send each message, collect each echoed reply in order, and return them."""
    # TODO: create a socket, connect((host, port)); for each message send_message then
    #       recv_message, collecting replies; close the socket; return the list
    raise NotImplementedError
