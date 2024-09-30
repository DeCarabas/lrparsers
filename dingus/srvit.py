#!/usr/bin/env python3
import os

from http.server import test, SimpleHTTPRequestHandler, ThreadingHTTPServer


class MyHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extensions_map[".js"] = "application/javascript"
        self.extensions_map[".wasm"] = "application/wasm"

    def end_headers(self):
        self.send_my_headers()
        super().end_headers()

    def send_my_headers(self):
        if "pyodide" not in self.path:
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")

        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")


if __name__ == "__main__":
    import argparse
    import contextlib

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--bind",
        metavar="ADDRESS",
        help="bind to this address " "(default: all interfaces)",
    )
    parser.add_argument(
        "-d",
        "--directory",
        default=os.getcwd(),
        help="serve this directory " "(default: current directory)",
    )
    parser.add_argument(
        "-p",
        "--protocol",
        metavar="VERSION",
        default="HTTP/1.0",
        help="conform to this HTTP version " "(default: %(default)s)",
    )
    parser.add_argument(
        "port",
        default=8000,
        type=int,
        nargs="?",
        help="bind to this port " "(default: %(default)s)",
    )
    args = parser.parse_args()
    handler_class = MyHTTPRequestHandler

    # ensure dual-stack is not disabled; ref #38907
    class DualStackServer(ThreadingHTTPServer):

        def server_bind(self):
            # suppress exception when protocol is IPv4
            with contextlib.suppress(Exception):
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            return super().server_bind()

        def finish_request(self, request, client_address):
            self.RequestHandlerClass(request, client_address, self, directory=args.directory)

    test(
        HandlerClass=handler_class,
        ServerClass=DualStackServer,
        port=args.port,
        bind=args.bind,
        protocol=args.protocol,
    )
