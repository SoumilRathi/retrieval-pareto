#!/usr/bin/env python3
"""Local dev server for site/ that disables aggressive browser caching.

Default Python http.server lets Chrome cache JS/CSS aggressively, which means
edits in flight don't show up after a normal reload. This wrapper sends
no-store cache headers so every reload fetches fresh assets.
"""

from __future__ import annotations

import argparse
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--dir", default="site")
    args = parser.parse_args()

    os.chdir(args.dir)
    server = HTTPServer((args.host, args.port), NoCacheHandler)
    print(f"Serving {args.dir}/ at http://{args.host}:{args.port}/ (no-cache)")
    server.serve_forever()


if __name__ == "__main__":
    main()
