#!/usr/bin/env python3
"""Serve the local banking observability API on a loopback port."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.api import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1',
                        help='Default is localhost. There is no authentication; do not expose publicly.')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    if args.host not in ('127.0.0.1', 'localhost'):
        parser.error('Only loopback binding is supported until authentication is implemented.')
    try:
        import uvicorn
    except ImportError as exc:
        parser.error(f'Install requirements-api.txt: {exc}')
    uvicorn.run(create_app(), host='127.0.0.1', port=args.port, access_log=False)


if __name__ == '__main__':
    main()
