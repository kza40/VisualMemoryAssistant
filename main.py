# Unified CLI entry point for the Visual Memory Assistant.
# Delegates to individual pipeline scripts based on the subcommand.
#
# Usage:
#   python main.py capture          — start the webcam capture loop
#   python main.py embed            — generate CLIP embeddings for new images
#   python main.py index            — build / rebuild the FAISS index
#   python main.py ask "my query"   — search visual memory and get a VLM answer

import argparse
import sys


def build_parser():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Visual Memory Assistant",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.add_parser("capture", help="Start the webcam capture loop")
    sub.add_parser("embed",   help="Generate CLIP embeddings for new images")
    sub.add_parser("index",   help="Build or rebuild the FAISS index")
    ask = sub.add_parser("ask", help="Query visual memory and get a VLM answer")
    ask.add_argument("query", nargs="+", help="Natural language question")
    return parser


def run( args=None ):
    parsed = build_parser().parse_args(args)

    if parsed.command == "capture":
        from scripts.capture import main as _main
        _main()
    elif parsed.command == "embed":
        from scripts.embed_images import main as _main
        _main()
    elif parsed.command == "index":
        from scripts.build_index import main as _main
        _main()
    elif parsed.command == "ask":
        from scripts.query import main as _main
        _main(" ".join(parsed.query))
    else:
        build_parser().print_help()
        sys.exit(1)


if __name__ == "__main__":
    run()
