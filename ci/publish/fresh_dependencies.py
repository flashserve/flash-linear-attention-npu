"""Compatibility entry point for clean, complete dependency preparation."""
from prepare_publish_dependencies import main

if __name__ == "__main__":
    import sys
    main(["--fresh", *sys.argv[1:]])
