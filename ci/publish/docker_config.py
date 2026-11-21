"""Use the exact preloaded image references selected by repository CI.

These legacy tag names currently alias CANN 9.2.1 images. Do not infer the
installed CANN version from the tag; build_local records and pins the image ID.
The CI image owner upgrades these images. Publishing never rebuilds them.
"""
import argparse
import os

ARM_IMAGE = 'fla-npu-ci:9.1.0-910b'
X86_IMAGE = 'fla-npu-ci:9.1.0-950'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arch', choices=('arm', 'x86'))
    args = parser.parse_args()
    variable = 'FLA_PKG_DOCKER_IMAGE' if args.arch == 'arm' else 'FLA_PKG_X86_DOCKER_IMAGE'
    default = ARM_IMAGE if args.arch == 'arm' else X86_IMAGE
    print(os.getenv(variable, '').strip() or default)
