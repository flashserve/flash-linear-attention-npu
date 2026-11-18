"""Shared build/publish platform registry.

Add a platform here only after the source tree and build images support its SOC.
Both the build matrix and the publish gate use this registry, including future A6.
"""

import argparse

SOC_TO_PACKAGE = {
    "ascend910b": "flash_linear_attention_npu_a2",
    "ascend910_93": "flash_linear_attention_npu_a3",
    "ascend950": "flash_linear_attention_npu_a5",
}
ARCH_TO_WHEEL = {"arm": "aarch64", "x86": "x86_64"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--soc-list", action="store_true")
    args = parser.parse_args()
    if args.soc_list:
        print(",".join(SOC_TO_PACKAGE))
