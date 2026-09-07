import os
import sys
import sysconfig
import subprocess
from setuptools import setup, Extension, find_packages
import torch
import torch_npu
from torch.utils.cpp_extension import BuildExtension, CppExtension

# Get PyTorch version
PYTORCH_VERSION = subprocess.check_output([sys.executable, '-c', 'import torch; print(torch.__version__.split("+")[0])']).decode('utf-8').strip()
version_parts = PYTORCH_VERSION.split('.')
PYTORCH_VERSION_DIR = f"v{version_parts[0]}r{version_parts[1]}"

# Set os env
os.environ["PYTORCH_VERSION"] = PYTORCH_VERSION
os.environ["PYTORCH_CUSTOM_DERIVATIVES_PATH"] = os.path.join(os.path.dirname(__file__), f"op-plugin/config/{PYTORCH_VERSION_DIR}/derivatives.yaml")
os.environ["ACNN_EXTENSION_PATH"] = os.path.dirname(__file__)
os.environ["ACNN_EXTENSION_SWITCH"] = "TRUE"


# Get all source files that need to be compiled
def get_sources():
    sources = []
    # 添加csrc/aten目录下的源文件
    aten_dir = os.path.join(os.path.dirname(__file__), "torch_npu/csrc/aten")
    if os.path.exists(aten_dir):
        for root, _, files in os.walk(aten_dir):
            for file in files:
                if file.endswith(".cpp") or file.endswith(".cc"):
                    sources.append(os.path.join(root, file))
    # 添加op-plugin/ops目录下的源文件
    ops_dir = os.path.join(os.path.dirname(__file__), "op_plugin")
    if os.path.exists(ops_dir):
        for root, _, files in os.walk(ops_dir):
            for file in files:
                if file.endswith(".cpp") or file.endswith(".cc"):
                    sources.append(os.path.join(root, file))

    BUILD_EXCLUDE_LIST = [
        os.path.join(aten_dir, "VariableTypeEverything.cpp"),
        os.path.join(aten_dir, "ADInplaceOrViewTypeEverything.cpp"),
        os.path.join(aten_dir, "python_functionsEverything.cpp"),
        os.path.join(aten_dir, "RegisterFunctionalizationEverything.cpp"),
        os.path.join(ops_dir, "OpInterfaceEverything.cpp"),
        os.path.join(ops_dir, "ops", "opapi", "StructKernelNpuOpApiEverything.cpp"),
    ]

    # Newer torchnpugen emits an aggregate monolith (StructKernelNpuOpApi.cpp /
    # OpInterface.cpp) *together with* per-op split files (StructKernelNpuOpApi_0.cpp,
    # OpInterface_0.cpp, ...). Compiling both causes multiple-definition at link time.
    # Only drop the aggregate when its split replacement is actually present, so older
    # single-file torchnpugen output still builds.
    for aggregate in (
        os.path.join(ops_dir, "OpInterface.cpp"),
        os.path.join(ops_dir, "ops", "opapi", "StructKernelNpuOpApi.cpp"),
    ):
        aggregate_dir = os.path.dirname(aggregate)
        prefix = os.path.splitext(os.path.basename(aggregate))[0] + "_"
        if any(
            f.startswith(prefix) and f.endswith(".cpp")
            for f in os.listdir(aggregate_dir)
        ):
            BUILD_EXCLUDE_LIST.append(aggregate)

    sources_new = []
    seen = set()
    for cur_file in sources:
        if cur_file in BUILD_EXCLUDE_LIST:
            continue
        rp = os.path.realpath(cur_file)
        if rp in seen:
            continue
        seen.add(rp)
        sources_new.append(cur_file)
    print("====sources_new:", sources_new)

    return sources_new


# Get all needed head files
def get_include_dirs():
    PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.realpath(torch_npu.__file__))

    include_dirs = []
    # Add csrc/aten path
    aten_dir = os.path.join(os.path.dirname(__file__), "torch_npu/csrc/aten")
    if os.path.exists(aten_dir):
        include_dirs.append(aten_dir)
    # Add op-plugin path
    ops_dir = os.path.join(os.path.dirname(__file__), "op_plugin")
    if os.path.exists(ops_dir):
        include_dirs.append(ops_dir)

    base_dir = os.path.dirname(__file__)
    if os.path.exists(base_dir):
        include_dirs.append(base_dir)

    torch_npu_dir = PYTORCH_NPU_INSTALL_PATH
    include_dirs.append(os.path.join(torch_npu_dir, 'include'))
    include_dirs.append(os.path.join(torch_npu_dir, 'include', 'third_party', 'acl', 'inc'))
    include_dirs.append(os.path.join(torch_npu_dir, 'include', 'third_party', 'hccl', 'inc'))
    include_dirs.append(os.path.join(torch_npu_dir, 'include', 'third_party', 'op-plugin'))
    return include_dirs


def get_compile_args():
    compile_args = ["-std=c++17"]
    # for Windows
    if sys.platform == "win32":
        compile_args.append("/MD")
    # for Linux
    elif sys.platform == "linux":
        compile_args.append("-fPIC")
    return compile_args


def get_dependency_paths():
    python_lib = sysconfig.get_config_var("LIBDIR")
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    torch_npu_path = os.path.dirname(torch_npu.__file__)
    torch_npu_lib = os.path.join(torch_npu_path, "lib")

    all_libs = list([
        python_lib,
        torch_lib,
        torch_npu_lib,
    ])

    return {
        "all_libs": all_libs
    }


def get_link_args():
    link_args = []

    link_args.append("-ltorch_npu")
    link_args.append("-ltorch")
    link_args.append("-lc10")

    dep_paths = get_dependency_paths()
    for lib_dir in dep_paths["all_libs"]:
        link_args.append(f"-L{lib_dir}")
    return link_args

# Set extension configuration
# Use CppExtension in PyTorch instead of the standard Extension for better PyTorch adaption
extensions = [
    CppExtension(
        "fla_npu.custom_aclnn_extension_lib",
        sources=get_sources(),
        include_dirs=get_include_dirs(),
        extra_compile_args=get_compile_args(),
        extra_link_args=get_link_args(),
    )
]

setup(
    name="fla_npu",
    version="1.0.0",
    description="FLA NPU extension for PyTorch",
    ext_modules=extensions,
    cmdclass={
        'build_ext': BuildExtension,
    },
    zip_safe=False,
    install_requires=[
        f"torch=={PYTORCH_VERSION}"
    ],
    # 显式列出包，避免 find_packages() 把构建目录里残留的杂包（如从 main 工作区混入的
    # fla/）一并打进 wheel：那会让卸载删除 site-packages/fla/__init__.py，而干净重建后
    # 重装无法还原，导致运行时提示缺少 __version__。
    packages=["fla_npu"],
)
