#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ============================================================================

set -euo pipefail

run_command() {
    local cmd="$*"
    echo "Executing command: $cmd"

    if ! output=$("$@" 2>&1); then
         local exit_code=$?
         echo -e "\nCommand execution failed!"
         echo -e "\nFailed command: $cmd"
         echo -e "\nError output: $output"
         echo -e "\nExit code: $exit_code"
         exit $exit_code
    fi
}

#* 如果有 sudo 则使用 sudo
if command -v sudo &>/dev/null && [ "$EUID" -ne 0 ]; then
    try_sudo="sudo"
else
    try_sudo=""
fi


version_ge() {
    # Version comparison, format: xx.xx.xx
    IFS='.' read -r -a curr_arr <<< "$1"
    IFS='.' read -r -a req_arr <<< "$2"

    for ((i=0; i<${#req_arr[@]}; i++)); do
        curr=${curr_arr[i]:-0}
        req=${req_arr[i]}
        if (( curr > req )); then
            return 0
        elif (( curr < req )); then
            return 1
        fi
    done
    return 0
}

detect_os() {
    # OS detection, supports debian (uses apt), rhel (uses dnf or yum), openEuler(uses dnf), macos
    if [[ "$(uname -s)" == "Linux" ]]; then
        if [[ -f /etc/debian_version ]]; then
            OS="debian"
            PKG_MANAGER="apt"
        elif [[ -f /etc/redhat-release ]]; then
            OS="rhel"
            if command -v dnf &> /dev/null; then
                PKG_MANAGER="dnf"
            else
                PKG_MANAGER="yum"
            fi
        elif [[ -f /etc/euleros-release ]] || [[ -f /etc/openEuler-release ]]; then
            OS="euler"
            if command -v dnf &> /dev/null; then
                PKG_MANAGER="dnf"
            else
                PKG_MANAGER="yum"
            fi
        else
            echo "Unsupported OS type, please install manually"
            exit 1
        fi
    elif [[ "$(uname -s)" == "Darwin" ]]; then
        OS="macos"
        if ! command -v brew &> /dev/null; then
            echo "Please install Homebrew first"
            exit 1
        fi
        PKG_MANAGER="brew"
    else
        echo "Unsupported OS type"
        exit 1
    fi
    #EulerOS(uses dnf) version >= 2.10
    if [[ -f /etc/euleros-release ]]; then
        euler_version=$(grep '^VERSION_ID=' /etc/os-release | cut -d'"' -f2)
        if version_ge "2.10" "$euler_version"; then
            echo "Unsupported EulerOS $euler_version (<2.10) type"
            exit 1
        fi
    fi
}

install_gawk() {
    echo -e "\n==== Checking gawk ===="

    if command -v gawk &> /dev/null; then
        echo "gawk has been installed"
        return
    fi

    echo "Installing gawk..."
    case "$OS" in
        debian)
            run_command $try_sudo $PKG_MANAGER update
            run_command $try_sudo $PKG_MANAGER install -y gawk
            ;;
        rhel|euler)
            run_command $try_sudo $PKG_MANAGER install -y gawk
            ;;
        macos)
            run_command brew install gawk
            ;;
    esac

    if command -v gawk &> /dev/null; then
        echo "gawk installed successfully"
    else
        echo "gawk installation failed"
        exit 1
    fi
}

install_python_deps() {
    echo -e "\n==== Installing CANN 8.5 required Python packages ===="

    if python3 -c "import numpy" &>/dev/null; then
        # Get current version
        local current_ver
        current_ver=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null)
        if [ -n "$current_ver" ]; then
            echo "numpy $current_ver already installed, skipping pinned version."
            return
        fi
    fi

    # Let pip handle redundancy — it's safe and idempotent
    pip3 install \
        "numpy>=1.21.6" \
        "sympy>=1.10.1" \
        "psutil>=5.9" \
        "scipy>=1.7.3" \
        cloudpickle \
        ml-dtypes \
        tornado \
        absl-py \
        "decorator>=5.1.0" \
        attrs \
        jinja2 \
        mpmath \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        --trusted-host pypi.tuna.tsinghua.edu.cn \
        --no-deps \
        --timeout=60
    echo "CANN Python dependencies installed."
}

install_python() {
    # Python version >= 3.7.0
    echo -e "\n==== Checking Python ===="
    local req_ver="3.7.0"
    local curr_ver=""

    if command -v python3 &> /dev/null; then
        curr_ver=$(python3 --version 2>&1 | awk '{print $2}')
        echo "Current Python version: $curr_ver"
        if version_ge "$curr_ver" "$req_ver"; then
            echo "Python version meets requirements"
            return
        fi
    fi
    echo "Installing Python..."
    case "$OS" in
        debian)
            run_command $try_sudo $PKG_MANAGER update
            run_command $try_sudo $PKG_MANAGER install -y python3 python3-pip python3-dev
            ;;
        rhel)
            if grep -q "release 7" /etc/redhat-release; then
                run_command $try_sudo $PKG_MANAGER install -y centos-release-scl
                run_command $try_sudo $PKG_MANAGER install -y rh-python38 rh-python38-python-devel
                run_command source /opt/rh/rh-python38/enable
                echo "Need to execute 'source /opt/rh/rh-python38/enable' to activate python3.8"
            else
                run_command $try_sudo $PKG_MANAGER install -y python3 python3-pip python3-devel
            fi
            ;;
        euler)
            run_command $try_sudo $PKG_MANAGER install -y python3 python3-pip python3-devel
            ;;
        macos)
            run_command brew install python@3.11
            echo 'export PATH="/usr/local/opt/python@3.11/bin:$PATH"' >> ~/.zshrc
            run_command source ~/.zshrc
            ;;
    esac

    if command -v python3 &> /dev/null; then
        curr_ver=$(python3 --version 2>&1 | awk '{print $2}')
        if version_ge "$curr_ver" "$req_ver"; then
            echo "Python installed successfully ($curr_ver)"
        else
            echo "Python version still doesn't meet requirements, please install manually"
            exit 1
        fi
    else
        echo "Python installation failed"
        exit 1
    fi
}

install_gcc() {
    # GCC version >= 7.3.0
    echo -e "\n==== Checking GCC ===="
    local req_ver="7.3.0"
    local gcc_curr_ver=""
    local gxx_curr_ver=""

    if command -v gcc &> /dev/null; then
        gcc_curr_ver=$(gcc --version | head -n1 | awk '{print $3}')
    else
        gcc_curr_ver="0.0.0"
    fi

    if command -v g++ &> /dev/null; then
        gxx_curr_ver=$(g++ --version | head -n1 | awk '{print $3}')
    else
        gxx_curr_ver="0.0.0"
    fi
    echo "Current GCC version: $gcc_curr_ver G++ version: $gxx_curr_ver"
    if version_ge "$gcc_curr_ver" "$req_ver"; then
        if version_ge "$gxx_curr_ver" "$req_ver"; then
            echo "GCC ($gcc_curr_ver) and G++ ($gxx_curr_ver) version meets requirements."
            return
        fi
    fi

    echo "Installing GCC and G++..."
    case "$OS" in
        debian)
            run_command $try_sudo $PKG_MANAGER update
            run_command $try_sudo $PKG_MANAGER install -y gcc-9 g++-9
            run_command $try_sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 \
                --slave /usr/bin/g++ g++ /usr/bin/g++-9
            ;;
        rhel)
            if grep -q "release 7" /etc/redhat-release; then
                run_command $try_sudo $PKG_MANAGER install -y centos-release-scl
                run_command $try_sudo $PKG_MANAGER install -y devtoolset-9-gcc devtoolset-9-gcc-c++
                run_command source /opt/rh/devtoolset-9/enable
                echo "Need to execute 'source /opt/rh/devtoolset-9/enable' to activate GCC9"
            else
                run_command $try_sudo $PKG_MANAGER install -y gcc gcc-c++
            fi
            ;;
        euler)
            run_command $try_sudo $PKG_MANAGER install -y gcc gcc-c++
            ;;
        macos)
            if ! xcode-select -p &> /dev/null; then
                xcode-select --install
            fi
            run_command brew install gcc@11
            echo 'export CC=/usr/local/bin/gcc-11' >> ~/.zshrc
            echo 'export CXX=/usr/local/bin/g++-11' >> ~/.zshrc
            run_command source ~/.zshrc
            ;;
    esac

    if command -v gcc &> /dev/null; then
        gcc_curr_ver=$(gcc --version | head -n1 | awk '{print $3}')
        gxx_curr_ver=$(g++ --version | head -n1 | awk '{print $3}')
        if version_ge "$gcc_curr_ver" "$req_ver" && version_ge "$gxx_curr_ver" "$req_ver"; then
            echo "GCC ($gcc_curr_ver) and G++ ($gxx_curr_ver) installed successfully."
        else
            echo "GCC version or G++ version still doesn't meet requirements, please install manually."
            exit 1
        fi
    else
        echo "GCC or G++ installation failed"
        exit 1
    fi
}

install_patch() {
    echo -e "\n==== Checking and Installing patch ===="

    # Check if 'patch' is already available in PATH
    if command -v patch &> /dev/null; then
        echo "patch is already installed."
        return 0
    fi

    echo "patch not found. Installing..."

    case "$OS" in
        debian)
            run_command $try_sudo "$PKG_MANAGER" update
            run_command $try_sudo "$PKG_MANAGER" install -y patch
            ;;
        rhel)
            if grep -q "release 7" /etc/redhat-release 2>/dev/null; then
                run_command $try_sudo "$PKG_MANAGER" install -y patch
            else
                run_command $try_sudo "$PKG_MANAGER" install -y patch
            fi
            ;;
        euler)
            run_command $try_sudo "$PKG_MANAGER" install -y patch
            ;;
        macos)
            echo "macOS usually ships with 'patch'. If missing, install Xcode Command Line Tools:"
            echo "    xcode-select --install"
            if ! xcode-select -p &> /dev/null; then
                echo "ERROR: Xcode Command Line Tools not installed. Please run 'xcode-select --install' manually."
                exit 1
            else
                echo "WARNING: 'patch' is missing despite Xcode CLI tools being present. Please investigate."
                exit 1
            fi
            ;;
        *)
            echo "Unsupported OS: $OS. Cannot auto-install patch."
            exit 1
            ;;
    esac

    if command -v patch &> /dev/null; then
        echo "patch installed successfully."
    else
        echo "ERROR: Failed to install patch. Please install it manually."
        exit 1
    fi
}

install_cmake() {
    # CMake version >= 3.16.0
    echo -e "\n==== Checking CMake ===="
    local req_ver="3.16.0"
    local curr_ver=""

    if command -v cmake &> /dev/null; then
        curr_ver=$(cmake --version | awk '/^cmake/ {print $3}')
        echo "Current CMake version: $curr_ver"
        if version_ge "$curr_ver" "$req_ver"; then
            echo "CMake meets requirements"
            return
        fi
    fi

    echo "Installing CMake..."
    case "$OS" in
        debian)
            if grep -q "Ubuntu 18.04" /etc/os-release; then
                run_command wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | $try_sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
                run_command echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
                run_command $try_sudo apt update
                run_command $try_sudo apt install -y cmake make
            else
                run_command $try_sudo $PKG_MANAGER update
                run_command $try_sudo $PKG_MANAGER install -y cmake make
            fi
            ;;
        rhel)
            if grep -q "release 7" /etc/redhat-release; then
                run_command $try_sudo $PKG_MANAGER install -y epel-release
                run_command $try_sudo $PKG_MANAGER install -y cmake3 make
                run_command $try_sudo ln -s /usr/bin/cmake3 /usr/bin/cmake
            else
                run_command $try_sudo $PKG_MANAGER install -y cmake make
            fi
            ;;
        euler)
            run_command $try_sudo $PKG_MANAGER install -y cmake make
            ;;
        macos)
            run_command brew install cmake
            ;;
    esac

    if command -v cmake &> /dev/null; then
        curr_ver=$(cmake --version | awk '/^cmake/ {print $3}')
        if version_ge "$curr_ver" "$req_ver"; then
            echo "CMake installed successfully ($curr_ver)"
        else
            echo "CMake version still doesn't meet requirements, please install manually"
            exit 1
        fi
    else
        echo "CMake installation failed"
        exit 1
    fi
}

install_pigz() {
    # pigz version >= 2.4
    echo -e "\n==== Checking pigz ===="
    local req_ver="2.4"
    local curr_ver=""

    if command -v pigz &> /dev/null; then
        curr_ver=$(pigz --version 2>&1 | awk '{print $2}')
        echo "Current pigz version: $curr_ver"
        if version_ge "$curr_ver" "$req_ver"; then
            echo "pigz meets requirements"
            return
        fi
    fi

    read -p "Install pigz? [Y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping pigz installation"
        return
    fi

    echo "Installing pigz..."
    case "$OS" in
        debian|rhel|euler)
            run_command $try_sudo $PKG_MANAGER install -y pigz
            ;;
        macos)
            run_command brew install pigz
            ;;
    esac

    if command -v pigz &> /dev/null; then
        curr_ver=$(pigz --version 2>&1 | awk '{print $2}')
        echo "pigz installed successfully ($curr_ver)"
    else
        echo "pigz installation failed, can be ignored"
    fi
}

install_dos2unix() {
    echo -e "\n==== Checking dos2unix ===="

    if command -v dos2unix &> /dev/null; then
        echo "dos2unix has been installed"
        return
    fi

    echo "Installing dos2unix..."
    case "$OS" in
        debian|rhel|euler)
            run_command $try_sudo $PKG_MANAGER install -y dos2unix
            ;;
        macos)
            run_command brew install dos2unix
            ;;
    esac

    if command -v dos2unix &> /dev/null; then
        echo "dos2unix installed successfully"
    else
        echo "dos2unix installation failed"
        exit 1
    fi
}

install_googletest() {
    # Recommended googletest version: release-1.11.0
    echo -e "\n==== Checking googletest ===="
    local req_ver="1.11.0"
    local curr_ver=""
    local gtest_src_dir="/usr/src/gtest"

    if pkg-config --exists gtest; then
        curr_ver=$(pkg-config --modversion gtest)
        echo "Current googletest version: $curr_ver"
        if version_ge "$curr_ver" "$req_ver"; then
            echo "googletest meets requirements"
            return
        fi
    fi
    read -p "Install googletest? [Y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping googletest installation"
        return
    fi

    echo "Installing googletest..."
    case "$OS" in
        debian)
            # Install libgtest-dev
            run_command $try_sudo $PKG_MANAGER install -y libgtest-dev
            # Check if gtest source directory exists
            if [ ! -d "$gtest_src_dir" ]; then
                echo "googletest source directory not found: $gtest_src_dir"
                echo "Attempting to reinstall libgtest-dev..."
                run_command $try_sudo $PKG_MANAGER purge -y libgtest-dev
                run_command $try_sudo $PKG_MANAGER install -y libgtest-dev
                # Check directory again
                if [ ! -d "$gtest_src_dir" ]; then
                    echo "Still cannot find $gtest_src_dir, please install manually:"
                    echo "1. Download source: wget https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz"
                    echo "2. Extract and compile: tar -zxf release-1.11.0.tar.gz && cd googletest-release-1.11.0 && cmake . && make && sudo make install"
                    exit 1
                fi
            fi
            # Force cmake execution in gtest source directory (even if cd fails)
            echo "Entering $gtest_src_dir to compile..."
            run_command $try_sudo cmake -S "$gtest_src_dir" -B "$gtest_src_dir/build"
            run_command $try_sudo make -C "$gtest_src_dir/build"
            run_command $try_sudo cp "$gtest_src_dir/build/lib/"*.a /usr/lib
            ;;
        rhel|euler)
            run_command $try_sudo $PKG_MANAGER install -y gtest gtest-devel
            ;;
        macos)
            run_command brew install googletest
            echo 'export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"'
            ;;
    esac

    if pkg-config --exists gtest; then
        curr_ver=$(pkg-config --modversion gtest)
        echo "googletest installed successfully ($curr_ver)"
    else
        echo "googletest installation failed"
        exit 1
    fi
}

main() {
    echo "===================================================="
    echo "Starting project dependency installation"
    echo "===================================================="

    detect_os
    install_gawk
    install_python
    install_python_deps
    install_gcc
    install_patch
    install_cmake
    install_pigz
    install_dos2unix
    install_googletest

    echo -e "===================================================="
    echo "All dependencies installed successfully!"
    echo "===================================================="
}

main "$@"