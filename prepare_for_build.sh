#!/bin/bash
# Script to prepare the build environment for Mamba.
#
# Example usage:
#   ./prepare_for_build.sh v2.2.6.post3

set -euxo pipefail

export ROOT=`pwd`

if [ $# -ne 1 ]; then
    echo "Usage: $0 <mamba_version>"
    echo "Example: $0 v2.2.6.post3"
    exit 1
fi

MAMBA_VERSION=$1

# Ensure that the Mamba version is supported.
if [ ! -d "${ROOT}/build_scripts/patches/${MAMBA_VERSION}" ]; then
    echo "Error: patches/${MAMBA_VERSION} directory does not exist"
    exit 1
fi

# Apply patches.
for patch in "${ROOT}/build_scripts/patches/${MAMBA_VERSION}"/*.patch; do
    patch -p1 -d ${ROOT} -i ${patch}
done
