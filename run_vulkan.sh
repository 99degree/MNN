#!/bin/bash
# run_vulkan.sh – run any executable with Vulkan ICD from the repo's vulkan_icd/ directory
# Usage: ./run_vulkan.sh <your_executable> [args...]

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Point Vulkan ICD to our bundled JSON files
export VK_ICD_FILENAMES="${REPO_DIR}/vulkan_icd/lvp_icd.aarch64.json:${REPO_DIR}/vulkan_icd/freedreno_icd.aarch64.json"
# Add Vulkan libraries to the dynamic loader path
export LD_LIBRARY_PATH="${REPO_DIR}/vulkan_icd:${LD_LIBRARY_PATH}"

# Run the command
exec "$@"
