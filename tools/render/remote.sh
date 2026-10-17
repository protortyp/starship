#!/usr/bin/env bash
# Headless Blender on a rented Linux GPU box (vast.ai image: nvidia/cuda ... ubuntu24.04).
#
#   bash tools/render/remote.sh install            # Blender LTS tarball + runtime libs + ffmpeg
#   bash tools/render/remote.sh render <args...>   # blender -b -P starship_scene.py -- <args>
#   bash tools/render/remote.sh encode <frame dir> <out.mp4> [fps]
#
# Run from the repo root (`just vast --name render exec bash tools/render/remote.sh ...`).
set -euo pipefail

BLENDER_VERSION="${BLENDER_VERSION:-4.5.9}"
BLENDER_DIR="/opt/blender-${BLENDER_VERSION}"
BLENDER_URL="https://download.blender.org/release/Blender${BLENDER_VERSION%.*}/blender-${BLENDER_VERSION}-linux-x64.tar.xz"

install() {
    if ! command -v ffmpeg >/dev/null || ! ldconfig -p | grep -q libXi.so.6; then
        apt-get update -qq
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
            libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libegl1 libxkbcommon0 libsm6 libice6 \
            libgomp1 ffmpeg xz-utils ca-certificates curl >/dev/null
    fi
    if [ ! -x "${BLENDER_DIR}/blender" ]; then
        echo "downloading Blender ${BLENDER_VERSION}"
        mkdir -p "${BLENDER_DIR}"
        curl -sSL "${BLENDER_URL}" | tar -xJ --strip-components=1 -C "${BLENDER_DIR}"
    fi
    ln -sf "${BLENDER_DIR}/blender" /usr/local/bin/blender
    blender -b --version | head -1
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}

render() {
    mkdir -p out/render
    blender -b -noaudio -P tools/render/starship_scene.py -- "$@"
}

encode() {
    local dir="$1" out="$2" fps="${3:-30}"
    ffmpeg -y -loglevel error -framerate "${fps}" -pattern_type glob -i "${dir}/frame_*.png" \
        -c:v libx264 -pix_fmt yuv420p -crf 17 -preset slow -movflags +faststart "${out}"
    echo "wrote ${out}"
}

cmd="${1:-}"
shift || true
case "${cmd}" in
    install) install ;;
    render) render "$@" ;;
    encode) encode "$@" ;;
    *) sed -n 2,9p "$0"; exit 1 ;;
esac
