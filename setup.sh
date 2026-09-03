# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license
#
# Modified from TRELLIS.2 (https://github.com/microsoft/TRELLIS.2)
# Copyright (c) Microsoft Corporation. Licensed under the MIT License.

# Read Arguments
TEMP=`getopt -o h --long help,new-env,basic,flash-attn,cumesh,o-voxel,flexgemm,nvdiffrast,nvdiffrec -n 'setup.sh' -- "$@"`

eval set -- "$TEMP"

HELP=false
NEW_ENV=false
BASIC=false
FLASHATTN=false
CUMESH=false
OVOXEL=false
FLEXGEMM=false
NVDIFFRAST=false
NVDIFFREC=false
ERROR=false


if [ "$#" -eq 1 ] ; then
    HELP=true
fi

while true ; do
    case "$1" in
        -h|--help) HELP=true ; shift ;;
        --new-env) NEW_ENV=true ; shift ;;
        --basic) BASIC=true ; shift ;;
        --flash-attn) FLASHATTN=true ; shift ;;
        --cumesh) CUMESH=true ; shift ;;
        --o-voxel) OVOXEL=true ; shift ;;
        --flexgemm) FLEXGEMM=true ; shift ;;
        --nvdiffrast) NVDIFFRAST=true ; shift ;;
        --nvdiffrec) NVDIFFREC=true ; shift ;;
        --) shift ; break ;;
        *) ERROR=true ; break ;;
    esac
done

if [ "$ERROR" = true ] ; then
    echo "Error: Invalid argument"
    HELP=true
fi

if [ "$HELP" = true ] ; then
    echo "Usage: setup.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  --new-env               Create a new conda environment"
    echo "  --basic                 Install basic dependencies"
    echo "  --flash-attn            Install flash-attention"
    echo "  --cumesh                Install cumesh"
    echo "  --o-voxel               Install o-voxel"
    echo "  --flexgemm              Install flexgemm"
    echo "  --nvdiffrast            Install nvdiffrast"
    echo "  --nvdiffrec             Install nvdiffrec"
    return
fi

# Get system information
WORKDIR=$(pwd)
if command -v nvidia-smi > /dev/null; then
    PLATFORM="cuda"
elif command -v rocminfo > /dev/null; then
    PLATFORM="hip"
else
    echo "Error: No supported GPU found"
    exit 1
fi

if [ "$NEW_ENV" = true ] ; then
    conda create -n inspace python=3.10
    conda activate inspace
    if [ "$PLATFORM" = "cuda" ] ; then
        pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    elif [ "$PLATFORM" = "hip" ] ; then
        pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/rocm6.2.4
    fi
fi

if [ "$BASIC" = true ] ; then
    pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
    sudo apt install -y libjpeg-dev
    pip install pillow-simd
    pip install kornia timm
    # InSpace-specific dependencies (360 panorama / scene pipeline, viz, metrics)
    pip install py360convert plyfile open3d matplotlib scipy plotly pyrender scikit-image
    # Optional: pytorch3d is only needed for the OBB bbox-inference eval scripts
    # (eval/stage1/bbox_inference_centerpoint_obb*.py). Install separately if you need them:
    #   pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
fi

# Setup CUDA_HOME if not set
if [ "$PLATFORM" = "cuda" ] && [ -z "$CUDA_HOME" ]; then
    # echo "CUDA_HOME not set. Installing cuda-toolkit 12.4 via conda..." 
    # conda install -y cuda-toolkit=12.4 -c nvidia # Pulls the newest individual components from the nvidia channel, which can mix versions.
    echo "CUDA_HOME not set. Installing CUDA 12.4 development tools via conda..."
    # Install specific CUDA 12.4 components instead of full toolkit to avoid version conflicts
    # The nvidia/label/cuda-12.4.1 channel pins components to exactly CUDA 12.4
    conda install -y -c nvidia/label/cuda-12.4.0 cuda-nvcc cuda-cudart-dev cuda-toolkit=12.4.0
    export CUDA_HOME=$CONDA_PREFIX
    echo "CUDA_HOME set to: $CUDA_HOME"
fi

if [ "$FLASHATTN" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        pip install flash-attn==2.7.3 --no-build-isolation
    elif [ "$PLATFORM" = "hip" ] ; then
        echo "[FLASHATTN] Prebuilt binaries not found. Building from source..."
        mkdir -p /tmp/extensions
        git clone --recursive https://github.com/ROCm/flash-attention.git /tmp/extensions/flash-attention
        cd /tmp/extensions/flash-attention
        git checkout tags/v2.7.3-cktile
        GPU_ARCHS=gfx942 python setup.py install #MI300 series
        cd $WORKDIR
    else
        echo "[FLASHATTN] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$NVDIFFRAST" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        export CUDA_HOME=${CUDA_HOME:-$CONDA_PREFIX}
        mkdir -p /tmp/extensions
        git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
        pip install /tmp/extensions/nvdiffrast --no-build-isolation
    else
        echo "[NVDIFFRAST] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$NVDIFFREC" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        export CUDA_HOME=${CUDA_HOME:-$CONDA_PREFIX}
        mkdir -p /tmp/extensions
        git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
        pip install /tmp/extensions/nvdiffrec --no-build-isolation
    else
        echo "[NVDIFFREC] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$CUMESH" = true ] ; then
    export CUDA_HOME=${CUDA_HOME:-$CONDA_PREFIX}
    mkdir -p /tmp/extensions
    # Configure git to use HTTPS instead of SSH for GitHub (to avoid SSH key issues with submodules)
    git config --global url."https://github.com/".insteadOf git@github.com:
    git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive
    pip install /tmp/extensions/CuMesh --no-build-isolation
fi

if [ "$FLEXGEMM" = true ] ; then
    export CUDA_HOME=${CUDA_HOME:-$CONDA_PREFIX}
    mkdir -p /tmp/extensions
    # Configure git to use HTTPS instead of SSH for GitHub (to avoid SSH key issues with submodules)
    git config --global url."https://github.com/".insteadOf git@github.com:
    git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive
    pip install /tmp/extensions/FlexGEMM --no-build-isolation
fi

if [ "$OVOXEL" = true ] ; then
    export CUDA_HOME=${CUDA_HOME:-$CONDA_PREFIX}
    mkdir -p /tmp/extensions
    cp -r o-voxel /tmp/extensions/o-voxel
    pip install /tmp/extensions/o-voxel --no-build-isolation
fi

