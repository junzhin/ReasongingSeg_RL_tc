#!/bin/bash
set -euo pipefail

source /mnt/petrelfs/tangcheng/miniconda3/etc/profile.d/conda.sh
conda activate papo

export CUDA_STUB=/tmp/papo_cuda_home_${SLURM_JOB_ID:-manual}
rm -rf "$CUDA_STUB"
mkdir -p "$CUDA_STUB/bin" "$CUDA_STUB/lib64"

cp -as "$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/include" "$CUDA_STUB/include"
ln -s "$CONDA_PREFIX/include/crt" "$CUDA_STUB/include/crt"
ln -s "$CONDA_PREFIX/include/nv" "$CUDA_STUB/include/nv"

cat > "$CUDA_STUB/bin/nvcc" <<EOF
#!/bin/bash
exec "$CONDA_PREFIX/bin/nvcc" -allow-unsupported-compiler "\$@"
EOF
chmod +x "$CUDA_STUB/bin/nvcc"

while IFS= read -r d; do
    ln -sf "$d"/* "$CUDA_STUB/lib64/" || true
done < <(find "$CONDA_PREFIX/lib/python3.10/site-packages/nvidia" -maxdepth 2 -type d -name lib)

if [ -f "$CUDA_STUB/lib64/libcudart.so.12" ] && [ ! -e "$CUDA_STUB/lib64/libcudart.so" ]; then
    ln -s "$CUDA_STUB/lib64/libcudart.so.12" "$CUDA_STUB/lib64/libcudart.so"
fi

EXTRA_INC=$(find "$CONDA_PREFIX/lib/python3.10/site-packages/nvidia" -maxdepth 2 -type d -name include | paste -sd: -)
EXTRA_LIB=$(find "$CONDA_PREFIX/lib/python3.10/site-packages/nvidia" -maxdepth 2 -type d -name lib | paste -sd: -)

export CUDA_HOME="$CUDA_STUB"
export PATH="$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$EXTRA_LIB:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:$EXTRA_LIB:${LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/include:$CONDA_PREFIX/include:$EXTRA_INC:${CPATH:-}"
export CC=x86_64-conda-linux-gnu-gcc
export CXX=x86_64-conda-linux-gnu-g++
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=4

python -m pip install --no-build-isolation --no-cache-dir --no-deps xformers==0.0.29.post2
