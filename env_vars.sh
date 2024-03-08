export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export TF_CPP_MIN_LOG_LEVEL="2"