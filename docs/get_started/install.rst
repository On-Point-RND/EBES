Installation
============

The recommended way to use EBES is through Docker.
The EBES Docker image is built on the `nvidia-pytorch <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>`_ image, ensuring compatibility with NVIDIA GPUs.
The Python packages necessary for running the benchmark can be found in the ``requirements.txt`` file.
For development purposes, additional packages are listed in the ``requirements-dev.txt`` file.
We tested EBES on NVIDIA TU102 and NVIDIA A100 with CUDA 12.2.
