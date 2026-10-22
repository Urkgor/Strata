# A RHEL 7 / CentOS 7 build environment for the CUDA engine: CentOS 7.9 (glibc 2.17, kernel headers 3.10), Red Hat's
# devtoolset-11 (GCC 11, binutils 2.36), NVIDIA's CUDA 12.x for RHEL 7, and CMake from pip.  It is what docs/RHEL7.md
# describes, so you can check a change against the same system, or build the engine without touching the host:
#
#   docker build -t strata-rhel7 docker/rhel7
#   docker run --rm -v "$PWD":/src -w /src strata-rhel7 tools/build_rhel7.sh --arch 61
#
# CentOS 7 ended in June 2024: its repositories moved to vault.centos.org (rewritten below).  NVIDIA's RHEL 7
# repository carries CUDA 12.0 to 12.4 (checked in its metadata); 12.4 is the newest.  The packages are not
# signature-checked here (gpgcheck=0): this image is a build tool, not something to run.
ARG BASE=quay.io/centos/centos:centos7
FROM ${BASE}
ARG CUDA_PKG=12-4
ARG DEVTOOLSET=11
# a company proxy that re-signs HTTPS: its certificate goes in extra-ca/ (a proxy itself: --build-arg https_proxy=...)
COPY extra-ca/ /etc/pki/ca-trust/source/anchors/
RUN update-ca-trust
ENV PIP_CERT=/etc/pki/tls/certs/ca-bundle.crt
RUN sed -i -e 's|^mirrorlist=|#mirrorlist=|' -e 's|^#\s*baseurl=http://mirror.centos.org|baseurl=https://vault.centos.org|' /etc/yum.repos.d/CentOS-*.repo \
 && yum -y install centos-release-scl \
 && sed -i -e 's|^mirrorlist=|#mirrorlist=|' -e 's|^#\s*baseurl=http://mirror.centos.org|baseurl=https://vault.centos.org|' /etc/yum.repos.d/CentOS-SCLo-*.repo \
 && yum -y install devtoolset-${DEVTOOLSET}-gcc-c++ devtoolset-${DEVTOOLSET}-binutils make git patch which curl \
        ca-certificates python3 python3-pip \
 && yum clean all
RUN printf '[cuda-rhel7]\nname=NVIDIA CUDA (RHEL 7)\nbaseurl=https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64\nenabled=1\ngpgcheck=0\n' \
        > /etc/yum.repos.d/cuda-rhel7.repo \
 && yum -y install cuda-nvcc-${CUDA_PKG} cuda-cudart-devel-${CUDA_PKG} cuda-driver-devel-${CUDA_PKG} \
        libcublas-devel-${CUDA_PKG} cuda-cccl-${CUDA_PKG} \
 && yum clean all
RUN python3 -m pip install --upgrade 'pip<22' \
 && python3 -m pip install 'cmake>=3.25.2' ninja
ENV PATH=/usr/local/cuda/bin:$PATH
