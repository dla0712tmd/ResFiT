FROM anaconda/miniconda:latest

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libopengl0 \
    libglvnd0 \
    libglx0 \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

# Optional: conda ToS
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# Expose all NVIDIA driver capabilities (EGL, OpenGL, etc.) for headless rendering
ENV NVIDIA_DRIVER_CAPABILITIES=all
# Use EGL for offscreen rendering (MuJoCo/robosuite headless)
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

CMD ["/bin/bash"]
