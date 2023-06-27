# We choose ubuntu 22.04 as our base docker image
FROM ghcr.io/scientificcomputing/fenics-gmsh:2023-04-21

ARG REPO_BRANCH="main"
ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system

# We set the working directory to install docker dependencies
WORKDIR /src

# Install the Python-module
RUN git clone --branch ${REPO_BRANCH} --single-branch https://github.com/jorgensd/drag-evaluation
RUN python3 -m pip install ./drag-evaluation[all]
