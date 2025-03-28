FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

#ENV
ARG GitLabLogin=
ARG GitLabPassword=
ENV gitLabLogin=$GitLabLogin
ENV gitLabPassword=$GitLabPassword

# git clone
WORKDIR /workspace
COPY core core
COPY utils utils
COPY startup.sh .
COPY requirements.txt .
RUN mkdir data models

# Run if container is started
CMD ["/bin/bash", "/workspace/startup.sh"]