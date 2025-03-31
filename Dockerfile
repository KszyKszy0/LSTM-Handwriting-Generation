FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

#ENV
ARG GitLabLogin=
ARG GitLabPassword=
ENV gitLabLogin=$GitLabLogin
ENV gitLabPassword=$GitLabPassword

# git clone
WORKDIR /workspace
COPY core core
COPY utils utils
COPY models models
COPY startup.sh .
COPY requirements.txt .
RUN mkdir data

# Run if container is started
CMD ["/bin/bash", "/workspace/startup.sh"]