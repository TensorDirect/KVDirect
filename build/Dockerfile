# Use the official Miniconda image as a base image
FROM continuumio/miniconda3

# Set environment variables
ENV CONDA_ENV_NAME=vllm
ENV PYTHON_VERSION=3.9

# Create and activate the conda environment, then install the required packages
RUN conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION && \
    conda install -n $CONDA_ENV_NAME -c conda-forge -c rapidsai ucx-proc=*=gpu "ucx>=1.14" ucx-py -y

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "vllm", "/bin/bash", "-c"]

# Optionally set the environment to be activated on shell startup
RUN echo "conda activate $CONDA_ENV_NAME" >> ~/.bashrc

# Specify the entrypoint
ENTRYPOINT ["conda", "run", "-n", "vllm", "/bin/bash"]
