# Use the official Miniconda image from the Docker Hub
FROM continuumio/miniconda3:latest

# Set the working directory in the container
WORKDIR /app

# Create the Conda environment
RUN conda create -n vllm python=3.9 -y

# Install the required Conda packages including PyTorch
RUN /bin/bash -c "source activate vllm && conda install -c conda-forge -c rapidsai ucx-proc=*=gpu 'ucx>=1.14' ucx-py -y"
RUN /bin/bash -c "source activate vllm && conda install -c conda-forge aiohttp uvicorn uvloop fastapi -y"
RUN /bin/bash -c "source activate vllm && conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia -y"

# Copy the decode_worker.py file and other necessary files into the container
COPY ../splitwise/decode_worker.py /app/decode_worker.py
COPY ../splitwise/test_stub.py /app/test_stub.py
COPY ../splitwise/utils.py /app/utils.py
COPY ../vllm /app/vllm

# Expose the port that the HTTP server will run on
EXPOSE 8001

# Set the entry point to run the HTTP server
ENTRYPOINT ["/bin/bash", "-c", "source activate vllm && python /app/decode_worker.py"]
