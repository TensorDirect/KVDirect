# conda install -c conda-forge -c rapidsai ucx-proc=*=gpu "ucx>=1.14" ucx-py -y

python -c "import utils, json; print(json.dumps(utils.dump_NIC()))" > /tmp/rdma_link.json

# compile ucx from source
# git clone --depth=1 https://github.com/openucx/ucx.git /tmp/ucx
# cd /tmp/ucx
# bash ./autogen.sh 
# mkdir build
# cd build
# ../configure --enable-logging --enable-mt --with-verbs --with-rdmacm --with-cuda=/usr/local/cuda  --prefix=$CONDA_PREFIX
# make -j
# make -j install