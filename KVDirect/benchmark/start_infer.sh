export RAY_NUM_CPUS=64
export VLLM_ENGINE_ITERATION_TIMEOUT_S=60
# export VLLM_ATTENTION_BACKEND=XFORMERS
model_path=/data
# model_name=Meta-Llama-3-70B
# model_name=mistral-7b-instruct-v0_2
model_name=Meta-Llama-3-70B-dummy
TP=8
# python api_server.py --model $model_path/$model_name  -tp $TP --enforce-eager --disable-custom-all-reduce #--enable-chunked-prefill 

python api_server.py --model "$model_path/$model_name" \
    --enforce-eager --disable-custom-all-reduce --load-format dummy \
    --distributed-executor-backend ray --enable-chunked-prefill False \
    -tp "$TP" > "${model_name}_$TP.txt"
