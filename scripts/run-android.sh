REMOTE_PATH="/data/local/tmp/"
GGUF_MODEL_NAME="/data/data/com.termux/files/home/model"

function run_llamacli()
{
    adb shell "export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-cli ${running_params} -no-cnv -m ${GGUF_MODEL_NAME} -p \"introduce the movie Once Upon a Time in America briefly.\n\""
}