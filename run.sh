#!/bin/bash

# Stop script on error
set -e

entrypoint="efficient_rsnn_bmi.main"
experiment="baseline"

extra_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--entrypoint)
            entrypoint="$2"
            shift 2
            ;;
        *=*)
            extra_args+=("$1")
            shift
            ;;
        *)
            echo "Unknown arguments: $1"
            exit 1
            ;;
    esac
done

if [ ${#extra_args[@]} -eq 0 ]; then
    echo "Usage: ./run.sh --entrypoint <script.py> model=<model_name> experiment=<experiment_name>"
    exit 1
fi

for arg in "${extra_args[@]}"; do
    if [[ "$arg" == experiment=* ]]; then
        experiment="${arg#*=}"
    fi
done

printf "\n"
printf "[PARAMETERS]\n"
printf "    Entrypoint: $entrypoint\n"
printf "    Extra Params: ${extra_args[*]}\n"
printf "\n"

echo "Start Running"

VENV_DIR="venvs/efficient-rsnn-bmi"

# Ensure the virtual environment is activated
if [ -d "$VENV_DIR" ]; then
    echo "Using virtual environment: $VENV_DIR"
    source "$VENV_DIR/bin/activate"
else
    echo "Virtual environment not found!"
    exit 1
fi

mkdir -p outputs/log/${experiment}

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
logfile="outputs/log/${experiment}/${experiment}_${timestamp}.log"

run_cmd="python3 -m $entrypoint ${extra_args[@]} | tee \"$logfile\""
echo "Running command: $run_cmd"

eval "${run_cmd}"

echo "Experiment completed!"