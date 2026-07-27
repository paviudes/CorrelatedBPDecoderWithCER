#!/bin/bash

INTERVAL=30
LOG_FILE="./../data/72q_BB_p_0.010_std_0.01_q_0.001_std_0.00_data/cluster/logs/gpu_monitoring.log"

# Parse the --interval argument
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --interval) INTERVAL="$2"; shift ;;
    esac
    shift
done

echo "Starting GPU logger every $INTERVAL seconds..."
echo "Monitoring for active 'julia' processes..."

# 1. Start nvidia-smi in the background
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l "$INTERVAL" > "$LOG_FILE" &
SMI_PID=$!

# 2. Check if Julia is running. 
# pgrep checks for processes named 'julia' owned by you ($USER).
# It will sleep and check again as long as the process exists.
while pgrep -u "$USER" julia > /dev/null; do
    sleep 10
done

# 3. Once the julia process is gone, kill the logger
kill $SMI_PID
echo "Julia process ended. GPU logger terminated."
