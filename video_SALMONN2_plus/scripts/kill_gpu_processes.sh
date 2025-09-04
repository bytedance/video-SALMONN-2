#!/bin/bash

echo "Killing all GPU processes and freeing GPU memory..."

# Get all PIDs using GPU from nvidia-smi
gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)

if [ ! -z "$gpu_pids" ]; then
    echo "Found GPU processes with PIDs: $gpu_pids"
    
    # Kill each process
    for pid in $gpu_pids; do
        if kill -0 $pid 2>/dev/null; then
            echo "Killing process $pid..."
            kill -TERM $pid
            sleep 2
            # Force kill if still running
            if kill -0 $pid 2>/dev/null; then
                echo "Force killing process $pid..."
                kill -KILL $pid
            fi
        else
            echo "Process $pid already dead."
        fi
    done
else
    echo "No GPU processes found in nvidia-smi."
fi

# Kill PyTorch data workers that might be holding GPU memory
echo "Killing PyTorch data workers..."
pt_worker_pids=$(pgrep -f "pt_data_worker" 2>/dev/null || true)
if [ ! -z "$pt_worker_pids" ]; then
    echo "Found PyTorch workers: $pt_worker_pids"
    sudo kill -9 $pt_worker_pids 2>/dev/null || true
    echo "Killed PyTorch data workers."
else
    echo "No PyTorch data workers found."
fi

# Find any processes using NVIDIA devices
echo "Checking for processes using NVIDIA devices..."
nvidia_device_pids=$(sudo fuser /dev/nvidia* 2>/dev/null | grep -o '[0-9]\+' || true)
if [ ! -z "$nvidia_device_pids" ]; then
    echo "Found processes using NVIDIA devices: $nvidia_device_pids"
    for pid in $nvidia_device_pids; do
        # Skip nvidia-persistenced (PID 620)
        if [ "$pid" != "620" ]; then
            echo "Killing process $pid using NVIDIA device..."
            sudo kill -9 $pid 2>/dev/null || true
        fi
    done
else
    echo "No additional processes using NVIDIA devices."
fi

# Final GPU status check
echo ""
echo "Final GPU status:"
nvidia-smi

echo "GPU cleanup complete."