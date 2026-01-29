#!/bin/bash

# Base directory for all Run_XX folders
BASE_DIR="$HOME/ondemand/data/sys/myjobs/projects/default"

# Runs
RUNS=("Run_01" "Run_02" "Run_03")

# Y_i folders
Y_FOLDERS=("Y_1" "Y_2" "Y_3" "Y_4" "Y_5" "Y_6")

# BS folders
BS_FOLDERS=("BS10" "BS100" "BS500" "BS1000")

echo "Submitting main_job.sh in all BS folders..."

# Loop over Runs
for R in "${RUNS[@]}"; do
    RUN_DIR="$BASE_DIR/$R"
    echo "=== $R ==="

    # Loop over Y_i
    for Y in "${Y_FOLDERS[@]}"; do
        Y_PATH="$RUN_DIR/$Y"

        # Loop over BS folders
        for B in "${BS_FOLDERS[@]}"; do
            BS_PATH="$Y_PATH/$B"
            JOB_FILE="$BS_PATH/main_job.sh"

            if [ -f "$JOB_FILE" ]; then
                echo "Submitting $JOB_FILE..."
                sbatch "$JOB_FILE"
            else
                echo "Warning: $JOB_FILE not found, skipping."
            fi
        done
    done
done

echo "All jobs submitted!"
