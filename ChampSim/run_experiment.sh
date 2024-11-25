#!/bin/bash

# Default behavior: Run 1C_BASE experiments
DEFAULT=true
RUN_1C_BASE=false
RUN_1C_DRAM=false
RUN_4C=false

# Parse command-line arguments
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --1C_BASE)
            RUN_1C_BASE=true
            DEFAULT=false
            shift
            ;;
        --1C_DRAM)
            RUN_1C_DRAM=true
            DEFAULT=false
            shift
            ;;
        --4C)
            RUN_4C=true
            DEFAULT=false
            shift
            ;;
        --all)
            RUN_1C_BASE=true
            RUN_1C_DRAM=true
            RUN_4C=true
            DEFAULT=false
            shift
            ;;
        -*|--*)
            echo "Unknown option: $1"
            echo "Usage: $0 [--1C_BASE | --1C_DRAM | --4C | --all] <tlist_file>"
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional parameters
set -- "${POSITIONAL_ARGS[@]}"

# Validate the tlist argument
if [ ${#POSITIONAL_ARGS[@]} -eq 0 ]; then
    echo "Usage: $0 [--1C_BASE | --1C_DRAM | --4C | --all] <tlist_file>"
    exit 1
fi

TLIST="${POSITIONAL_ARGS[0]}"

# Apply default behavior if no specific options were selected
if $DEFAULT; then
    RUN_1C_BASE=true
fi

# Source environment variables
if [ -f "./setvars.sh" ]; then
    echo "Setting environment variables..."
    source setvars.sh
else
    echo "setvars.sh file not found!"
    exit 1
fi

# Define variables
EXE="$PYTHIA_HOME/bin/perceptron-pmp-no-no-ship-1core"
ROLLUP_SCRIPT="$PYTHIA_HOME/scripts/rollup.pl"
EXPERIMENTS_DIR="$PYTHIA_HOME/experiments"

# Experiment-specific variables
EXP_1C_BASE="single_1C_base_config.exp"
MFILE_1C_BASE="rollup_1C_base_config.mfile"
EXP_1C_DRAM="rollup_1C_varying_DRAM_bw.exp"
MFILE_1C_DRAM="rollup_1C_varying_DRAM_bw.mfile"
EXP_4C="rollup_4C.exp"
MFILE_4C="rollup_4C.mfile"

# Ensure necessary directories exist
mkdir -p "$EXPERIMENTS_DIR/experiments_1C"
mkdir -p "$EXPERIMENTS_DIR/experiments_4C"

cp "$TLIST" "$EXPERIMENTS_DIR"

# Run 1C_BASE experiments
if [ "$RUN_1C_BASE" = true ]; then
    echo "Running 1C_BASE experiments..."
    cd "$EXPERIMENTS_DIR" || exit
    perl ../scripts/create_jobfile.pl --exe "$EXE" --tlist "$TLIST" --exp "$EXP_1C_BASE" --local 1 > jobfile_1C_BASE.sh
    cd experiments_1C || exit
    source ../jobfile_1C_BASE.sh
    cd ..
    echo "Rolling up statistics for 1C_BASE..."
    cd experiments_1C || exit
    $ROLLUP_SCRIPT --tlist "../$TLIST" --exp "../$EXP_1C_BASE" --mfile "../$MFILE_1C_BASE" > ../rollup_1C_base_config.csv
    cd ..
fi

# Run 1C_DRAM experiments
if [ "$RUN_1C_DRAM" = true ]; then
    echo "Running 1C_DRAM experiments..."
    cd "$EXPERIMENTS_DIR" || exit
    perl ../scripts/create_jobfile.pl --exe "$EXE" --tlist "$TLIST" --exp "$EXP_1C_DRAM" --local 1 > jobfile_1C_DRAM.sh
    cd experiments_1C || exit
    source ../jobfile_1C_DRAM.sh
    cd ..
    echo "Rolling up statistics for 1C_DRAM..."
    cd experiments_1C || exit
    $ROLLUP_SCRIPT --tlist "../$TLIST" --exp "../$EXP_1C_DRAM" --mfile "../$MFILE_1C_DRAM" > ../rollup_1C_varying_DRAM_bw.csv
    cd ..
fi

# Run 4C experiments
if [ "$RUN_4C" = true ]; then
    echo "Running 4C experiments..."
    cd "$EXPERIMENTS_DIR" || exit
    perl ../scripts/create_jobfile.pl --exe "$EXE" --tlist "$TLIST" --exp "$EXP_4C" --local 1 > jobfile_4C.sh
    cd experiments_4C || exit
    source ../jobfile_4C.sh
    cd ..
    echo "Rolling up statistics for 4C..."
    cd experiments_4C || exit
    $ROLLUP_SCRIPT --tlist "../$TLIST" --exp "../$EXP_4C" --mfile "../$MFILE_4C" > ../rollup_4C.csv
    cd ..
fi

# Move rollup_*.csv files into PYTHIA_HOME
echo "Moving rollup_*.csv files to $PYTHIA_HOME..."
find "$EXPERIMENTS_DIR" -type f -name "rollup_*.csv" -exec mv {} "$PYTHIA_HOME" \;

# Final Summary
echo "Experiment workflow completed. Summary of rollup files moved:"
find "$PYTHIA_HOME" -type f -name "rollup_*.csv"
