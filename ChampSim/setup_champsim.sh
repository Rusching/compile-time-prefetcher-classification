#!/bin/bash

# Parse arguments for test mode
TEST_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set trace and checksum files based on mode
if [ "$TEST_MODE" = true ]; then
    TRACE_FILE="artifact_traces_test.csv"
    CHECKSUM_FILE="artifact_traces_test.md5"
    echo "Test mode enabled. Using $TRACE_FILE and $CHECKSUM_FILE."
else
    TRACE_FILE="artifact_traces.csv"
    CHECKSUM_FILE="artifact_traces.md5"
    echo "Using $TRACE_FILE and $CHECKSUM_FILE."
fi

# Source environment variables
if [ -f "./setvars.sh" ]; then
    echo "Setting environment variables..."
    source setvars.sh
else
    echo "setvars.sh file not found!"
    exit 1
fi

# Define script variables
LIBBF_DIR="$PYTHIA_HOME/libbf"
SNN_DIR="$PYTHIA_HOME/snnpp"
TRACES_DIR="$PYTHIA_HOME/traces"
MEGATOOLS_URL="https://megatools.megous.com/builds/builds/megatools-1.11.1.20230212-linux-x86_64.tar.gz"

# Check and install prerequisites
echo "Checking and installing prerequisites..."
if ! command -v perl &> /dev/null; then
    echo "Perl not found. Installing..."
    sudo apt update && sudo apt install -y perl
else
    echo "Perl is already installed."
fi

# Clone and build the bloomfilter library
bloomfilter_dir_existed=false
if [[ -d "$LIBBF_DIR" ]]; then
    bloomfilter_dir_existed=true
else
    echo "Cloning bloomfilter library..."
    cd "$PYTHIA_HOME" || { echo "Failed to access $PYTHIA_HOME"; exit 1; }
    git clone https://github.com/mavam/libbf.git libbf || { echo "Failed to clone bloomfilter library"; exit 1; }
fi

if [[ "$TEST_MODE" == "true" || "$bloomfilter_dir_existed" == "false" ]]; then
    echo "Building bloomfilter library..."
    cd "$LIBBF_DIR" || { echo "Failed to access SNN directory"; exit 1; }
    mkdir -p build || { echo "Failed to create build directory"; exit 1; }
    cd build || { echo "Failed to access build directory"; exit 1; }
    cmake || { echo "CMake configuration failed"; exit 1; }
    make clean && make || { echo "Failed to build bloomfilter library"; exit 1; }
fi

# Clone and build the SNN library
SNN_dir_existed=false
if [[ -d "$SNN_DIR" ]]; then
    SNN_dir_existed=true
else
    echo "Cloning SNN library..."
    cd "$PYTHIA_HOME" || { echo "Failed to access $PYTHIA_HOME"; exit 1; }
    git clone git@github.com:ianmkim/snnpp.git || { echo "Failed to clone SNN library"; exit 1; }
fi

if [[ "$TEST_MODE" == "true" || "$SNN_dir_existed" == "false" ]]; then
    echo "Building SNN library..."
    cd "$SNN_DIR" || { echo "Failed to access SNN directory"; exit 1; }
    mkdir -p build || { echo "Failed to create build directory"; exit 1; }
    cd build || { echo "Failed to access build directory"; exit 1; }
    cmake .. || { echo "CMake configuration failed"; exit 1; }
    make clean && make || { echo "Failed to build SNN library"; exit 1; }

    if [[ "$TEST_MODE" == "true" ]]; then
        echo "Testing C++ Spiking Neural Network library..."
        cd ..
        ./build/unit_tests
    fi
fi

# Build Pythia
echo "Building Pythia..."
cd $PYTHIA_HOME
if [ ! -f "./build_champsim.sh" ]; then
    echo "build_champsim.sh script not found!"
    exit 1
fi
./build_champsim.sh multi multi no 1 || { echo "Failed to build Pythia"; exit 1; }

# Download and prepare traces
if [ ! -d "$TRACES_DIR" ]; then
    echo "Creating traces directory..."
    mkdir -p $TRACES_DIR
fi

cd $PYTHIA_HOME/scripts

# Download traces using Perl script if not already downloaded
if [ ! "$(ls -A $TRACES_DIR)" ]; then
    echo "Downloading traces..."
    perl download_traces.pl --csv $TRACE_FILE --dir ../traces/ || { echo "Trace download failed. Use Google Drive links."; exit 1; }
else
    echo "Traces already downloaded."
fi

# Verify checksum
echo "Verifying checksum..."
cd $TRACES_DIR
if [ -f "../scripts/$CHECKSUM_FILE" ]; then
    md5sum -c ../scripts/$CHECKSUM_FILE || echo "Checksum verification failed. Recheck traces."
else
    echo "Checksum file $CHECKSUM_FILE not found!"
fi

echo "Setup completed successfully."
