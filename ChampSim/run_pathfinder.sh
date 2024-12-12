source setvars.sh

EXE="$PYTHIA_HOME/bin/perceptron-multi-multi-no-ship-1core"
EXPERIMENTS_DIR="$PYTHIA_HOME/experiments"
ROLLUP_SCRIPT="$PYTHIA_HOME/scripts/rollup.pl"
EXP_1C_PATHFINDER="single_1C_base_config.exp"
TLIST="$PYTHIA_HOME/demo_short.tlist"

echo "Running PATHFINDER experiments..."
cd "$EXPERIMENTS_DIR" || exit
perl ../scripts/create_jobfile.pl --exe "$EXE" --tlist "$TLIST" --exp "$EXP_1C_PATHFINDER" --local 1 > jobfile_1C_PATHFINDER.sh
cd experiments_1C || exit
source ../jobfile_1C_PATHFINDER.sh
cd ..
echo "Rolling up statistics for PATHFINDER..."
cd experiments_1C || exit
$ROLLUP_SCRIPT --tlist "../$TLIST" --exp "../$EXP_1C_PATHFINDER" --mfile "../$MFILE_1C_BASE" > ../rollup_1C_PATHFINDER.csv
cd ..