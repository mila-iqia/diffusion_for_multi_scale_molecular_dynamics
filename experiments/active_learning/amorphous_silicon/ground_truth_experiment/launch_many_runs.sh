export OMP_NUM_THREADS=4

REFERENCE_DIR=../../../reference/


mkdir ./calculation_runs/
cd calculation_runs/

for run_id in {1..1000}; do
    mkdir run"$run_id"

    cp ../lammps.in ../Si.sw run"$run_id"/
    cd  run"$run_id"

    ln -s $REFERENCE_DIR/artn.in artn.in
    ln -s $REFERENCE_DIR/initial_configuration.dat initial_configuration.dat

    mpirun -np 8 /Users/brunorousseau/sources/lammps/build/lmp < lammps.in
    cd ..
done

