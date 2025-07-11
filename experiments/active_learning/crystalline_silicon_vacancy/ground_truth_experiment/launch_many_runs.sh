export OMP_NUM_THREADS=4
for run_id in {1..10}; do
    mkdir run"$run_id"
    cp artn.in clean.sh conf.sw lammps.in Si.sw run"$run_id"/
    cd  run"$run_id"
    mpirun -np 4 /Users/brunorousseau/sources/lammps/build/lmp < lammps.in
    cd ..
done

