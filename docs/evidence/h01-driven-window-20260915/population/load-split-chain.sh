#!/usr/bin/env bash
cd /workspace/braintrace
E="--cell 955432427 --donor l2-pyramidal-allen-541563728 --polarity E --pulse-on-ms 1020 --pulse-ms 1000 --duration-ms 2100 --current-na 0.31 --registered-count 10 --donor-model-count 10"
for k in 3 10; do
  OUT=var/h01-driven/split-e310-dendload-x$k; mkdir -p $OUT
  CPUSET=232-247 XLA_FLAGS=--xla_gpu_force_compilation_parallelism=8 XLA_PYTHON_CLIENT_MEM_FRACTION=.3 PYTHONPATH=. timeout -k 30 1800 taskset -c 232-247 /workspace/venv314/bin/python3 -u docs/evidence/h01_transfer_load_split.py --dend-load-scale $k $E --output $OUT/run > $OUT/run.log 2> $OUT/run.err
  echo "DONE x$k rc=$?" >> $OUT/run.log
done
