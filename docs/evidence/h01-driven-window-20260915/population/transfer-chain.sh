#!/usr/bin/env bash
# Stage C chain: six registered transfer runs, then dt-halving repeats of the primary inputs.
cd /workspace/braintrace
R=var/h01-driven/run_transfer.sh
E="--cell 955432427 --donor l2-pyramidal-allen-541563728 --polarity E --pulse-on-ms 1020 --pulse-ms 1000 --duration-ms 2100"
L4="--cell 3761379470 --donor l4-pyramidal-allen-527952884 --polarity E --pulse-on-ms 1020 --pulse-ms 1000 --duration-ms 2100"
SST="--cell 4420044370 --donor l3-sst-interneuron-hl5mn1 --polarity I --pulse-on-ms 270 --pulse-ms 1000 --duration-ms 1500"
PV="--cell 4853956860 --donor l5-pv-basket-hl5bn1 --polarity I --pulse-on-ms 270 --pulse-ms 1000 --duration-ms 1500"
$R transfer-e310 1800 -- $E --current-na 0.31 --registered-count 10 --donor-model-count 10
$R transfer-e200 1800 -- $E --current-na 0.20 --registered-count 1 --repeat-counts 1 1 1 1 1 0 0 --donor-model-count 4
$R transfer-l4-090 1800 -- $L4 --current-na 0.09 --registered-count 12 --donor-model-count 8
$R transfer-sst-100 1800 -- $SST --current-na 0.10 --registered-count 14 --repeat-counts 14 14 13 12 --donor-model-count 16
$R transfer-pv-190 1800 -- $PV --current-na 0.19 --registered-count 12 --donor-model-count 14
$R transfer-pv-270 1800 -- $PV --current-na 0.27 --registered-count 43 --donor-model-count 37
$R transfer-e310-dthalf 3600 -- $E --current-na 0.31 --registered-count 10 --donor-model-count 10 --dt-ms 0.0025
$R transfer-l4-090-dthalf 3600 -- $L4 --current-na 0.09 --registered-count 12 --donor-model-count 8 --dt-ms 0.0025
$R transfer-sst-100-dthalf 3600 -- $SST --current-na 0.10 --registered-count 14 --repeat-counts 14 14 13 12 --donor-model-count 16 --dt-ms 0.0025
$R transfer-pv-190-dthalf 3600 -- $PV --current-na 0.19 --registered-count 12 --donor-model-count 14 --dt-ms 0.0025
echo CHAIN-DONE > var/h01-driven/transfer-chain.done
