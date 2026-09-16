#!/usr/bin/env bash
# Serial-pair queue for the remaining population runs, guarded by the container thread budget.
cd /workspace/braintrace
C="--topology docs/evidence/h01-verified-network.json --cache .cache/h01 --solver h01_staggered_calcium_implicit --include-isolated --cells 104 --current-na 1 --components docs/evidence/h01-population-components.json --heartbeat-s 60"
LOG=var/h01-driven/queue.log
threads() { ps -eo nlwp | awk '{s+=$1} END {print s}'; }
wait_done() { while [ ! -f "var/h01-driven/$1/terminal.json" ]; do sleep 60; done; echo "$(date -u +%FT%TZ) done $1 $(cat var/h01-driven/$1/terminal.json)" >> $LOG; }
guard() { while [ "$(threads)" -gt "$1" ]; do echo "$(date -u +%FT%TZ) threads $(threads) > $1, waiting" >> $LOG; sleep 120; done; }
launch() { local label=$1 cpus=$2; shift 2; guard 4200; echo "$(date -u +%FT%TZ) launch $label threads=$(threads)" >> $LOG; CPUSET=$cpus nohup var/h01-driven/run_h01.sh "$label" 18000 -- $C "$@" > /dev/null 2>&1 & }
wait_done ctrl-ei-50ms-r3
wait_done ctrl-i_only-50ms-r3
launch ctrl-e_only-50ms-r4 0-63 --duration-ms 50 --dt-ms 0.000625 --control e_only
sleep 900
launch ctrl-disconnected-50ms-r4 64-127 --duration-ms 50 --dt-ms 0.000625 --control disconnected
wait_done ctrl-e_only-50ms-r4
wait_done ctrl-disconnected-50ms-r4
launch refine-ei-10ms-dt000625-r4 0-63 --duration-ms 10 --dt-ms 0.000625 --control ei
sleep 900
launch refine-ei-10ms-dt0003125-r4 64-127 --duration-ms 10 --dt-ms 0.0003125 --control ei
wait_done refine-ei-10ms-dt000625-r4
wait_done refine-ei-10ms-dt0003125-r4
echo "$(date -u +%FT%TZ) QUEUE-DONE" >> $LOG
