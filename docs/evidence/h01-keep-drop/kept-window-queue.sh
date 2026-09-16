#!/usr/bin/env bash
# Kept-network driven window; registered in docs/specs/2026-09-16-h01-keep-drop.md.
cd /workspace/braintrace
R=var/h01-driven/run_h01.sh
wait_done() { while [ ! -f var/h01-driven/$1/terminal.json ]; do sleep 60; done; }
run() { CPUSET=$1 nohup $2 > /dev/null 2>&1 & }
CPUSET=0-63 $R kept-bench-ei-1ms 14400 -- --topology docs/evidence/h01-kept-network.json --cache .cache/h01 --solver h01_staggered_calcium_implicit --include-isolated --cells 17 --components docs/evidence/h01-kept-components.json --currents-json docs/evidence/h01-keep-drop/kept-currents.json --pulse-delay-ms 2 --pulse-duration-ms 38 --max-cv-um 10 --heartbeat-s 60 --duration-ms 1 --dt-ms 0.000625 --control ei
run 0-63 "$R kept-ctrl-ei-50ms 14400 -- --topology docs/evidence/h01-kept-network.json --cache .cache/h01 --solver h01_staggered_calcium_implicit --include-isolated --cells 17 --components docs/evidence/h01-kept-components.json --currents-json docs/evidence/h01-keep-drop/kept-currents.json --pulse-delay-ms 2 --pulse-duration-ms 38 --max-cv-um 10 --heartbeat-s 60 --duration-ms 50 --dt-ms 0.000625 --control ei"
sleep 600
run 64-127 "$R kept-ctrl-e_only-50ms 14400 -- --topology docs/evidence/h01-kept-network.json --cache .cache/h01 --solver h01_staggered_calcium_implicit --include-isolated --cells 17 --components docs/evidence/h01-kept-components.json --currents-json docs/evidence/h01-keep-drop/kept-currents.json --pulse-delay-ms 2 --pulse-duration-ms 38 --max-cv-um 10 --heartbeat-s 60 --duration-ms 50 --dt-ms 0.000625 --control e_only"
wait_done kept-ctrl-ei-50ms
wait_done kept-ctrl-e_only-50ms
run 0-63 "$R kept-ctrl-i_only-50ms 14400 -- --topology docs/evidence/h01-kept-network.json --cache .cache/h01 --solver h01_staggered_calcium_implicit --include-isolated --cells 17 --components docs/evidence/h01-kept-components.json --currents-json docs/evidence/h01-keep-drop/kept-currents.json --pulse-delay-ms 2 --pulse-duration-ms 38 --max-cv-um 10 --heartbeat-s 60 --duration-ms 50 --dt-ms 0.000625 --control i_only"
sleep 600
run 64-127 "$R kept-ctrl-disconnected-50ms 14400 -- --topology docs/evidence/h01-kept-network.json --cache .cache/h01 --solver h01_staggered_calcium_implicit --include-isolated --cells 17 --components docs/evidence/h01-kept-components.json --currents-json docs/evidence/h01-keep-drop/kept-currents.json --pulse-delay-ms 2 --pulse-duration-ms 38 --max-cv-um 10 --heartbeat-s 60 --duration-ms 50 --dt-ms 0.000625 --control disconnected"
wait_done kept-ctrl-i_only-50ms
wait_done kept-ctrl-disconnected-50ms
run 0-63 "$R kept-refine-ei-10ms-dt000625 14400 -- --topology docs/evidence/h01-kept-network.json --cache .cache/h01 --solver h01_staggered_calcium_implicit --include-isolated --cells 17 --components docs/evidence/h01-kept-components.json --currents-json docs/evidence/h01-keep-drop/kept-currents.json --pulse-delay-ms 2 --pulse-duration-ms 38 --max-cv-um 10 --heartbeat-s 60 --duration-ms 10 --dt-ms 0.000625 --control ei"
sleep 600
run 64-127 "$R kept-refine-ei-10ms-dt0003125 14400 -- --topology docs/evidence/h01-kept-network.json --cache .cache/h01 --solver h01_staggered_calcium_implicit --include-isolated --cells 17 --components docs/evidence/h01-kept-components.json --currents-json docs/evidence/h01-keep-drop/kept-currents.json --pulse-delay-ms 2 --pulse-duration-ms 38 --max-cv-um 10 --heartbeat-s 60 --duration-ms 10 --dt-ms 0.0003125 --control ei"
wait_done kept-refine-ei-10ms-dt000625
wait_done kept-refine-ei-10ms-dt0003125
echo QUEUE-DONE > var/h01-driven/kept-window.done
