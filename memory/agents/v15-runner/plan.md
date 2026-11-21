# V15 runner plan (2026-11-20 01:05 UTC)

Facts: the fleet scheduler packs by declared gpu-mem only; it does not see the q2 evals that already fill each 4090 (about 16 of 24 GB) and 4060 (about 6.7 of 8 GB). 24 q2 lanes running, 40/45 claimed, 5 unclaimed.
Plan:
1. mini-00 is idle: start 4 q2 lanes there now, so the 5 unclaimed q2 runs go to mini-00 instead of to the 4090s.
2. Submit v15-seeds (4 runs) now with --gpu-mem 13 so only 1 run fits per 4090 (about 8 GB free): they start at once without OOM.
3. Submit v15-speed (24 runs, --gpu-mem 4.5, prio 40) once the q2 evals on the 4090s have drained (the first ones end about 02:00 UTC); check nvidia-smi free memory first. Pin with --host if a host drains late.
4. Posthoc for v15: a new queue posthoc_q3 (stream_eval then z_use, exit status checked), lanes on mini-00 and on 4060s/4090s as training finishes.
Log below.

## Log (UTC)
- 01:03 4 q2 lanes started on mini-00 (all 45 q2 runs are now claimed; no new evals start on the 4090s).
- 01:05 v15-seeds (4 runs) submitted with --gpu-mem 13 (1 per 4090). 4060 is not in --res 4090, so training never lands there: 4060s and mini-00 are posthoc only.
- Stage 2 watcher (scratchpad/stage2.sh): phase A pins 2 v15-speed runs per host when its 4090 has no q2 eval (used <= 6 GB); phase B, when all 4 hosts have drained, stops and retries the seeds runs with --gpu-mem 4.5 (so declared memory matches), then submits the other 16 v15-speed runs.
- Feeder/autoscaler (scratchpad/q3.sh): appends completed v15 runs (and any q2 run whose eval exited non-zero) to posthoc_q3/list; starts lane3.sh (stream_eval then z_use, retried once on failure) on a GPU when free memory >= 5.3 GB; 4090 lanes only after all training is submitted and no run is queued.
- 01:50 stage2 done: seeds restacked to 4.5 GB, 24 v15-speed submitted; waiting for q3.sh 'pending posthoc=0'.
- 05:40 UTC feeder bug: list stayed empty, v15 posthoc not started (4 h lost on posthoc). All training finished 05:35. Wrote list by hand (28 v15 + 2 q2 failures v14-width f50950b2, fce57682), started 5 lanes/4090, 2/4060, 4 on mini-00.
