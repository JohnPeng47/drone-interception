# Intercept Env Benchmarks - 2026-06-03

Scenario file:

```text
scripts/generators/sim_instances/sobol_samples_512.csimin
```

Observation contract:

```text
25D = pursuer position(3), velocity(3), attitude rotation matrix(9),
      previous normalized action(4), target position(3), target velocity(3)
```

## Short Run

Command:

```bash
python scripts/runners/benchmark_intercept_envs.py --mode both --num-envs 8 --steps 512 --scenario-file scripts/generators/sim_instances/sobol_samples_512.csimin
python scripts/runners/benchmark_intercept_envs.py --mode both --num-envs 8 --steps 128 --policy-latency-us 250 --scenario-file scripts/generators/sim_instances/sobol_samples_512.csimin
```

Results:

| Mode | Policy Latency | Env Steps | Elapsed s | Sim Steps/s | Obs Shape | Terminal Count |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| simengine_batch | 0 us | 4096 | 0.168761 | 24271.1 | [8, 25] | 1 |
| puffer_native | 0 us | 4096 | 0.022020 | 186012.1 | [8, 25] | 1 |
| simengine_batch | 250 us | 1024 | 0.072903 | 14046.1 | [8, 25] | 0 |
| puffer_native | 250 us | 1024 | 0.040493 | 25288.6 | [8, 25] | 0 |

## Longer Run

Command:

```bash
python scripts/runners/benchmark_intercept_envs.py --mode both --num-envs 8 --steps 4096 --scenario-file scripts/generators/sim_instances/sobol_samples_512.csimin
python scripts/runners/benchmark_intercept_envs.py --mode both --num-envs 8 --steps 512 --policy-latency-us 250 --scenario-file scripts/generators/sim_instances/sobol_samples_512.csimin
```

Results:

| Mode | Policy Latency | Env Steps | Elapsed s | Sim Steps/s | Obs Shape | Terminal Count |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| simengine_batch | 0 us | 32768 | 1.381748 | 23714.9 | [8, 25] | 52 |
| puffer_native | 0 us | 32768 | 0.145831 | 224697.7 | [8, 25] | 52 |
| simengine_batch | 250 us | 4096 | 0.363675 | 11262.8 | [8, 25] | 1 |
| puffer_native | 250 us | 4096 | 0.163924 | 24987.2 | [8, 25] | 1 |

## Notes

- The short no-latency Puffer run is very sensitive to timer noise because it only measures 4096 env steps.
- The longer no-latency Puffer run gives a more stable estimate for this 25D observation contract: about 225k sim steps/s.
- Compared with the earlier 26D benchmark in this thread, `simengine_batch` is not slower; it was around 18k sim steps/s before and is around 24k in the short 25D run / 23.7k in the longer 25D run.
- `puffer_native` is lower than the earlier short 26D run that measured about 440k sim steps/s. That old value was also a very short run and may have been inflated by measurement noise. The 25D C observation now computes and writes a full rotation matrix and previous-action fields, but we need a controlled before/after benchmark with identical run length to separate real overhead from short-run noise.
