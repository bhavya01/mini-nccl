# all_gather benchmark — mini_nccl vs nccl

Aggregated over **10 runs** per backend, 20 timed iterations each (5 warmup). world_size=2, float32, latency = max across ranks, averaged per timed iteration.

Algorithm bandwidth follows NCCL convention: `bus_bytes = input_bytes * (world_size - 1)`.

## Latency (ms) — mean ± stddev [min–max], n=10

| per-rank | total out | mini_nccl | nccl | speedup (nccl) |
|---|---|---|---|---|
| 1.0 MiB | 2.0 MiB | 10.222 ± 1.185 [6.667–10.645] | 0.346 ± 0.002 [0.344–0.348] | 29.5× |
| 4.0 MiB | 8.0 MiB | 10.174 ± 2.258 [3.399–10.938] | 1.332 ± 0.007 [1.316–1.339] | 7.6× |
| 16.0 MiB | 32.0 MiB | 10.416 ± 3.087 [4.424–13.312] | 5.256 ± 0.259 [4.486–5.401] | 2.0× |
| 64.0 MiB | 128.0 MiB | 18.641 ± 1.746 [15.714–21.403] | 20.570 ± 0.961 [18.262–21.788] | 0.9× |
| 128.0 MiB | 256.0 MiB | 31.961 ± 2.072 [29.235–35.819] | 40.296 ± 1.957 [36.422–43.276] | 0.8× |
| 256.0 MiB | 512.0 MiB | 60.002 ± 1.431 [57.283–62.639] | 79.792 ± 3.043 [72.801–84.085] | 0.8× |

## Algorithm bandwidth (GB/s) — mean ± stddev [min–max], n=10

| per-rank | total out | mini_nccl | nccl |
|---|---|---|---|
| 1.0 MiB | 2.0 MiB | 0.11 ± 0.02 [0.10–0.16] | 3.03 ± 0.01 [3.01–3.05] |
| 4.0 MiB | 8.0 MiB | 0.47 ± 0.26 [0.38–1.23] | 3.15 ± 0.02 [3.13–3.19] |
| 16.0 MiB | 32.0 MiB | 1.85 ± 0.85 [1.26–3.79] | 3.20 ± 0.18 [3.11–3.74] |
| 64.0 MiB | 128.0 MiB | 3.63 ± 0.34 [3.14–4.27] | 3.27 ± 0.16 [3.08–3.67] |
| 128.0 MiB | 256.0 MiB | 4.22 ± 0.26 [3.75–4.59] | 3.34 ± 0.17 [3.10–3.69] |
| 256.0 MiB | 512.0 MiB | 4.48 ± 0.11 [4.29–4.69] | 3.37 ± 0.13 [3.19–3.69] |

---

# all_reduce benchmark — mini_nccl vs nccl

Aggregated over **10 runs** per backend, 20 timed iterations each (5 warmup). world_size=2, float32, op=SUM, in-place. Latency is **rank 0's local** elapsed time per timed iteration — not the cross-rank max used in the all_gather benchmark (see note below).

Algorithm bandwidth follows the NCCL ring all-reduce convention: `bus_bytes = input_bytes * 2 * (world_size - 1) / world_size`.

## Latency (ms) — mean ± stddev [min–max], n=10

| per-rank | mini_nccl | nccl | speedup (nccl) |
|---|---|---|---|
| 1.0 MiB | 10.626 ± 0.015 [10.610–10.664] | 0.290 ± 0.001 [0.288–0.292] | 36.7× |
| 4.0 MiB | 10.457 ± 1.416 [6.209–10.943] | 1.054 ± 0.014 [1.026–1.072] | 9.9× |
| 16.0 MiB | 11.368 ± 2.491 [3.895–12.214] | 4.068 ± 0.092 [3.968–4.202] | 2.8× |
| 64.0 MiB | 20.547 ± 1.987 [17.255–22.718] | 15.893 ± 0.228 [15.694–16.498] | 1.3× |
| 128.0 MiB | 34.086 ± 2.680 [28.614–37.230] | 31.426 ± 0.092 [31.295–31.569] | 1.1× |
| 256.0 MiB | 59.860 ± 2.769 [56.356–63.944] | 62.704 ± 0.136 [62.551–62.974] | 1.0× |

## Algorithm bandwidth (GB/s) — mean ± stddev [min–max], n=10

| per-rank | mini_nccl | nccl |
|---|---|---|
| 1.0 MiB | 0.10 ± 0.00 [0.10–0.10] | 3.62 ± 0.02 [3.59–3.64] |
| 4.0 MiB | 0.41 ± 0.09 [0.38–0.68] | 3.98 ± 0.06 [3.91–4.09] |
| 16.0 MiB | 1.67 ± 0.88 [1.37–4.31] | 4.13 ± 0.09 [3.99–4.23] |
| 64.0 MiB | 3.30 ± 0.34 [2.95–3.89] | 4.22 ± 0.06 [4.07–4.28] |
| 128.0 MiB | 3.96 ± 0.33 [3.61–4.69] | 4.27 ± 0.01 [4.25–4.29] |
| 256.0 MiB | 4.50 ± 0.21 [4.20–4.76] | 4.28 ± 0.01 [4.26–4.29] |

---

# all_to_all benchmark — mini_nccl vs nccl

Aggregated over **10 runs** per backend, 20 timed iterations each (5 warmup). world_size=2, float32, `all_to_all_single` with an even split. Latency is **rank 0's local** elapsed time per timed iteration (no cross-rank reduction — see the all_reduce note).

Algorithm bandwidth follows the NCCL all-to-all convention: `bus_bytes = input_bytes * (world_size - 1) / world_size` (the self-destined chunk stays local).

## Latency (ms) — mean ± stddev [min–max], n=10

| per-rank | mini_nccl | nccl | speedup (nccl) |
|---|---|---|---|
| 1.0 MiB | 10.756 ± 0.010 [10.741–10.772] | 0.166 ± 0.001 [0.164–0.168] | 64.8× |
| 4.0 MiB | 11.102 ± 0.024 [11.076–11.153] | 0.554 ± 0.003 [0.549–0.558] | 20.0× |
| 16.0 MiB | 11.905 ± 0.115 [11.563–11.965] | 2.069 ± 0.016 [2.051–2.098] | 5.8× |
| 64.0 MiB | 16.410 ± 0.549 [15.472–17.351] | 7.659 ± 0.041 [7.597–7.715] | 2.1× |
| 128.0 MiB | 23.846 ± 0.727 [22.678–25.496] | 15.235 ± 0.060 [15.101–15.304] | 1.6× |
| 256.0 MiB | 42.591 ± 1.018 [40.913–44.305] | 30.456 ± 0.109 [30.264–30.603] | 1.4× |

## Algorithm bandwidth (GB/s) — mean ± stddev [min–max], n=10

| per-rank | mini_nccl | nccl |
|---|---|---|
| 1.0 MiB | 0.05 ± 0.00 [0.05–0.05] | 3.16 ± 0.02 [3.12–3.19] |
| 4.0 MiB | 0.19 ± 0.00 [0.19–0.19] | 3.79 ± 0.02 [3.76–3.82] |
| 16.0 MiB | 0.70 ± 0.01 [0.70–0.73] | 4.05 ± 0.03 [4.00–4.09] |
| 64.0 MiB | 2.05 ± 0.07 [1.93–2.17] | 4.38 ± 0.02 [4.35–4.42] |
| 128.0 MiB | 2.82 ± 0.09 [2.63–2.96] | 4.40 ± 0.02 [4.39–4.44] |
| 256.0 MiB | 3.15 ± 0.07 [3.03–3.28] | 4.41 ± 0.01 [4.39–4.43] |