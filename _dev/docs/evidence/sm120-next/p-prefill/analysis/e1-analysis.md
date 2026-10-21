## prefill cold TTFT (profile_client prefill, 6 requests, same prompts per index)
- e1-base1: n=6 median 2.2515 s, range 2.2375-2.2656, prompt_tokens 18939, hits 0.0
- e1-base2: n=6 median 2.2808 s, range 2.2645-2.2952, prompt_tokens 18939, hits 0.0
- e1-junction: n=6 median 2.2783 s, range 2.2653-2.2944, prompt_tokens 18939, hits 0.0
- e1-modeall: n=6 median 2.2863 s, range 2.2692-2.2989, prompt_tokens 18939, hits 0.0
- e1-nomtp: n=6 median 2.2616 s, range 2.2484-2.2717, prompt_tokens 18939, hits 0.0
- e1-nopc: n=6 median 2.2465 s, range 2.2353-2.2582, prompt_tokens 18939, hits 0.0
- no-prefix-caching vs mean(base1,base2): paired median -0.915% (range -1.118..-0.453); base1->base2 drift median +1.298%

## scheduler chunking (SPLITPROBE in server.log)
### e1-ctrl
- prompt 3028: 3028 (asked 3028, start 0) + 3028 (asked 3028, start 0) + 3028 (asked 3028, start 0)
- prompt 3029: 3029 (asked 3029, start 0) + 3029 (asked 3029, start 0) + 3029 (asked 3029, start 0)
- prompt 3030: 3030 (asked 3030, start 0)
- prompt 3031: 3031 (asked 3031, start 0)
- prompt 3033: 3033 (asked 3033, start 0) + 3033 (asked 3033, start 0)
- prompt 16062: 12928 (asked 16062, start 0) + 3134 (asked 3134, start 12928) + 12928 (asked 16062, start 0) + 3134 (asked 3134, start 12928) + 12928 (asked 16062, start 0) + 3134 (asked 3134, start 12928) + 12928 (asked 16062, start 0) + 3134 (asked 3134, start 12928)
- prompt 16067: 3139 (asked 3139, start 0) + 12928 (asked 16067, start 0) + 3139 (asked 3139, start 12928) + 12928 (asked 16067, start 0) + 3139 (asked 3139, start 12928) + 12928 (asked 16067, start 0) + 3139 (asked 3139, start 12928)
- prompt 17065: 1616 (asked 4137, start 0) + 2521 (asked 2521, start 14544) + 14544 (asked 17065, start 0) + 2521 (asked 2521, start 14544) + 14544 (asked 17065, start 0) + 2521 (asked 2521, start 14544) + 14544 (asked 17065, start 0) + 2521 (asked 2521, start 14544)
- prompt 17069: 1616 (asked 4141, start 0) + 2525 (asked 2525, start 14544) + 14544 (asked 17069, start 0) + 2525 (asked 2525, start 14544) + 14544 (asked 17069, start 0) + 2525 (asked 2525, start 14544) + 14544 (asked 17069, start 0) + 2525 (asked 2525, start 14544)
- prompt 18936: 16160 (asked 18936, start 0) + 2776 (asked 2776, start 16160)
- prompt 18946: 16160 (asked 18946, start 0) + 2786 (asked 2786, start 16160)
- prompt 18947: 16160 (asked 18947, start 0) + 2787 (asked 2787, start 16160)
- prompt 19063: 16160 (asked 19063, start 0) + 2903 (asked 2903, start 16160)
- prompt 19065: 3232 (asked 6137, start 0) + 2905 (asked 2905, start 16160) + 16160 (asked 19065, start 0) + 2905 (asked 2905, start 16160) + 16160 (asked 19065, start 0) + 2905 (asked 2905, start 16160) + 16160 (asked 19065, start 0) + 2905 (asked 2905, start 16160)
- prompt 19069: 16160 (asked 19069, start 0) + 2909 (asked 2909, start 16160)
- prompt 19071: 12928 (asked 19071, start 0) + 3232 (asked 6143, start 12928) + 2911 (asked 2911, start 16160) + 16160 (asked 19071, start 0) + 2911 (asked 2911, start 16160) + 16160 (asked 19071, start 0) + 2911 (asked 2911, start 16160) + 16160 (asked 19071, start 0) + 2911 (asked 2911, start 16160)
- prompt 19219: 3059 (asked 3059, start 0) + 16160 (asked 19219, start 0) + 3059 (asked 3059, start 16160) + 16160 (asked 19219, start 0) + 3059 (asked 3059, start 16160) + 16160 (asked 19219, start 0) + 3059 (asked 3059, start 16160)
- prompt 19220: 3060 (asked 3060, start 0) + 16160 (asked 19220, start 0) + 3060 (asked 3060, start 16160) + 16160 (asked 19220, start 0) + 3060 (asked 3060, start 16160) + 16160 (asked 19220, start 0) + 3060 (asked 3060, start 16160)
- prompt 19468: 1616 (asked 3308, start 0) + 1692 (asked 1692, start 17776) + 17776 (asked 19468, start 0) + 1692 (asked 1692, start 17776) + 17776 (asked 19468, start 0) + 1692 (asked 1692, start 17776) + 17776 (asked 19468, start 0) + 1692 (asked 1692, start 17776)
- prompt 19488: 1616 (asked 3328, start 0) + 1712 (asked 1712, start 17776) + 17776 (asked 19488, start 0) + 1712 (asked 1712, start 17776) + 17776 (asked 19488, start 0) + 1712 (asked 1712, start 17776) + 17776 (asked 19488, start 0) + 1712 (asked 1712, start 17776)
- prompt 43937: 32320 (asked 32768, start 0) + 9696 (asked 11617, start 32320) + 1921 (asked 1921, start 42016)
### e1-split-mtp4
- prompt 1001: 1001 (asked 1001, start 0)
- prompt 2929: 2929 (asked 2929, start 0) + 2929 (asked 2929, start 0)
- prompt 4232: 1616 (asked 4232, start 0) + 2616 (asked 2616, start 1616)
- prompt 7464: 4848 (asked 7464, start 0) + 2616 (asked 2616, start 4848)
- prompt 10696: 8080 (asked 10696, start 0) + 2616 (asked 2616, start 8080)
- prompt 13928: 11312 (asked 13928, start 0) + 2616 (asked 2616, start 11312)
- prompt 16067: 12928 (asked 16067, start 0) + 3139 (asked 3139, start 12928) + 12928 (asked 16067, start 0) + 3139 (asked 3139, start 12928) + 12928 (asked 16067, start 0) + 3139 (asked 3139, start 12928)
- prompt 16069: 12928 (asked 16069, start 0) + 3141 (asked 3141, start 12928) + 12928 (asked 16069, start 0) + 3141 (asked 3141, start 12928) + 12928 (asked 16069, start 0) + 3141 (asked 3141, start 12928)
- prompt 17066: 1616 (asked 4138, start 0) + 2522 (asked 2522, start 14544) + 14544 (asked 17066, start 0) + 2522 (asked 2522, start 14544) + 14544 (asked 17066, start 0) + 2522 (asked 2522, start 14544)
- prompt 17069: 1616 (asked 4141, start 0) + 2525 (asked 2525, start 14544) + 14544 (asked 17069, start 0) + 2525 (asked 2525, start 14544) + 14544 (asked 17069, start 0) + 2525 (asked 2525, start 14544)
- prompt 17159: 12928 (asked 17159, start 0) + 1616 (asked 4231, start 12928) + 2615 (asked 2615, start 14544)
- prompt 18834: 16160 (asked 18834, start 0) + 2674 (asked 2674, start 16160)
- prompt 18835: 16160 (asked 18835, start 0) + 2675 (asked 2675, start 16160)
- prompt 18936: 16160 (asked 18936, start 0) + 2776 (asked 2776, start 16160)
- prompt 18944: 16160 (asked 18944, start 0) + 2784 (asked 2784, start 16160) + 2784 (asked 2784, start 0) + 2784 (asked 2784, start 0)
- prompt 18946: 16160 (asked 18946, start 0) + 2786 (asked 2786, start 16160)
- prompt 18948: 16160 (asked 18948, start 0) + 2788 (asked 2788, start 16160) + 2788 (asked 2788, start 0) + 2788 (asked 2788, start 0)
- prompt 18950: 16160 (asked 18950, start 0) + 2790 (asked 2790, start 16160)
- prompt 19000: 16160 (asked 19000, start 0) + 2840 (asked 2840, start 16160) + 16160 (asked 19000, start 0) + 2840 (asked 2840, start 16160) + 16160 (asked 19000, start 0) + 2840 (asked 2840, start 16160) + 16160 (asked 19000, start 0) + 2840 (asked 2840, start 16160) + 16160 (asked 19000, start 0) + 2840 (asked 2840, start 16160) + 16160 (asked 19000, start 0) + 2840 (asked 2840, start 16160)
- prompt 19067: 3232 (asked 6139, start 0) + 2907 (asked 2907, start 16160) + 16160 (asked 19067, start 0) + 2907 (asked 2907, start 16160) + 16160 (asked 19067, start 0) + 2907 (asked 2907, start 16160)
- prompt 19069: 3232 (asked 6141, start 0) + 2909 (asked 2909, start 16160) + 16160 (asked 19069, start 0) + 2909 (asked 2909, start 16160) + 16160 (asked 19069, start 0) + 2909 (asked 2909, start 16160)
- prompt 19071: 16160 (asked 19071, start 0) + 2911 (asked 2911, start 16160)
- prompt 19073: 16160 (asked 19073, start 0) + 2913 (asked 2913, start 16160)
- prompt 19219: 3059 (asked 3059, start 0) + 16160 (asked 19219, start 0) + 3059 (asked 3059, start 16160) + 16160 (asked 19219, start 0) + 3059 (asked 3059, start 16160)
- prompt 19223: 3063 (asked 3063, start 0) + 16160 (asked 19223, start 0) + 3063 (asked 3063, start 16160) + 16160 (asked 19223, start 0) + 3063 (asked 3063, start 16160)
- prompt 19469: 1616 (asked 3309, start 0) + 1693 (asked 1693, start 17776) + 17776 (asked 19469, start 0) + 1693 (asked 1693, start 17776) + 17776 (asked 19469, start 0) + 1693 (asked 1693, start 17776)
- prompt 19478: 1616 (asked 3318, start 0) + 1702 (asked 1702, start 17776) + 17776 (asked 19478, start 0) + 1702 (asked 1702, start 17776) + 17776 (asked 19478, start 0) + 1702 (asked 1702, start 17776)
- prompt 32930: 30704 (asked 32768, start 0) + 2226 (asked 2226, start 30704)
- prompt 32931: 30704 (asked 32768, start 0) + 2227 (asked 2227, start 30704)
- prompt 43937: 32320 (asked 32768, start 0) + 9696 (asked 11617, start 32320) + 1921 (asked 1921, start 42016)
### e1-split-nomtp
- prompt 2929: 1568 (asked 2929, start 0) + 1361 (asked 1361, start 1568) + 1568 (asked 2929, start 0) + 1361 (asked 1361, start 1568)
- prompt 9930: 9408 (asked 9930, start 0) + 522 (asked 522, start 9408)
- prompt 9933: 9408 (asked 9933, start 0) + 525 (asked 525, start 9408)
- prompt 18834: 18816 (asked 18834, start 0) + 18 (asked 18, start 18816)
- prompt 18835: 18816 (asked 18835, start 0) + 19 (asked 19, start 18816)
- prompt 24930: 23520 (asked 24930, start 0) + 1410 (asked 1410, start 23520)
- prompt 24931: 23520 (asked 24931, start 0) + 1411 (asked 1411, start 23520)
- prompt 32930: 31360 (asked 32768, start 0) + 1568 (asked 1570, start 31360) + 2 (asked 2, start 32928)
- prompt 32931: 31360 (asked 32768, start 0) + 1568 (asked 1571, start 31360) + 3 (asked 3, start 32928)
- prompt 43931: 31360 (asked 32768, start 0) + 12544 (asked 12571, start 31360) + 27 (asked 27, start 43904)
- prompt 43934: 31360 (asked 32768, start 0) + 12544 (asked 12574, start 31360) + 30 (asked 30, start 43904)
### e1-chunk-a1 iteration_tokens deltas (count/sum per request)
- L=18900 prompt=18835 steps=1 sum=18836 ttft=2.245
- L=18900 prompt=18834 steps=1 sum=18835 ttft=2.253
- L=18900 prompt=18833 steps=1 sum=18834 ttft=2.259
- L=18900 prompt=18831 steps=1 sum=18832 ttft=2.268
- L=18900 prompt=18835 steps=1 sum=18836 ttft=2.274
- L=18900 prompt=18833 steps=1 sum=18834 ttft=2.279
- L=44000 prompt=43934 steps=1 sum=43935 ttft=6.303
- L=44000 prompt=43931 steps=1 sum=43932 ttft=6.335
- L=44000 prompt=43934 steps=1 sum=43935 ttft=6.359
- L=44000 prompt=43935 steps=1 sum=43936 ttft=6.384
- L=44000 prompt=43930 steps=1 sum=43931 ttft=6.408
- L=44000 prompt=43932 steps=1 sum=43933 ttft=6.420
### e1-chunk-a2 iteration_tokens deltas (count/sum per request)
- L=18900 prompt=18835 steps=1 sum=18836 ttft=2.243
- L=18900 prompt=18834 steps=1 sum=18835 ttft=2.254
- L=18900 prompt=18833 steps=1 sum=18834 ttft=2.259
- L=18900 prompt=18831 steps=1 sum=18832 ttft=2.266
- L=18900 prompt=18835 steps=1 sum=18836 ttft=2.270
- L=18900 prompt=18833 steps=1 sum=18834 ttft=2.275
- L=44000 prompt=43934 steps=1 sum=43935 ttft=6.295
- L=44000 prompt=43931 steps=1 sum=43932 ttft=6.330
- L=44000 prompt=43934 steps=1 sum=43935 ttft=6.359
- L=44000 prompt=43935 steps=1 sum=43936 ttft=6.382
- L=44000 prompt=43930 steps=1 sum=43931 ttft=6.407
- L=44000 prompt=43932 steps=1 sum=43933 ttft=6.415
### e1-chunk-b1 iteration_tokens deltas (count/sum per request)
- L=18900 prompt=18835 steps=1 sum=18836 ttft=2.213
- L=18900 prompt=18834 steps=1 sum=18835 ttft=2.223
- L=18900 prompt=18833 steps=1 sum=18834 ttft=2.225
- L=18900 prompt=18831 steps=1 sum=18832 ttft=2.232
- L=18900 prompt=18835 steps=1 sum=18836 ttft=2.238
- L=18900 prompt=18833 steps=1 sum=18834 ttft=2.244
- L=44000 prompt=43934 steps=1 sum=43935 ttft=6.229
- L=44000 prompt=43931 steps=1 sum=43932 ttft=6.262
- L=44000 prompt=43934 steps=1 sum=43935 ttft=6.295
- L=44000 prompt=43935 steps=1 sum=43936 ttft=6.316
- L=44000 prompt=43930 steps=1 sum=43931 ttft=6.339
- L=44000 prompt=43932 steps=1 sum=43933 ttft=6.356
### e1-chunk-b2 iteration_tokens deltas (count/sum per request)
- L=18900 prompt=18835 steps=1 sum=18836 ttft=2.216
- L=18900 prompt=18834 steps=1 sum=18835 ttft=2.225
- L=18900 prompt=18833 steps=1 sum=18834 ttft=2.230
- L=18900 prompt=18831 steps=1 sum=18832 ttft=2.235
- L=18900 prompt=18835 steps=1 sum=18836 ttft=2.241
- L=18900 prompt=18833 steps=1 sum=18834 ttft=2.245
- L=44000 prompt=43934 steps=1 sum=43935 ttft=6.233
- L=44000 prompt=43931 steps=1 sum=43932 ttft=6.262
- L=44000 prompt=43934 steps=1 sum=43935 ttft=6.294
- L=44000 prompt=43935 steps=1 sum=43936 ttft=6.315
- L=44000 prompt=43930 steps=1 sum=43931 ttft=6.335
- L=44000 prompt=43932 steps=1 sum=43933 ttft=6.350
### e1-nopc iteration_tokens deltas (count/sum per request)
- L=18900 prompt=18835 steps=1 sum=18836 ttft=2.288
- L=18900 prompt=18834 steps=1 sum=18835 ttft=2.293
- L=44000 prompt=43934 steps=1 sum=43935 ttft=6.355
- L=44000 prompt=43931 steps=1 sum=43932 ttft=6.373
### e1-split-mtp4 iteration_tokens deltas (count/sum per request)
- L=3000 prompt=2929 steps=0 sum=0 ttft=0.285
- L=3000 prompt=2929 steps=0 sum=0 ttft=0.285
- L=18900 prompt=18835 steps=0 sum=0 ttft=2.223
- L=18900 prompt=18834 steps=0 sum=0 ttft=2.231
- L=33000 prompt=32930 steps=0 sum=0 ttft=4.340
- L=33000 prompt=32931 steps=0 sum=0 ttft=4.356
### e1-split-nomtp iteration_tokens deltas (count/sum per request)
- L=3000 prompt=2929 steps=1 sum=2930 ttft=0.314
- L=3000 prompt=2929 steps=1 sum=2930 ttft=0.313
- L=10000 prompt=9930 steps=1 sum=9931 ttft=1.097
- L=10000 prompt=9933 steps=1 sum=9934 ttft=1.099
- L=18900 prompt=18835 steps=1 sum=18836 ttft=2.253
- L=18900 prompt=18834 steps=1 sum=18835 ttft=2.259
- L=25000 prompt=24931 steps=1 sum=24932 ttft=3.106
- L=25000 prompt=24930 steps=1 sum=24931 ttft=3.110
- L=33000 prompt=32930 steps=1 sum=32931 ttft=4.354
- L=33000 prompt=32931 steps=1 sum=32932 ttft=4.367
- L=44000 prompt=43934 steps=1 sum=43935 ttft=6.193
- L=44000 prompt=43931 steps=1 sum=43932 ttft=6.217

## e1-base1: prefix scenarios
| scen | request | prompt tok | hit tok | recomputed | TTFT s | role |
|---|---|---:|---:|---:|---:|---|
| S1 | 19000/s0/cold | 18948 | 0 | 18948 | 2.326 | cold |
| S1 | 19000/s0/repeat1 | 18948 | 16160 | 2788 | 0.377 | hit |
| S1 | 19000/s0/repeat2 | 18948 | 16160 | 2788 | 0.377 | hit |
| S1 | 19000/s1/cold | 18950 | 0 | 18950 | 2.326 | cold |
| S1 | 19000/s1/repeat1 | 18950 | 16160 | 2790 | 0.378 | hit |
| S1 | 19000/s1/repeat2 | 18950 | 16160 | 2790 | 0.377 | hit |
| S1 | 44000/s0/cold | 43945 | 0 | 43945 | 6.394 | cold |
| S1 | 44000/s0/repeat1 | 43945 | 42016 | 1929 | 0.383 | hit |
| S1 | 44000/s0/repeat2 | 43945 | 42016 | 1929 | 0.383 | hit |
| S1 | 44000/s1/cold | 43947 | 0 | 43947 | 6.423 | cold |
| S1 | 44000/s1/repeat1 | 43947 | 42016 | 1931 | 0.385 | hit |
| S1 | 44000/s1/repeat2 | 43947 | 42016 | 1931 | 0.385 | hit |
| S2 | 19000/s0/seed(cold) | 19069 | 0 | 19069 | 2.366 | seed |
| S2 | 19000/s0/tail1000 | 16069 | 0 | 16069 | 1.934 | hit |
| S2 | 19000/s0/tail1000/cold-a | 16069 | 0 | 16069 | 1.935 | cold |
| S2 | 19000/s0/tail1000/cold-b | 16069 | 0 | 16069 | 1.935 | cold |
| S2 | 19000/s0/tail2000 | 17070 | 12928 | 4142 | 0.570 | hit |
| S2 | 19000/s0/tail2000/cold-a | 17070 | 0 | 17070 | 2.092 | cold |
| S2 | 19000/s0/tail2000/cold-b | 17070 | 0 | 17070 | 2.095 | cold |
| S2 | 19000/s0/tail4000 | 19067 | 12928 | 6139 | 0.830 | hit |
| S2 | 19000/s0/tail4000/cold-a | 19067 | 0 | 19067 | 2.374 | cold |
| S2 | 19000/s0/tail4000/cold-b | 19067 | 0 | 19067 | 2.378 | cold |
| S2 | 19000/s1/seed(cold) | 19066 | 0 | 19066 | 2.377 | seed |
| S2 | 19000/s1/tail4000 | 19068 | 0 | 19068 | 2.362 | hit |
| S2 | 19000/s1/tail4000/cold-a | 19068 | 0 | 19068 | 2.381 | cold |
| S2 | 19000/s1/tail4000/cold-b | 19068 | 0 | 19068 | 2.382 | cold |
| S2 | 19000/s1/tail2000 | 17069 | 12928 | 4141 | 0.572 | hit |
| S2 | 19000/s1/tail2000/cold-a | 17069 | 0 | 17069 | 2.103 | cold |
| S2 | 19000/s1/tail2000/cold-b | 17069 | 0 | 17069 | 2.106 | cold |
| S2 | 19000/s1/tail1000 | 16068 | 12928 | 3140 | 0.417 | hit |
| S2 | 19000/s1/tail1000/cold-a | 16068 | 0 | 16068 | 1.947 | cold |
| S2 | 19000/s1/tail1000/cold-b | 16068 | 0 | 16068 | 1.949 | cold |
| S2 | 44000/s0/seed(cold) | 44070 | 0 | 44070 | 6.523 | seed |
| S2 | 44000/s0/tail1000 | 41070 | 0 | 41070 | 5.990 | hit |
| S2 | 44000/s0/tail1000/cold-a | 41070 | 0 | 41070 | 5.961 | cold |
| S2 | 44000/s0/tail1000/cold-b | 41070 | 0 | 41070 | 5.969 | cold |
| S2 | 44000/s0/tail4000 | 44069 | 37168 | 6901 | 1.290 | hit |
| S2 | 44000/s0/tail4000/cold-a | 44069 | 0 | 44069 | 6.547 | cold |
| S2 | 44000/s0/tail4000/cold-b | 44069 | 0 | 44069 | 6.553 | cold |
| S2 | 44000/s1/seed(cold) | 44067 | 0 | 44067 | 6.556 | seed |
| S2 | 44000/s1/tail4000 | 44064 | 0 | 44064 | 6.565 | hit |
| S2 | 44000/s1/tail4000/cold-a | 44064 | 0 | 44064 | 6.558 | cold |
| S2 | 44000/s1/tail4000/cold-b | 44064 | 0 | 44064 | 6.563 | cold |
| S2 | 44000/s1/tail1000 | 41064 | 37168 | 3896 | 0.742 | hit |
| S2 | 44000/s1/tail1000/cold-a | 41064 | 0 | 41064 | 5.988 | cold |
| S2 | 44000/s1/tail1000/cold-b | 41064 | 0 | 41064 | 5.989 | cold |
| S3 | 19000/s0/turn1 | 18947 | 0 | 18947 | 2.391 | turn1 |
| S3 | 19000/s0/turn2 | 19220 | 16160 | 3060 | 0.426 | turn2 |
| S3 | 19000/s0/turn2/cold-a | 19220 | 0 | 19220 | 2.426 | cold |
| S3 | 19000/s0/turn2/cold-b | 19220 | 0 | 19220 | 2.429 | cold |
| S3 | 19000/s0/turn3 | 19473 | 16160 | 3313 | 0.496 | turn3 |
| S3 | 19000/s0/turn3/cold-a | 19473 | 0 | 19473 | 2.483 | cold |
| S3 | 19000/s0/turn3/cold-b | 19473 | 0 | 19473 | 2.481 | cold |
| S3 | 19000/s1/turn1 | 18948 | 0 | 18948 | 2.383 | turn1 |
| S3 | 19000/s1/turn2 | 19220 | 16160 | 3060 | 0.425 | turn2 |
| S3 | 19000/s1/turn2/cold-a | 19220 | 0 | 19220 | 2.419 | cold |
| S3 | 19000/s1/turn2/cold-b | 19220 | 0 | 19220 | 2.422 | cold |
| S3 | 19000/s1/turn3 | 19473 | 16160 | 3313 | 0.495 | turn3 |
| S3 | 19000/s1/turn3/cold-a | 19473 | 0 | 19473 | 2.478 | cold |
| S3 | 19000/s1/turn3/cold-b | 19473 | 0 | 19473 | 2.479 | cold |
| S3 | 44000/s0/turn1 | 43950 | 0 | 43950 | 6.509 | turn1 |
| S3 | 44000/s0/turn2 | 44223 | 42016 | 2207 | 0.441 | turn2 |
| S3 | 44000/s0/turn2/cold-a | 44223 | 0 | 44223 | 6.567 | cold |
| S3 | 44000/s0/turn2/cold-b | 44223 | 0 | 44223 | 6.577 | cold |
| S3 | 44000/s0/turn3 | 44483 | 42016 | 2467 | 0.478 | turn3 |
| S3 | 44000/s0/turn3/cold-a | 44483 | 0 | 44483 | 6.614 | cold |
| S3 | 44000/s0/turn3/cold-b | 44483 | 0 | 44483 | 6.621 | cold |
| S3 | 44000/s1/turn1 | 43948 | 0 | 43948 | 6.534 | turn1 |
| S3 | 44000/s1/turn2 | 44221 | 42016 | 2205 | 0.444 | turn2 |
| S3 | 44000/s1/turn2/cold-a | 44221 | 0 | 44221 | 6.583 | cold |
| S3 | 44000/s1/turn2/cold-b | 44221 | 0 | 44221 | 6.589 | cold |
| S3 | 44000/s1/turn3 | 44490 | 42016 | 2474 | 0.479 | turn3 |
| S3 | 44000/s1/turn3/cold-a | 44490 | 0 | 44490 | 6.622 | cold |
| S3 | 44000/s1/turn3/cold-b | 44490 | 0 | 44490 | 6.624 | cold |

### S4 sweep (shared prefix P -> hit tokens; test prompt = P + ~1000 new tokens)
| seed L | P | P mod 1616 | hit | hit/1616 | recomputed | TTFT s |
|---:|---:|---:|---:|---:|---:|---:|
| 19000 | 1 | 1 | 0 | 0.00 | 1001 | 0.108 |
| 19000 | 800 | 800 | 0 | 0.00 | 1800 | 0.198 |
| 19000 | 1615 | 1615 | 0 | 0.00 | 2615 | 0.276 |
| 19000 | 1616 | 0 | 0 | 0.00 | 2616 | 0.276 |
| 19000 | 1617 | 1 | 0 | 0.00 | 2617 | 0.276 |
| 19000 | 3231 | 1615 | 0 | 0.00 | 4231 | 0.471 |
| 19000 | 3232 | 0 | 0 | 0.00 | 4232 | 0.471 |
| 19000 | 3233 | 1 | 0 | 0.00 | 4233 | 0.471 |
| 19000 | 4847 | 1615 | 0 | 0.00 | 5847 | 0.673 |
| 19000 | 4848 | 0 | 0 | 0.00 | 5848 | 0.655 |
| 19000 | 4849 | 1 | 0 | 0.00 | 5849 | 0.655 |
| 19000 | 6463 | 1615 | 0 | 0.00 | 7463 | 0.861 |
| 19000 | 6464 | 0 | 0 | 0.00 | 7464 | 0.841 |
| 19000 | 6465 | 1 | 0 | 0.00 | 7465 | 0.841 |
| 19000 | 8079 | 1615 | 0 | 0.00 | 9079 | 1.054 |
| 19000 | 8080 | 0 | 0 | 0.00 | 9080 | 1.038 |
| 19000 | 8081 | 1 | 0 | 0.00 | 9081 | 1.038 |
| 19000 | 9695 | 1615 | 0 | 0.00 | 10695 | 1.256 |
| 19000 | 9696 | 0 | 0 | 0.00 | 10696 | 1.245 |
| 19000 | 9697 | 1 | 0 | 0.00 | 10697 | 1.245 |
| 19000 | 11311 | 1615 | 0 | 0.00 | 12311 | 1.467 |
| 19000 | 11312 | 0 | 0 | 0.00 | 12312 | 1.449 |
| 19000 | 11313 | 1 | 0 | 0.00 | 12313 | 1.449 |
| 19000 | 12927 | 1615 | 0 | 0.00 | 13927 | 1.676 |
| 19000 | 12928 | 0 | 0 | 0.00 | 13928 | 1.667 |
| 19000 | 12929 | 1 | 0 | 0.00 | 13929 | 1.666 |
| 19000 | 14543 | 1615 | 0 | 0.00 | 15543 | 1.901 |
| 19000 | 14544 | 0 | 0 | 0.00 | 15544 | 1.887 |
| 19000 | 14545 | 1 | 0 | 0.00 | 15545 | 1.888 |
| 19000 | 15000 | 456 | 0 | 0.00 | 16000 | 1.943 |
| 19000 | 16159 | 1615 | 0 | 0.00 | 17159 | 2.128 |
| 19000 | 16160 | 0 | 0 | 0.00 | 17160 | 2.129 |
| 19000 | 16161 | 1 | 0 | 0.00 | 17161 | 2.128 |
| 19000 | 17775 | 1615 | 0 | 0.00 | 18775 | 2.374 |
| 19000 | 17776 | 0 | 16160 | 10.00 | 2616 | 0.362 |
| 19000 | 17777 | 1 | 16160 | 10.00 | 2617 | 0.362 |
| 19000 | 18000 | 224 | 16160 | 10.00 | 2840 | 0.387 |
| 19000 | 18900 | 1124 | 16160 | 10.00 | 3740 | 0.542 |
| 19000 | 19000 | 1224 | 16160 | 10.00 | 3840 | 0.555 |
| 44000 | 1616 | 0 | 0 | 0.00 | 2616 | 0.276 |
| 44000 | 3232 | 0 | 0 | 0.00 | 4232 | 0.473 |
| 44000 | 8080 | 0 | 0 | 0.00 | 9080 | 1.042 |
| 44000 | 15000 | 456 | 0 | 0.00 | 16000 | 1.949 |
| 44000 | 16159 | 1615 | 0 | 0.00 | 17159 | 2.137 |
| 44000 | 16160 | 0 | 0 | 0.00 | 17160 | 2.138 |
| 44000 | 16161 | 1 | 0 | 0.00 | 17161 | 2.139 |
| 44000 | 24239 | 1615 | 0 | 0.00 | 25239 | 3.371 |
| 44000 | 24240 | 0 | 0 | 0.00 | 25240 | 3.347 |
| 44000 | 24241 | 1 | 0 | 0.00 | 25241 | 3.340 |
| 44000 | 30000 | 912 | 0 | 0.00 | 31000 | 4.285 |
| 44000 | 30703 | 1615 | 0 | 0.00 | 31703 | 4.398 |
| 44000 | 30704 | 0 | 0 | 0.00 | 31704 | 4.382 |
| 44000 | 30705 | 1 | 0 | 0.00 | 31705 | 4.385 |
| 44000 | 32319 | 1615 | 0 | 0.00 | 33319 | 4.679 |
| 44000 | 32320 | 0 | 0 | 0.00 | 33320 | 4.650 |
| 44000 | 32321 | 1 | 0 | 0.00 | 33321 | 4.651 |
| 44000 | 33935 | 1615 | 0 | 0.00 | 34935 | 4.956 |
| 44000 | 33936 | 0 | 0 | 0.00 | 34936 | 4.924 |
| 44000 | 33937 | 1 | 0 | 0.00 | 34937 | 4.922 |
| 44000 | 40000 | 1216 | 0 | 0.00 | 41000 | 6.032 |
| 44000 | 40399 | 1615 | 0 | 0.00 | 41399 | 6.105 |
| 44000 | 40400 | 0 | 0 | 0.00 | 41400 | 6.078 |
| 44000 | 40401 | 1 | 0 | 0.00 | 41401 | 6.078 |
| 44000 | 43000 | 984 | 0 | 0.00 | 44000 | 6.585 |
| 44000 | 43631 | 1615 | 0 | 0.00 | 44631 | 6.709 |
| 44000 | 43632 | 0 | 42016 | 26.00 | 2616 | 0.499 |
| 44000 | 43633 | 1 | 42016 | 26.00 | 2617 | 0.499 |
| 44000 | 44000 | 368 | 42016 | 26.00 | 2984 | 0.551 |

### correctness gate (greedy 256 tokens; hit vs salted-cold)
| request | hit tok | first div hit-vs-coldA | coldA-vs-coldB | margin(top1-top2) at div, cold / hit |
|---|---:|---|---|---|
| S1 19000/s0/repeat1 | 16160 | None | - |  |
| S1 19000/s0/repeat2 | 16160 | None | - |  |
| S1 19000/s1/repeat1 | 16160 | None | - |  |
| S1 19000/s1/repeat2 | 16160 | None | - |  |
| S1 44000/s0/repeat1 | 42016 | None | - |  |
| S1 44000/s0/repeat2 | 42016 | None | - |  |
| S1 44000/s1/repeat1 | 42016 | None | - |  |
| S1 44000/s1/repeat2 | 42016 | None | - |  |
| S2 19000/s0/tail1000 | 0 | None | None |  |
| S2 19000/s0/tail2000 | 12928 | 42 | None | 0.000 / 0.000 |
| S2 19000/s0/tail4000 | 12928 | 30 | None | 0.125 / 0.000 |
| S2 19000/s1/tail4000 | 0 | 69 | None | 0.125 / 0.125 |
| S2 19000/s1/tail2000 | 12928 | 43 | None | 0.250 / 0.125 |
| S2 19000/s1/tail1000 | 12928 | 42 | None | 0.125 / 0.000 |
| S2 44000/s0/tail1000 | 0 | 53 | None | 0.375 / 0.125 |
| S2 44000/s0/tail4000 | 37168 | 23 | None | 0.000 / 0.125 |
| S2 44000/s1/tail4000 | 0 | 177 | None | 0.000 / 0.250 |
| S2 44000/s1/tail1000 | 37168 | 0 | None | 0.250 / 0.250 |
| S3 19000/s0/turn2 | 16160 | None | None |  |
| S3 19000/s0/turn3 | 16160 | 26 | None | 0.000 / 0.125 |
| S3 19000/s1/turn2 | 16160 | None | None |  |
| S3 19000/s1/turn3 | 16160 | 159 | None | 0.000 / 0.125 |
| S3 44000/s0/turn2 | 42016 | None | None |  |
| S3 44000/s0/turn3 | 42016 | None | None |  |
| S3 44000/s1/turn2 | 42016 | None | None |  |
| S3 44000/s1/turn3 | 42016 | None | None |  |

## e1-ctrl: prefix scenarios
| scen | request | prompt tok | hit tok | recomputed | TTFT s | role |
|---|---|---:|---:|---:|---:|---|
| S2 | 19000/s0/seed(cold) | 19063 | 0 | 19063 | 2.705 | seed |
| S2 | 19000/s0/tail1000 | 16062 | 0 | 16062 | 1.880 | hit |
| S2 | 19000/s0/tail1000/cold-a | 16062 | 0 | 16062 | 1.885 | cold |
| S2 | 19000/s0/tail1000/cold-b | 16062 | 0 | 16062 | 1.890 | cold |
| S2 | 19000/s0/tail1000/cold-c | 16062 | 0 | 16062 | 1.894 | cold |
| S2 | 19000/s0/tail2000 | 17065 | 12928 | 4137 | 0.560 | hit |
| S2 | 19000/s0/tail2000/cold-a | 17065 | 0 | 17065 | 2.056 | cold |
| S2 | 19000/s0/tail2000/cold-b | 17065 | 0 | 17065 | 2.060 | cold |
| S2 | 19000/s0/tail2000/cold-c | 17065 | 0 | 17065 | 2.063 | cold |
| S2 | 19000/s0/tail4000 | 19065 | 12928 | 6137 | 0.820 | hit |
| S2 | 19000/s0/tail4000/cold-a | 19065 | 0 | 19065 | 2.346 | cold |
| S2 | 19000/s0/tail4000/cold-b | 19065 | 0 | 19065 | 2.350 | cold |
| S2 | 19000/s0/tail4000/cold-c | 19065 | 0 | 19065 | 2.354 | cold |
| S2 | 19000/s1/seed(cold) | 19069 | 0 | 19069 | 2.355 | seed |
| S2 | 19000/s1/tail4000 | 19071 | 0 | 19071 | 2.343 | hit |
| S2 | 19000/s1/tail4000/cold-a | 19071 | 0 | 19071 | 2.366 | cold |
| S2 | 19000/s1/tail4000/cold-b | 19071 | 0 | 19071 | 2.369 | cold |
| S2 | 19000/s1/tail4000/cold-c | 19071 | 0 | 19071 | 2.372 | cold |
| S2 | 19000/s1/tail2000 | 17069 | 12928 | 4141 | 0.570 | hit |
| S2 | 19000/s1/tail2000/cold-a | 17069 | 0 | 17069 | 2.094 | cold |
| S2 | 19000/s1/tail2000/cold-b | 17069 | 0 | 17069 | 2.096 | cold |
| S2 | 19000/s1/tail2000/cold-c | 17069 | 0 | 17069 | 2.097 | cold |
| S2 | 19000/s1/tail1000 | 16067 | 12928 | 3139 | 0.416 | hit |
| S2 | 19000/s1/tail1000/cold-a | 16067 | 0 | 16067 | 1.943 | cold |
| S2 | 19000/s1/tail1000/cold-b | 16067 | 0 | 16067 | 1.946 | cold |
| S2 | 19000/s1/tail1000/cold-c | 16067 | 0 | 16067 | 1.945 | cold |
| S3 | 19000/s0/turn1 | 18947 | 0 | 18947 | 2.369 | turn1 |
| S3 | 19000/s0/turn2 | 19220 | 16160 | 3060 | 0.422 | turn2 |
| S3 | 19000/s0/turn2/cold-a | 19220 | 0 | 19220 | 2.410 | cold |
| S3 | 19000/s0/turn2/cold-b | 19220 | 0 | 19220 | 2.412 | cold |
| S3 | 19000/s0/turn2/cold-c | 19220 | 0 | 19220 | 2.414 | cold |
| S3 | 19000/s0/turn3 | 19468 | 16160 | 3308 | 0.493 | turn3 |
| S3 | 19000/s0/turn3/cold-a | 19468 | 0 | 19468 | 2.469 | cold |
| S3 | 19000/s0/turn3/cold-b | 19468 | 0 | 19468 | 2.470 | cold |
| S3 | 19000/s0/turn3/cold-c | 19468 | 0 | 19468 | 2.470 | cold |
| S3 | 19000/s1/turn1 | 18946 | 0 | 18946 | 2.377 | turn1 |
| S3 | 19000/s1/turn2 | 19219 | 16160 | 3059 | 0.424 | turn2 |
| S3 | 19000/s1/turn2/cold-a | 19219 | 0 | 19219 | 2.416 | cold |
| S3 | 19000/s1/turn2/cold-b | 19219 | 0 | 19219 | 2.422 | cold |
| S3 | 19000/s1/turn2/cold-c | 19219 | 0 | 19219 | 2.419 | cold |
| S3 | 19000/s1/turn3 | 19488 | 16160 | 3328 | 0.494 | turn3 |
| S3 | 19000/s1/turn3/cold-a | 19488 | 0 | 19488 | 2.476 | cold |
| S3 | 19000/s1/turn3/cold-b | 19488 | 0 | 19488 | 2.475 | cold |
| S3 | 19000/s1/turn3/cold-c | 19488 | 0 | 19488 | 2.476 | cold |

### correctness gate (greedy 256 tokens; hit vs salted-cold)
| request | hit tok | first div hit-vs-coldA | coldA-vs-coldB | margin(top1-top2) at div, cold / hit |
|---|---:|---|---|---|
| S2 19000/s0/tail1000 | 0 | None | None |  |
| S2 19000/s0/tail2000 | 12928 | 41 | None | 0.125 / 0.000 |
| S2 19000/s0/tail4000 | 12928 | 87 | None | 0.125 / 0.125 |
| S2 19000/s1/tail4000 | 0 | 8 | None | 0.375 / 0.000 |
| S2 19000/s1/tail2000 | 12928 | 0 | None | 0.125 / 0.125 |
| S2 19000/s1/tail1000 | 12928 | 21 | None | 0.000 / 0.125 |
| S3 19000/s0/turn2 | 16160 | None | None |  |
| S3 19000/s0/turn3 | 16160 | 26 | None | 0.250 / 0.125 |
| S3 19000/s1/turn2 | 16160 | None | None |  |
| S3 19000/s1/turn3 | 16160 | 122 | None | 0.125 / 0.000 |

## e1-junction: prefix scenarios
| scen | request | prompt tok | hit tok | recomputed | TTFT s | role |
|---|---|---:|---:|---:|---:|---|
| S2 | 19000/s0/seed(cold) | 19068 | 0 | 19068 | 2.744 | seed |
| S2 | 19000/s0/tail1000 | 16069 | 0 | 16069 | 1.907 | hit |
| S2 | 19000/s0/tail1000/cold-a | 16069 | 0 | 16069 | 1.911 | cold |
| S2 | 19000/s0/tail1000/cold-b | 16069 | 0 | 16069 | 1.914 | cold |
| S2 | 19000/s0/tail2000 | 17069 | 12928 | 4141 | 0.565 | hit |
| S2 | 19000/s0/tail2000/cold-a | 17069 | 0 | 17069 | 2.074 | cold |
| S2 | 19000/s0/tail2000/cold-b | 17069 | 0 | 17069 | 2.076 | cold |
| S2 | 19000/s0/tail4000 | 19069 | 12928 | 6141 | 0.824 | hit |
| S2 | 19000/s0/tail4000/cold-a | 19069 | 0 | 19069 | 2.356 | cold |
| S2 | 19000/s0/tail4000/cold-b | 19069 | 0 | 19069 | 2.360 | cold |
| S2 | 19000/s1/seed(cold) | 19067 | 0 | 19067 | 2.364 | seed |
| S2 | 19000/s1/tail4000 | 19066 | 0 | 19066 | 2.350 | hit |
| S2 | 19000/s1/tail4000/cold-a | 19066 | 0 | 19066 | 2.372 | cold |
| S2 | 19000/s1/tail4000/cold-b | 19066 | 0 | 19066 | 2.372 | cold |
| S2 | 19000/s1/tail2000 | 17067 | 12928 | 4139 | 0.571 | hit |
| S2 | 19000/s1/tail2000/cold-a | 17067 | 0 | 17067 | 2.099 | cold |
| S2 | 19000/s1/tail2000/cold-b | 17067 | 0 | 17067 | 2.099 | cold |
| S2 | 19000/s1/tail1000 | 16065 | 12928 | 3137 | 0.417 | hit |
| S2 | 19000/s1/tail1000/cold-a | 16065 | 0 | 16065 | 1.945 | cold |
| S2 | 19000/s1/tail1000/cold-b | 16065 | 0 | 16065 | 1.946 | cold |

### S4 sweep (shared prefix P -> hit tokens; test prompt = P + ~1000 new tokens)
| seed L | P | P mod 1616 | hit | hit/1616 | recomputed | TTFT s |
|---:|---:|---:|---:|---:|---:|---:|
| 19000 | 1 | 1 | 0 | 0.00 | 1001 | 0.107 |
| 19000 | 1615 | 1615 | 0 | 0.00 | 2615 | 0.273 |
| 19000 | 1617 | 1 | 0 | 0.00 | 2617 | 0.274 |
| 19000 | 3232 | 0 | 0 | 0.00 | 4232 | 0.468 |
| 19000 | 4847 | 1615 | 0 | 0.00 | 5847 | 0.669 |
| 19000 | 4849 | 1 | 0 | 0.00 | 5849 | 0.652 |
| 19000 | 6464 | 0 | 0 | 0.00 | 7464 | 0.838 |
| 19000 | 8079 | 1615 | 0 | 0.00 | 9079 | 1.051 |
| 19000 | 8081 | 1 | 0 | 0.00 | 9081 | 1.035 |
| 19000 | 9696 | 0 | 0 | 0.00 | 10696 | 1.240 |
| 19000 | 11311 | 1615 | 0 | 0.00 | 12311 | 1.464 |
| 19000 | 11313 | 1 | 0 | 0.00 | 12313 | 1.445 |
| 19000 | 12928 | 0 | 0 | 0.00 | 13928 | 1.666 |
| 19000 | 14543 | 1615 | 0 | 0.00 | 15543 | 1.898 |

### correctness gate (greedy 256 tokens; hit vs salted-cold)
| request | hit tok | first div hit-vs-coldA | coldA-vs-coldB | margin(top1-top2) at div, cold / hit |
|---|---:|---|---|---|
| S2 19000/s0/tail1000 | 0 | None | None |  |
| S2 19000/s0/tail2000 | 12928 | 147 | None | 0.125 / 0.000 |
| S2 19000/s0/tail4000 | 12928 | 28 | None | 0.000 / 0.125 |
| S2 19000/s1/tail4000 | 0 | 130 | None | 0.000 / 0.375 |
| S2 19000/s1/tail2000 | 12928 | 8 | None | 0.000 / 0.250 |
| S2 19000/s1/tail1000 | 12928 | 113 | None | 0.000 / 0.250 |

## e1-modeall: prefix scenarios
| scen | request | prompt tok | hit tok | recomputed | TTFT s | role |
|---|---|---:|---:|---:|---:|---|
| S2 | 19000/s0/seed(cold) | 19066 | 0 | 19066 | 2.754 | seed |
| S2 | 19000/s0/tail1000 | 16068 | 0 | 16068 | 1.910 | hit |
| S2 | 19000/s0/tail1000/cold-a | 16068 | 0 | 16068 | 1.914 | cold |
| S2 | 19000/s0/tail1000/cold-b | 16068 | 0 | 16068 | 1.918 | cold |
| S2 | 19000/s0/tail2000 | 17064 | 12928 | 4136 | 0.565 | hit |
| S2 | 19000/s0/tail2000/cold-a | 17064 | 0 | 17064 | 2.077 | cold |
| S2 | 19000/s0/tail2000/cold-b | 17064 | 0 | 17064 | 2.079 | cold |
| S2 | 19000/s0/tail4000 | 19067 | 12928 | 6139 | 0.825 | hit |
| S2 | 19000/s0/tail4000/cold-a | 19067 | 0 | 19067 | 2.361 | cold |
| S2 | 19000/s0/tail4000/cold-b | 19067 | 0 | 19067 | 2.365 | cold |
| S2 | 19000/s1/seed(cold) | 19069 | 0 | 19069 | 2.369 | seed |
| S2 | 19000/s1/tail4000 | 19067 | 0 | 19067 | 2.354 | hit |
| S2 | 19000/s1/tail4000/cold-a | 19067 | 0 | 19067 | 2.375 | cold |
| S2 | 19000/s1/tail4000/cold-b | 19067 | 0 | 19067 | 2.378 | cold |
| S2 | 19000/s1/tail2000 | 17069 | 12928 | 4141 | 0.571 | hit |
| S2 | 19000/s1/tail2000/cold-a | 17069 | 0 | 17069 | 2.101 | cold |
| S2 | 19000/s1/tail2000/cold-b | 17069 | 0 | 17069 | 2.101 | cold |
| S2 | 19000/s1/tail1000 | 16071 | 12928 | 3143 | 0.416 | hit |
| S2 | 19000/s1/tail1000/cold-a | 16071 | 0 | 16071 | 1.946 | cold |
| S2 | 19000/s1/tail1000/cold-b | 16071 | 0 | 16071 | 1.948 | cold |

### S4 sweep (shared prefix P -> hit tokens; test prompt = P + ~1000 new tokens)
| seed L | P | P mod 1616 | hit | hit/1616 | recomputed | TTFT s |
|---:|---:|---:|---:|---:|---:|---:|
| 19000 | 1 | 1 | 0 | 0.00 | 1001 | 0.107 |
| 19000 | 1615 | 1615 | 0 | 0.00 | 2615 | 0.275 |
| 19000 | 1617 | 1 | 0 | 0.00 | 2617 | 0.274 |
| 19000 | 3232 | 0 | 0 | 0.00 | 4232 | 0.469 |
| 19000 | 4847 | 1615 | 0 | 0.00 | 5847 | 0.669 |
| 19000 | 4849 | 1 | 0 | 0.00 | 5849 | 0.652 |
| 19000 | 6464 | 0 | 0 | 0.00 | 7464 | 0.838 |
| 19000 | 8079 | 1615 | 0 | 0.00 | 9079 | 1.050 |
| 19000 | 8081 | 1 | 0 | 0.00 | 9081 | 1.037 |
| 19000 | 9696 | 0 | 0 | 0.00 | 10696 | 1.242 |
| 19000 | 11311 | 1615 | 0 | 0.00 | 12311 | 1.466 |
| 19000 | 11313 | 1 | 0 | 0.00 | 12313 | 1.448 |
| 19000 | 12928 | 0 | 0 | 0.00 | 13928 | 1.667 |
| 19000 | 14543 | 1615 | 0 | 0.00 | 15543 | 1.901 |

### correctness gate (greedy 256 tokens; hit vs salted-cold)
| request | hit tok | first div hit-vs-coldA | coldA-vs-coldB | margin(top1-top2) at div, cold / hit |
|---|---:|---|---|---|
| S2 19000/s0/tail1000 | 0 | None | None |  |
| S2 19000/s0/tail2000 | 12928 | 141 | None | 0.250 / 0.000 |
| S2 19000/s0/tail4000 | 12928 | 42 | None | 0.000 / 0.125 |
| S2 19000/s1/tail4000 | 0 | 96 | None | 0.250 / 0.250 |
| S2 19000/s1/tail2000 | 12928 | 28 | None | 0.375 / 0.000 |
| S2 19000/s1/tail1000 | 12928 | 0 | None | 0.000 / 0.125 |

## e1-nomtp: prefix scenarios
| scen | request | prompt tok | hit tok | recomputed | TTFT s | role |
|---|---|---:|---:|---:|---:|---|
| S1 | 19000/s0/cold | 18948 | 0 | 18948 | 2.328 | cold |
| S1 | 19000/s0/repeat1 | 18948 | 18816 | 132 | 0.056 | hit |
| S1 | 19000/s0/repeat2 | 18948 | 18816 | 132 | 0.057 | hit |
| S1 | 19000/s1/cold | 18947 | 0 | 18947 | 2.325 | cold |
| S1 | 19000/s1/repeat1 | 18947 | 18816 | 131 | 0.056 | hit |
| S1 | 19000/s1/repeat2 | 18947 | 18816 | 131 | 0.057 | hit |
| S1 | 44000/s0/cold | 43949 | 0 | 43949 | 6.259 | cold |
| S1 | 44000/s0/repeat1 | 43949 | 43904 | 45 | 0.073 | hit |
| S1 | 44000/s0/repeat2 | 43949 | 43904 | 45 | 0.072 | hit |
| S1 | 44000/s1/cold | 43948 | 0 | 43948 | 6.276 | cold |
| S1 | 44000/s1/repeat1 | 43948 | 43904 | 44 | 0.072 | hit |
| S1 | 44000/s1/repeat2 | 43948 | 43904 | 44 | 0.072 | hit |
| S2 | 19000/s0/seed(cold) | 19070 | 0 | 19070 | 2.350 | seed |
| S2 | 19000/s0/tail1000 | 16072 | 0 | 16072 | 1.946 | hit |
| S2 | 19000/s0/tail1000/cold-a | 16072 | 0 | 16072 | 1.944 | cold |
| S2 | 19000/s0/tail1000/cold-b | 16072 | 0 | 16072 | 1.943 | cold |
| S2 | 19000/s0/tail2000 | 17066 | 14112 | 2954 | 0.415 | hit |
| S2 | 19000/s0/tail2000/cold-a | 17066 | 0 | 17066 | 2.064 | cold |
| S2 | 19000/s0/tail2000/cold-b | 17066 | 0 | 17066 | 2.066 | cold |
| S2 | 19000/s0/tail4000 | 19070 | 14112 | 4958 | 0.672 | hit |
| S2 | 19000/s0/tail4000/cold-a | 19070 | 0 | 19070 | 2.355 | cold |
| S2 | 19000/s0/tail4000/cold-b | 19070 | 0 | 19070 | 2.356 | cold |
| S2 | 19000/s1/seed(cold) | 19070 | 0 | 19070 | 2.356 | seed |
| S2 | 19000/s1/tail4000 | 19067 | 0 | 19067 | 2.332 | hit |
| S2 | 19000/s1/tail4000/cold-a | 19067 | 0 | 19067 | 2.358 | cold |
| S2 | 19000/s1/tail4000/cold-b | 19067 | 0 | 19067 | 2.358 | cold |
| S2 | 19000/s1/tail2000 | 17068 | 14112 | 2956 | 0.416 | hit |
| S2 | 19000/s1/tail2000/cold-a | 17068 | 0 | 17068 | 2.073 | cold |
| S2 | 19000/s1/tail2000/cold-b | 17068 | 0 | 17068 | 2.072 | cold |
| S2 | 19000/s1/tail1000 | 16067 | 14112 | 1955 | 0.295 | hit |
| S2 | 19000/s1/tail1000/cold-a | 16067 | 0 | 16067 | 1.949 | cold |
| S2 | 19000/s1/tail1000/cold-b | 16067 | 0 | 16067 | 1.952 | cold |
| S2 | 44000/s0/seed(cold) | 44071 | 0 | 44071 | 6.327 | seed |
| S2 | 44000/s0/tail1000 | 41069 | 0 | 41069 | 5.834 | hit |
| S2 | 44000/s0/tail1000/cold-a | 41069 | 0 | 41069 | 5.811 | cold |
| S2 | 44000/s0/tail1000/cold-b | 41069 | 0 | 41069 | 5.810 | cold |
| S2 | 44000/s0/tail4000 | 44067 | 39200 | 4867 | 0.900 | hit |
| S2 | 44000/s0/tail4000/cold-a | 44067 | 0 | 44067 | 6.348 | cold |
| S2 | 44000/s0/tail4000/cold-b | 44067 | 0 | 44067 | 6.357 | cold |
| S2 | 44000/s1/seed(cold) | 44066 | 0 | 44066 | 6.359 | seed |
| S2 | 44000/s1/tail4000 | 44070 | 0 | 44070 | 6.359 | hit |
| S2 | 44000/s1/tail4000/cold-a | 44070 | 0 | 44070 | 6.361 | cold |
| S2 | 44000/s1/tail4000/cold-b | 44070 | 0 | 44070 | 6.360 | cold |
| S2 | 44000/s1/tail1000 | 41066 | 39200 | 1866 | 0.396 | hit |
| S2 | 44000/s1/tail1000/cold-a | 41066 | 0 | 41066 | 5.808 | cold |
| S2 | 44000/s1/tail1000/cold-b | 41066 | 0 | 41066 | 5.817 | cold |
| S3 | 19000/s0/turn1 | 18948 | 0 | 18948 | 2.357 | turn1 |
| S3 | 19000/s0/turn2 | 19221 | 18816 | 405 | 0.091 | turn2 |
| S3 | 19000/s0/turn2/cold-a | 19221 | 0 | 19221 | 2.393 | cold |
| S3 | 19000/s0/turn2/cold-b | 19221 | 0 | 19221 | 2.391 | cold |
| S3 | 19000/s0/turn3 | 19474 | 18816 | 658 | 0.133 | turn3 |
| S3 | 19000/s0/turn3/cold-a | 19474 | 0 | 19474 | 2.432 | cold |
| S3 | 19000/s0/turn3/cold-b | 19474 | 0 | 19474 | 2.435 | cold |
| S3 | 19000/s1/turn1 | 18945 | 0 | 18945 | 2.352 | turn1 |
| S3 | 19000/s1/turn2 | 19218 | 18816 | 402 | 0.091 | turn2 |
| S3 | 19000/s1/turn2/cold-a | 19218 | 0 | 19218 | 2.386 | cold |
| S3 | 19000/s1/turn2/cold-b | 19218 | 0 | 19218 | 2.389 | cold |
| S3 | 19000/s1/turn3 | 19473 | 18816 | 657 | 0.133 | turn3 |
| S3 | 19000/s1/turn3/cold-a | 19473 | 0 | 19473 | 2.431 | cold |
| S3 | 19000/s1/turn3/cold-b | 19473 | 0 | 19473 | 2.432 | cold |
| S3 | 44000/s0/turn1 | 43944 | 0 | 43944 | 6.313 | turn1 |
| S3 | 44000/s0/turn2 | 44216 | 43904 | 312 | 0.112 | turn2 |
| S3 | 44000/s0/turn2/cold-a | 44216 | 0 | 44216 | 6.361 | cold |
| S3 | 44000/s0/turn2/cold-b | 44216 | 0 | 44216 | 6.373 | cold |
| S3 | 44000/s0/turn3 | 44468 | 43904 | 564 | 0.186 | turn3 |
| S3 | 44000/s0/turn3/cold-a | 44468 | 0 | 44468 | 6.443 | cold |
| S3 | 44000/s0/turn3/cold-b | 44468 | 0 | 44468 | 6.447 | cold |
| S3 | 44000/s1/turn1 | 43948 | 0 | 43948 | 6.335 | turn1 |
| S3 | 44000/s1/turn2 | 44221 | 43904 | 317 | 0.111 | turn2 |
| S3 | 44000/s1/turn2/cold-a | 44221 | 0 | 44221 | 6.381 | cold |
| S3 | 44000/s1/turn2/cold-b | 44221 | 0 | 44221 | 6.386 | cold |
| S3 | 44000/s1/turn3 | 44474 | 43904 | 570 | 0.186 | turn3 |
| S3 | 44000/s1/turn3/cold-a | 44474 | 0 | 44474 | 6.452 | cold |
| S3 | 44000/s1/turn3/cold-b | 44474 | 0 | 44474 | 6.458 | cold |

### S4 sweep (shared prefix P -> hit tokens; test prompt = P + ~1000 new tokens)
| seed L | P | P mod 1616 | hit | hit/1616 | recomputed | TTFT s |
|---:|---:|---:|---:|---:|---:|---:|
| 19000 | 1 | 1 | 0 | 0.00 | 1001 | 0.107 |
| 19000 | 800 | 800 | 0 | 0.00 | 1800 | 0.207 |
| 19000 | 1615 | 1615 | 0 | 0.00 | 2615 | 0.296 |
| 19000 | 1616 | 0 | 0 | 0.00 | 2616 | 0.295 |
| 19000 | 1617 | 1 | 0 | 0.00 | 2617 | 0.295 |
| 19000 | 3231 | 1615 | 0 | 0.00 | 4231 | 0.466 |
| 19000 | 3232 | 0 | 0 | 0.00 | 4232 | 0.466 |
| 19000 | 3233 | 1 | 0 | 0.00 | 4233 | 0.466 |
| 19000 | 4847 | 1615 | 0 | 0.00 | 5847 | 0.638 |
| 19000 | 4848 | 0 | 0 | 0.00 | 5848 | 0.638 |
| 19000 | 4849 | 1 | 0 | 0.00 | 5849 | 0.638 |
| 19000 | 6463 | 1615 | 0 | 0.00 | 7463 | 0.833 |
| 19000 | 6464 | 0 | 0 | 0.00 | 7464 | 0.833 |
| 19000 | 6465 | 1 | 0 | 0.00 | 7465 | 0.833 |
| 19000 | 8079 | 1615 | 0 | 0.00 | 9079 | 1.029 |
| 19000 | 8080 | 0 | 0 | 0.00 | 9080 | 1.029 |
| 19000 | 8081 | 1 | 0 | 0.00 | 9081 | 1.030 |
| 19000 | 9695 | 1615 | 0 | 0.00 | 10695 | 1.227 |
| 19000 | 9696 | 0 | 0 | 0.00 | 10696 | 1.228 |
| 19000 | 9697 | 1 | 0 | 0.00 | 10697 | 1.228 |
| 19000 | 11311 | 1615 | 0 | 0.00 | 12311 | 1.430 |
| 19000 | 11312 | 0 | 0 | 0.00 | 12312 | 1.429 |
| 19000 | 11313 | 1 | 0 | 0.00 | 12313 | 1.429 |
| 19000 | 12927 | 1615 | 0 | 0.00 | 13927 | 1.636 |
| 19000 | 12928 | 0 | 0 | 0.00 | 13928 | 1.636 |
| 19000 | 12929 | 1 | 0 | 0.00 | 13929 | 1.635 |
| 19000 | 14543 | 1615 | 0 | 0.00 | 15543 | 1.868 |
| 19000 | 14544 | 0 | 0 | 0.00 | 15544 | 1.867 |
| 19000 | 14545 | 1 | 0 | 0.00 | 15545 | 1.869 |
| 19000 | 15000 | 456 | 0 | 0.00 | 16000 | 1.946 |
| 19000 | 16159 | 1615 | 0 | 0.00 | 17159 | 2.091 |
| 19000 | 16160 | 0 | 0 | 0.00 | 17160 | 2.094 |
| 19000 | 16161 | 1 | 0 | 0.00 | 17161 | 2.091 |
| 19000 | 17775 | 1615 | 0 | 0.00 | 18775 | 2.320 |
| 19000 | 17776 | 0 | 0 | 0.00 | 18776 | 2.319 |
| 19000 | 17777 | 1 | 0 | 0.00 | 18777 | 2.321 |
| 19000 | 18000 | 224 | 0 | 0.00 | 19000 | 2.368 |
| 19000 | 18900 | 1124 | 18816 | 11.64 | 1084 | 0.172 |
| 19000 | 19000 | 1224 | 18816 | 11.64 | 1184 | 0.188 |
| 44000 | 1616 | 0 | 0 | 0.00 | 2616 | 0.296 |
| 44000 | 3232 | 0 | 0 | 0.00 | 4232 | 0.467 |
| 44000 | 8080 | 0 | 0 | 0.00 | 9080 | 1.034 |
| 44000 | 15000 | 456 | 0 | 0.00 | 16000 | 1.954 |
| 44000 | 16159 | 1615 | 0 | 0.00 | 17159 | 2.096 |
| 44000 | 16160 | 0 | 0 | 0.00 | 17160 | 2.098 |
| 44000 | 16161 | 1 | 0 | 0.00 | 17161 | 2.099 |
| 44000 | 24239 | 1615 | 0 | 0.00 | 25239 | 3.288 |
| 44000 | 24240 | 0 | 0 | 0.00 | 25240 | 3.291 |
| 44000 | 24241 | 1 | 0 | 0.00 | 25241 | 3.290 |
| 44000 | 30000 | 912 | 0 | 0.00 | 31000 | 4.163 |
| 44000 | 30703 | 1615 | 0 | 0.00 | 31703 | 4.297 |
| 44000 | 30704 | 0 | 0 | 0.00 | 31704 | 4.297 |
| 44000 | 30705 | 1 | 0 | 0.00 | 31705 | 4.300 |
| 44000 | 32319 | 1615 | 0 | 0.00 | 33319 | 4.559 |
| 44000 | 32320 | 0 | 0 | 0.00 | 33320 | 4.562 |
| 44000 | 32321 | 1 | 0 | 0.00 | 33321 | 4.564 |
| 44000 | 33935 | 1615 | 0 | 0.00 | 34935 | 4.842 |
| 44000 | 33936 | 0 | 0 | 0.00 | 34936 | 4.844 |
| 44000 | 33937 | 1 | 0 | 0.00 | 34937 | 4.837 |
| 44000 | 40000 | 1216 | 0 | 0.00 | 41000 | 5.854 |
| 44000 | 40399 | 1615 | 0 | 0.00 | 41399 | 5.945 |
| 44000 | 40400 | 0 | 0 | 0.00 | 41400 | 5.949 |
| 44000 | 40401 | 1 | 0 | 0.00 | 41401 | 5.941 |
| 44000 | 43000 | 984 | 0 | 0.00 | 44000 | 6.393 |
| 44000 | 43631 | 1615 | 0 | 0.00 | 44631 | 6.516 |
| 44000 | 43632 | 0 | 0 | 0.00 | 44632 | 6.514 |
| 44000 | 43633 | 1 | 0 | 0.00 | 44633 | 6.515 |
| 44000 | 44000 | 368 | 43904 | 27.17 | 1096 | 0.243 |

### correctness gate (greedy 256 tokens; hit vs salted-cold)
| request | hit tok | first div hit-vs-coldA | coldA-vs-coldB | margin(top1-top2) at div, cold / hit |
|---|---:|---|---|---|
| S1 19000/s0/repeat1 | 18816 | None | - |  |
| S1 19000/s0/repeat2 | 18816 | None | - |  |
| S1 19000/s1/repeat1 | 18816 | None | - |  |
| S1 19000/s1/repeat2 | 18816 | None | - |  |
| S1 44000/s0/repeat1 | 43904 | None | - |  |
| S1 44000/s0/repeat2 | 43904 | None | - |  |
| S1 44000/s1/repeat1 | 43904 | None | - |  |
| S1 44000/s1/repeat2 | 43904 | None | - |  |
| S2 19000/s0/tail1000 | 0 | 90 | None | 0.000 / 0.125 |
| S2 19000/s0/tail2000 | 14112 | 33 | None | 0.000 / 0.500 |
| S2 19000/s0/tail4000 | 14112 | 8 | None | 0.125 / 0.125 |
| S2 19000/s1/tail4000 | 0 | 19 | None | 0.250 / 0.000 |
| S2 19000/s1/tail2000 | 14112 | 0 | None | 0.500 / 0.000 |
| S2 19000/s1/tail1000 | 14112 | 0 | None | 0.125 / 0.500 |
| S2 44000/s0/tail1000 | 0 | 0 | None | 0.125 / 0.125 |
| S2 44000/s0/tail4000 | 39200 | 0 | None | 0.375 / 0.000 |
| S2 44000/s1/tail4000 | 0 | 0 | None | 0.125 / 0.125 |
| S2 44000/s1/tail1000 | 39200 | 8 | None | 0.500 / 0.000 |
| S3 19000/s0/turn2 | 18816 | None | None |  |
| S3 19000/s0/turn3 | 18816 | None | None |  |
| S3 19000/s1/turn2 | 18816 | None | None |  |
| S3 19000/s1/turn3 | 18816 | None | None |  |
| S3 44000/s0/turn2 | 43904 | None | None |  |
| S3 44000/s0/turn3 | 43904 | None | None |  |
| S3 44000/s1/turn2 | 43904 | None | None |  |
| S3 44000/s1/turn3 | 43904 | None | None |  |

## chunk-structure noise floor (same 6 cold prompts, greedy 256): first divergence index vs e1-base1
- e1-base2: None, None, None, None, None, None
- e1-nomtp: 22, 30, 52, 34, 20, 22
- e1-nopc: 22, 30, 52, 34, 26, 22