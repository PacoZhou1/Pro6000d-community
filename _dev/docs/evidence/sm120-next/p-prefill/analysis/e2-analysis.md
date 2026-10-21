| length | e2-fi-a1 median s (min-max) | e2-flash-attn median s (min-max) | e2-triton median s (min-max) | e2-fi-a2 median s (min-max) | prompt tokens | hit tokens |
|---|---|---|---|---|---|---|
| 18900 | 2.257 (2.248-2.266, n=4) | n/a | 2.646 (2.632-2.658, n=4) | 2.264 (2.259-2.270, n=3) | 18838 | 0 |
| 44000 | 6.349 (6.299-6.382, n=4) | n/a | 8.503 (8.442-8.551, n=4) | 6.350 (6.313-6.369, n=3) | 43933 | 0 |
| 64000 | 10.587 (10.533-10.619, n=4) | n/a | 15.129 (15.072-15.173, n=4) | 10.564 (10.521-10.604, n=3) | 63937 | 0 |
| 128000 | 28.720 (28.675-28.737, n=4) | n/a | 47.019 (46.923-47.083, n=4) | 28.674 (28.643-28.704, n=2) | 127937 | 0 |

paired change vs mean(A1,A2) per index (positive = slower than FlashInfer):
- L=18900: A1->A2 drift median +0.449% (n=3)
  - e2-triton: median +16.929% range +16.829..+16.988
- L=44000: A1->A2 drift median +0.163% (n=3)
  - e2-triton: median +33.835% range +33.792..+33.874
- L=64000: A1->A2 drift median -0.075% (n=3)
  - e2-triton: median +43.027% range +42.824..+43.174
- L=128000: A1->A2 drift median -0.070% (n=2)
  - e2-triton: median +63.719% range +63.712..+63.726

D3 correctness (16 requests, 1 pass, greedy) vs e2-fi-a1: first divergence token index (None = identical)
- e2-flash-attn: no d3
- e2-triton: identical 2/16; decode#1:80; decode#2:25; decode#3:15; decode#4:109; decode#5:16; decode#6:36; thinking#1:19; thinking#2:72; thinking#3:90; thinking#4:216; thinking#5:105; thinking#6:110; open_thinking_arithmetic#1:74; open_thinking_arithmetic#2:74
- e2-fi-a2: identical 16/16; 