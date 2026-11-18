# M18 stage 1 runs (submitted 2026-11-15, snapshot 7ce166c1abd0 on de0f32c dirty, mujou 7a8b72e)

| opt | arm | train (m18-ae-v1) | posthoc (m18-ae-post-v1) |
| -- | -- | -- | -- |
| rmsprop | stream B1 | m18-ae_t128-3023cfa1-s100 | m18-ae_t128-9e63d8a4-s100 |
| rmsprop | shuffle B1 | m18-ae_t128-1e852626-s100 | m18-ae_t128-a831b42b-s100 |
| rmsprop | pool B1 | m18-ae_t128-0d5e96f1-s100 | - |
| rmsprop | stream B32 | m18-ae_t128-cb811245-s100 | m18-ae_t128-75bf7c7b-s100 |
| adamw | stream B1 | m18-ae_t128-7ead07c6-s100 | m18-ae_t128-be6482a5-s100 |
| adamw | shuffle B1 | m18-ae_t128-07f99025-s100 | m18-ae_t128-32bc102d-s100 |
| adamw | pool B1 | m18-ae_t128-82cfb1c7-s100 | - |
| adamw | stream B32 | m18-ae_t128-6c797bfc-s100 | m18-ae_t128-e06b3f9c-s100 |
| sgd | stream B1 | m18-ae_t128_sgd-b64e5b76-s100 | m18-ae_t128_sgd-58cbfc64-s100 |
| sgd | shuffle B1 | m18-ae_t128_sgd-4dadd19a-s100 | m18-ae_t128_sgd-253fb1a5-s100 |
| sgd | pool B1 | m18-ae_t128_sgd-85d36cfd-s100 | - |
| sgd | stream B32 | m18-ae_t128_sgd-6f26a230-s100 | m18-ae_t128_sgd-039df539-s100 |

baselines (m18-ae-base-v1): m18-ae_t128-d7da0ae6-s100
