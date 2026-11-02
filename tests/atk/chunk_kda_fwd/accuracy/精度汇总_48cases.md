# chunk_kda_fwd 48 例精度汇总（mixed_tolerance_bm，FP64 golden vs NPU DUT）

- 通过：**48/48**
- 同一用例多份报告时取最新结果。

| idx | id | T | 变体 | 精度 | 结果 | 报告 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 250 | 1024 | packed_single_recompute | ✅ | Pass | 2026-09-29-05-23-49 |
| 1 | 251 | 1024 | packed_single_export | ✅ | Pass | 2026-09-28-19-09-02 |
| 2 | 252 | 1024 | packed_balanced8_recompute | ✅ | Pass | 2026-09-28-19-09-02 |
| 3 | 253 | 1024 | packed_balanced8_export | ✅ | Pass | 2026-09-28-19-09-02 |
| 4 | 254 | 1024 | packed_mixed_tail_recompute | ✅ | Pass | 2026-09-28-19-09-02 |
| 5 | 255 | 1024 | packed_mixed_tail_export | ✅ | Pass | 2026-09-28-15-24-48 |
| 6 | 256 | 1024 | packed_short64_recompute | ✅ | Pass | 2026-09-28-15-24-48 |
| 7 | 257 | 1024 | packed_short64_export | ✅ | Pass | 2026-09-28-18-13-47 |
| 8 | 258 | 1536 | packed_single_recompute | ✅ | Pass | 2026-09-28-15-26-35 |
| 9 | 259 | 1536 | packed_single_export | ✅ | Pass | 2026-09-28-15-26-35 |
| 10 | 260 | 1536 | packed_balanced8_recompute | ✅ | Pass | 2026-09-28-15-26-35 |
| 11 | 261 | 1536 | packed_balanced8_export | ✅ | Pass | 2026-09-28-15-26-35 |
| 12 | 262 | 1536 | packed_mixed_tail_recompute | ✅ | Pass | 2026-09-28-15-26-35 |
| 13 | 263 | 1536 | packed_mixed_tail_export | ✅ | Pass | 2026-09-28-15-26-35 |
| 14 | 264 | 1536 | packed_short64_recompute | ✅ | Pass | 2026-09-28-15-26-35 |
| 15 | 265 | 1536 | packed_short64_export | ✅ | Pass | 2026-09-28-18-14-17 |
| 16 | 266 | 2048 | packed_single_recompute | ✅ | Pass | 2026-09-28-18-12-49 |
| 17 | 267 | 2048 | packed_single_export | ✅ | Pass | 2026-09-28-18-12-49 |
| 18 | 268 | 2048 | packed_balanced8_recompute | ✅ | Pass | 2026-09-28-18-12-49 |
| 19 | 269 | 2048 | packed_balanced8_export | ✅ | Pass | 2026-09-28-18-12-49 |
| 20 | 270 | 2048 | packed_mixed_tail_recompute | ✅ | Pass | 2026-09-28-18-12-49 |
| 21 | 271 | 2048 | packed_mixed_tail_export | ✅ | Pass | 2026-09-28-18-12-49 |
| 22 | 272 | 2048 | packed_short64_recompute | ✅ | Pass | 2026-09-28-18-12-49 |
| 23 | 273 | 2048 | packed_short64_export | ✅ | Pass | 2026-09-28-18-14-48 |
| 24 | 274 | 4096 | packed_single_recompute | ✅ | Pass | 2026-09-28-18-37-22 |
| 25 | 275 | 4096 | packed_single_export | ✅ | Pass | 2026-09-28-18-37-22 |
| 26 | 276 | 4096 | packed_balanced8_recompute | ✅ | Pass | 2026-09-28-18-37-22 |
| 27 | 277 | 4096 | packed_balanced8_export | ✅ | Pass | 2026-09-28-18-37-22 |
| 28 | 278 | 4096 | packed_mixed_tail_recompute | ✅ | Pass | 2026-09-28-18-37-22 |
| 29 | 279 | 4096 | packed_mixed_tail_export | ✅ | Pass | 2026-09-28-18-37-22 |
| 30 | 280 | 4096 | packed_short64_recompute | ✅ | Pass | 2026-09-28-18-37-22 |
| 31 | 281 | 4096 | packed_short64_export | ✅ | Pass | 2026-09-28-18-39-22 |
| 32 | 282 | 8192 | packed_single_recompute | ✅ | Pass | 2026-09-28-22-39-30 |
| 33 | 283 | 8192 | packed_single_export | ✅ | Pass | 2026-09-28-22-40-17 |
| 34 | 284 | 8192 | packed_balanced8_recompute | ✅ | Pass | 2026-09-28-22-41-22 |
| 35 | 285 | 8192 | packed_balanced8_export | ✅ | Pass | 2026-09-28-22-42-08 |
| 36 | 286 | 8192 | packed_mixed_tail_recompute | ✅ | Pass | 2026-09-28-22-43-13 |
| 37 | 287 | 8192 | packed_mixed_tail_export | ✅ | Pass | 2026-09-28-22-43-57 |
| 38 | 288 | 8192 | packed_short64_recompute | ✅ | Pass | 2026-09-28-22-45-04 |
| 39 | 289 | 8192 | packed_short64_export | ✅ | Pass | 2026-09-28-22-45-50 |
| 40 | 290 | 16384 | packed_single_recompute | ✅ | Pass | 2026-09-28-22-46-55 |
| 41 | 291 | 16384 | packed_single_export | ✅ | Pass | 2026-09-28-22-48-06 |
| 42 | 292 | 16384 | packed_balanced8_recompute | ✅ | Pass | 2026-09-28-22-52-21 |
| 43 | 293 | 16384 | packed_balanced8_export | ✅ | Pass | 2026-09-28-22-53-32 |
| 44 | 294 | 16384 | packed_mixed_tail_recompute | ✅ | Pass | 2026-09-28-22-56-37 |
| 45 | 295 | 16384 | packed_mixed_tail_export | ✅ | Pass | 2026-09-28-22-57-46 |
| 46 | 296 | 16384 | packed_short64_recompute | ✅ | Pass | 2026-09-28-23-00-50 |
| 47 | 297 | 16384 | packed_short64_export | ✅ | Pass | 2026-09-29-05-30-14 |
