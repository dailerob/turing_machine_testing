# M4 Yearly — GDC results

## TL;DR

On M4 yearly (23,000 series, h=6, season=1, median train length 29):

| approach | mean sMAPE | median sMAPE |
|---|---:|---:|
| naive_last | 16.34% | 11.36% |
| drift_recent6 | 16.22% | 8.40% |
| **drift (full series)** | 14.22% | 8.11% |
| **GDC-TS on diffs (L=8, sigma%=0.50, alpha=0.8, absorb)** | **13.93%** | **7.97%** |

~2% relative gain over drift, ~15% over naive_last. M4 yearly
winners are around 13.5-14.5% — we're inside the leaderboard band.

## Notes

- **drift dominates baselines** by a wide margin (14.22% vs naive_last
  16.34%). Yearly series have very strong directional trends.
- **alpha=0.8** is optimal — most aggressive damping of any frequency.
  Short series (median 29 obs) need heavy kernel iteration to denoise.
- **GDC barely beats drift.** With median series length of 29, there
  are very few historical windows to match against; the absorb-mode
  GDC essentially refines the drift estimate slightly.
- **GDC's `drift` fallback for series too short to support a window**
  is what makes the absolute numbers competitive. Many series fall
  back to drift, and drift is already strong.

## Files

- [v0_baselines.py](v0_baselines.py)
- [v1_gdc_diff_sweep.py](v1_gdc_diff_sweep.py) — best result (13.93%)

## Reproduce

```bash
python m4/yearly/v0_baselines.py
python m4/yearly/v1_gdc_diff_sweep.py
```
