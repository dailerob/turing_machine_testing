# M4 Quarterly — GDC results

## TL;DR

On M4 quarterly (24,000 series, h=8, season=4):

| approach | mean sMAPE | median sMAPE |
|---|---:|---:|
| naive_last | 11.61% | 7.10% |
| drift | 11.43% | 6.24% |
| naive_seasonal4 | 12.52% | 8.06% |
| **GDC-TS on diffs (L=12, sigma%=0.25, alpha=0.9, absorb)** | **10.53%** | **5.69%** |

~8% relative improvement over the best baseline (drift). M4 quarterly
winners are around 9.5-10.5%.

## Notes

- **drift beats naive_last** (unlike daily/monthly). Quarterly series
  have more pronounced multi-year trends.
- **alpha=0.9** is optimal — heavier damping than monthly's alpha=0.95
  or weekly's alpha=0.99. Continues the trend: shorter h-relative
  windows want more kernel iteration.
- **Seasonal naive loses**, same surprise as monthly — quarterly cycles
  are weaker than the underlying drift in most series.

## Files

- [v0_baselines.py](v0_baselines.py)
- [v1_gdc_diff_sweep.py](v1_gdc_diff_sweep.py) — best result (10.53%)

## Reproduce

```bash
python m4/quarterly/v0_baselines.py
python m4/quarterly/v1_gdc_diff_sweep.py
```
