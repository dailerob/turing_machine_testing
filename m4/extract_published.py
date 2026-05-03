"""Parse the M4 supplementary docx and extract per-method per-frequency
Total sMAPE/MASE/OWA values for: top-3 methods + statistical benchmarks.
Then compute series-weighted totals across all 100k series.
"""
from __future__ import annotations
import zipfile, re, os
from collections import defaultdict

DOCX = r'C:\Users\Roberto\Downloads\supplementary_m4_competition.docx'

FREQ_TABLES = {
    'Yearly':    'Table A1',
    'Quarterly': 'Table A2',
    'Monthly':   'Table A3',
    'Weekly':    'Table A4',
    'Daily':     'Table A5',
    'Hourly':    'Table A6',
}
N_SERIES = {'Yearly': 23000, 'Quarterly': 24000, 'Monthly': 48000,
            'Weekly': 359, 'Daily': 4227, 'Hourly': 414}

METHODS = [
    'Smyl', 'Montero-Manso', 'Pawlikowski',
    'ARIMA - Standard',
    'Theta - Benchmark', 'Comb - Benchmark', 'Damped - Benchmark',
    'ETS - Standard', 'Holt - Benchmark', 'SES - Benchmark',
    'Naïve 2 - Benchmark', 'Naïve 1 - Benchmark', 'Naïve S - Benchmark',
]


def get_text():
    z = zipfile.ZipFile(DOCX)
    xml = z.read('word/document.xml').decode('utf-8')
    text = re.sub(r'<[^>]+>', '', xml)
    text = re.sub(r'&amp;', '&', text); text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def parse_row(chunk, name):
    """Find name in chunk, parse the 12 numbers + rank that follow.
    Numbers are formatted as X.XXX (3 decimals each)."""
    m = re.search(re.escape(name), chunk)
    if not m:
        return None
    s = chunk[m.end(): m.end() + 200]
    # Greedy match: extract 12 X.XXX numbers in sequence + integer rank
    # Each number is 1 or 2 digits, dot, exactly 3 digits.
    nums_iter = re.finditer(r'(\d{1,2}\.\d{3})', s)
    vals = []
    for it in nums_iter:
        vals.append(float(it.group(1)))
        if len(vals) == 12:
            break
    if len(vals) < 12:
        return None
    sm_short, sm_med, sm_long, sm_total = vals[0:4]
    ma_short, ma_med, ma_long, ma_total = vals[4:8]
    ow_short, ow_med, ow_long, ow_total = vals[8:12]
    return dict(sMAPE=sm_total, MASE=ma_total, OWA=ow_total)


def main():
    text = get_text()
    table_positions = sorted([(name, text.find(t)) for name, t in FREQ_TABLES.items()],
                             key=lambda x: x[1])
    table_positions.append(('END', text.find('Table B1')))
    table_chunks = {}
    for i in range(len(table_positions) - 1):
        name, p = table_positions[i]
        end = table_positions[i+1][1]
        table_chunks[name] = text[p:end]
    # Parse
    results = defaultdict(dict)
    for freq, chunk in table_chunks.items():
        for method in METHODS:
            r = parse_row(chunk, method)
            if r:
                results[method][freq] = r

    # Compute series-weighted totals
    print(f"{'method':>22s}  "
          f"{'sMAPE':>7s}  {'MASE':>7s}  {'OWA':>7s}  "
          f"   per-freq breakdown (sMAPE/MASE/OWA)")
    rows = []
    for method in METHODS:
        rs = results[method]
        if len(rs) < 6:
            print(f"{method:>22s}  MISSING freqs: {set(FREQ_TABLES.keys()) - set(rs.keys())}")
            continue
        total_n = sum(N_SERIES.values())
        tot_sm = sum(rs[f]['sMAPE'] * N_SERIES[f] for f in N_SERIES) / total_n
        tot_ma = sum(rs[f]['MASE'] * N_SERIES[f] for f in N_SERIES) / total_n
        tot_ow = sum(rs[f]['OWA'] * N_SERIES[f] for f in N_SERIES) / total_n
        rows.append((tot_ow, method, tot_sm, tot_ma, rs))
        bd = ' | '.join(f"{f[:1]}:{rs[f]['OWA']:.3f}" for f in
                        ['Yearly','Quarterly','Monthly','Weekly','Daily','Hourly'])
        print(f"{method:>22s}  {tot_sm:>6.3f}  {tot_ma:>6.3f}  {tot_ow:>6.3f}  {bd}")

    # Save in CSV form for downstream use
    import csv
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'summary', 'published_methods.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'metric'] + list(FREQ_TABLES.keys()) + ['weighted_total'])
        for owa, method, tot_sm, tot_ma, rs in sorted(rows):
            for metric, total in [('sMAPE', tot_sm), ('MASE', tot_ma), ('OWA', owa)]:
                w.writerow([method, metric] +
                           [f"{rs[f][metric]:.3f}" for f in FREQ_TABLES] +
                           [f"{total:.3f}"])
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
