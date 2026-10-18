"""Reproduce all reported correctness checks and exploratory benchmarks."""
import csv
import hashlib
import json
import platform
from pathlib import Path

from validate import exhaustive, random_checks, benchmark
from structural_checks import main as structural

root = Path(__file__).parent
out = root / 'results'
out.mkdir(exist_ok=True)
for name, result in [('exhaustive', exhaustive(6)), ('random', random_checks(300))]:
    (out / (name + '.json')).write_text(json.dumps(result, indent=2) + '\n')
structural()
rows = benchmark(repetitions=3)
(out / 'benchmark.json').write_text(json.dumps(rows, indent=2) + '\n')
with (out / 'benchmark.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
manifest = {'python': platform.python_version(), 'platform': platform.platform(),
            'files': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                      for name in ['fort_profiles.py', 'validate.py',
                                   'structural_checks.py', 'run_all.py']}}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('All reported checks completed without discrepancies.', flush=True)
