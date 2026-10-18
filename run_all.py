"""Reproduce checks and measurements; --extended also runs the SciPy baseline."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

from validate import exhaustive, random_checks, benchmark
from structural_checks import main as structural

def main(extended=False):
    root = Path(__file__).parent
    out = root / 'results'
    out.mkdir(exist_ok=True)
    for name, result in [('exhaustive', exhaustive(6)), ('random', random_checks(300))]:
        (out / (name + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    structural()
    if extended:
        from revision_checks import main as revision_checks
        revision_checks()
    rows = benchmark(repetitions=3)
    (out / 'benchmark.json').write_text(json.dumps(rows, indent=2) + '\n')
    with (out / 'benchmark.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if extended:
        from extended_experiments import main as extended_benchmark
        extended_benchmark()
    names = ['fort_profiles.py','validate.py','structural_checks.py','run_all.py',
             'ilp_baseline.py','revision_checks.py','extended_experiments.py']
    manifest = {'python': platform.python_version(), 'platform': platform.platform(),
                'extended': extended,
                'files': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in names}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('All requested checks completed without discrepancies.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--extended', action='store_true',
                        help='also run optional SciPy/HiGHS checks and comparisons')
    main(extended=parser.parse_args().extended)
