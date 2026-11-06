#!/usr/bin/env python3
"""Compare a candidate run with the previous fix, keeping both older baselines."""
import argparse
import csv
import json
import pathlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=pathlib.Path)
    parser.add_argument('--candidate', required=True, type=pathlib.Path)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for group, count in [('matrix', 504), ('adapter', 32), ('ivf', 24)]:
        keys = (['type', 'metric', 'dim', 'record'] if group == 'ivf' else
                ['type', 'metric', 'rows', 'columns', 'dim'])

        def read(directory):
            with (directory / (group + '_comparison.csv')).open() as stream:
                rows = list(csv.DictReader(stream))
            mapped = {tuple(row[k] for k in keys): row for row in rows}
            if len(rows) != count or len(mapped) != count:
                raise ValueError(f'{directory}/{group}: expected {count} unique cases')
            return mapped

        baseline, candidate = read(args.baseline), read(args.candidate)
        if baseline.keys() != candidate.keys():
            raise ValueError(f'{group}: case sets differ')
        output = []
        for key, current in candidate.items():
            previous = baseline[key]
            row = dict(zip(keys, key))
            row['unit'] = 'us/query' if group == 'ivf' else 'ns/matrix'
            for name, value in [('parent', current['parent']),
                                ('before', current['before']),
                                ('previous_fix', previous['fixed']),
                                ('candidate', current['fixed'])]:
                row[name] = float(value)
            for name in ['parent', 'before', 'previous_fix']:
                row['vs_' + name] = row['candidate'] / row[name]
            for name, source in [('previous_fix', previous), ('candidate', current)]:
                row[name + '_min'] = float(source['fixed_min'])
                row[name + '_max'] = float(source['fixed_max'])
            row['parent_drift'] = float(current['parent']) / float(previous['parent'])
            row['before_drift'] = float(current['before']) / float(previous['before'])
            output.append(row)
        with (args.output / (group + '_four_versions.csv')).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
        summary[group] = {
            'cases': count,
            'candidate_over_previous_fix_5pct': [r for r in output if r['vs_previous_fix'] > 1.05],
            'candidate_over_before_5pct': [r for r in output if r['vs_before'] > 1.05],
            'candidate_over_parent_5pct': [r for r in output if r['vs_parent'] > 1.05],
            'baseline_drift_over_5pct': [r for r in output if
                abs(r['parent_drift'] - 1) > .05 or abs(r['before_drift'] - 1) > .05],
        }
        print(group, {k: len(v) if isinstance(v, list) else v
                      for k, v in summary[group].items()})
    (args.output / 'four_version_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    print('Ratios above 1 are slower. This is a review report, not an automatic pass.')


if __name__ == '__main__':
    main()
