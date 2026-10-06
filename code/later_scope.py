"""Select the manuscript's fixed later-cohort grid before running inference."""
import json
from itertools import product
from pathlib import Path

# The implementation calls the paper's Node MLP replacement 'no_graph'.
PAPER_MODELS = ('full', 'no_graph', 'temporal_only', 'gold_logistic',
                'xgboost', 'full_feature_lr', 'lightgbm')
PAPER_MINUTES = (10, 20, 30, 40)
PAPER_SEEDS = (17, 29, 43)


def select_runs(runs, scope='paper'):
    """Return (selected records, audit). Never load models or create outputs."""
    if scope not in ('paper', 'all'):
        raise ValueError('Unknown evaluation scope: ' + str(scope))
    expected = set(product(PAPER_MODELS, PAPER_MINUTES, PAPER_SEEDS))
    selected, ignored, seen = [], [], set()
    for path in sorted(Path(runs).glob('*/result.json')):
        result = json.loads(path.read_text(encoding='utf-8'))
        key = (result['model'], result['minute'], result['seed'])
        if scope == 'paper' and key[0] not in PAPER_MODELS:
            ignored.append(path.parent.name)
            continue
        if scope == 'paper' and key not in expected:
            raise ValueError('Unexpected paper time/seed: ' + repr(key))
        if key in seen:
            raise ValueError('Duplicate model/time/seed: ' + repr(key))
        seen.add(key)
        weights = [path.parent / name for name in ('model.pt', 'model.joblib')
                   if (path.parent / name).is_file()]
        if len(weights) != 1:
            raise ValueError('Expected exactly one checkpoint: ' + str(path.parent))
        selected.append((path, result, weights[0]))
    if scope == 'paper' and seen != expected:
        missing = sorted(expected - seen)
        raise ValueError(f'Incomplete paper grid: found {len(seen)}/84; '
                         f'missing {len(missing)} combinations: {missing}. '
                         'For an intentional partial/extra evaluation use --scope all '
                         'with a separate output directory.')
    if not selected:
        raise ValueError('No completed runs found')
    return selected, {
        'scope': scope,
        'paper_grid_complete': scope == 'paper' and seen == expected,
        'selected_count': len(selected),
        'ignored_count': len(ignored),
        'ignored_run_directories': ignored,
        'selected': [{'directory': p.parent.name, 'model': r['model'],
                      'minute': r['minute'], 'seed': r['seed']}
                     for p, r, _ in selected],
        'interpretation': ('paper grid: 7 models x 4 times x 3 seeds' if scope == 'paper'
                           else 'all supplied completed runs; separate from paper-only results'),
    }
