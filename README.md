# Dota 2 frozen-cohort reconstruction release

Local candidate: reconstruction-2026-10-06. No GitHub or anonymous review URL exists yet.
This version supersedes the code-only public candidate for future release planning.
Read RECONSTRUCTION_ZH.md for full Chinese instructions and limitations.

The release candidate contains model definitions, complete feature-extraction, training, evaluation and table-generation scripts, and recorded environment versions. It also contains frozen historical train/validation/test match IDs, later-cohort inclusion and exclusion records, per-horizon match ordering, raw-file SHA-256 manifests and per-field feature hashes. A downloader retrieves only the listed IDs from OpenDota, logs failures and byte-hash differences, and does not replace missing matches. Original project code is licensed under MIT, with third-party notices retained. Raw match JSON, feature arrays, fitted purchase priors, match-level predictions and trained checkpoints remain local and are not bundled.

The reconstruction entry point fits purchase priors on the frozen historical training partition and checks rebuilt fields against frozen dtype, shape and content hashes. Changed raw bytes require an explicit override and are recorded separately. Hashes detect differences but cannot restore unavailable records; formatting changes can also alter raw-file hashes. Re-fetched data are not assumed equivalent to the original snapshot. Exact reproduction of the reported results requires matching inputs and the relevant execution conditions; a changed-data rerun must be reported separately.

## Quick start (Python 3.12)

```text
python verify_release.py
python -m pip install -r requirements.txt
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m unittest discover -s tests
python code/fetch_frozen_matches.py --manifest manifests/historical.json --output ../raw_historical --limit 1
python code/fetch_frozen_matches.py --manifest manifests/later.json --output ../raw_later --limit 1
```

Remove --limit only when ready to fetch the full cohorts within provider quotas.
OPENDOTA_API_KEY is optional and read from the environment, never logged. HTTP 401,
403 and 429 stop the batch. Existing files are not overwritten. Missing matches
are never replaced. Response bytes may differ due to formatting or data changes.

After all required files have been obtained, use new output directories:

```text
python code/rebuild_frozen.py --cohort historical --raw-dir ../raw_historical --output inputs/historical
python code/rebuild_frozen.py --cohort later --raw-dir ../raw_later --prior inputs/historical/purchase_prior.json --output inputs/later
python run_all.py --output ../new_runs
python code/evaluate_later.py --runs ../new_runs --later inputs/later --output ../new_later_results --scope paper
python code/aggregate_results.py --runs ../new_runs --later-results ../new_later_results --output ../new_tables
```

The strict default stops on raw-file differences. For a deliberate changed-data
experiment, add --allow-changed-raw and use separate outputs. Always inspect
rebuild_status.json and features_equal_frozen. No files are silently dropped.
NPZ container bytes may vary; decoded field checks are authoritative for array equality.
Later auxiliary labels/masks are zero, matching the original evaluation-only inputs.
The original feature extractor and model sources are retained; legacy prepare_raw.py
expects the original richer manifest and is not the entry point for these public manifests.

The recorded environment is Windows/Python 3.12/PyTorch 2.11.0+cu128.
Other environments are not guaranteed bitwise equivalent. This code release does
not bundle data or checkpoints, and is not a completed new 168-fit experiment.
Some tests require withheld original study artifacts and explicitly skip when absent.

## Access

During double-anonymous review, requests for locally retained materials are intended
to go through the editorial office. After publication, contact the corresponding
author. Transfers require assessment of applicable source-data terms; access is not
guaranteed. This named public candidate is not yet an anonymous review copy.

Source API: https://docs.opendota.com/
Journal guidance: https://transactions.games/submit/submission-guidelines
