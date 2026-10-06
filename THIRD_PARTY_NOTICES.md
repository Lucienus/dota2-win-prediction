# Third-party notices

## Bundled hero metadata

code/hero_metadata.json is byte-identical to build/heroes.json in the OpenDota dotaconstants repository at commit b4b5a8299de5f3e0704e62fdd04a6a54c4d4548e.
Copyright (c) 2017 The OpenDota Project. The MIT license text is preserved in third_party_licenses/dotaconstants-MIT.txt. See HERO_METADATA_PROVENANCE.json for source links and verification hashes.
This notice covers the identified repository snapshot. It does not assert ownership of Valve assets or authorize redistribution/relicensing of match records. Image URL strings in the JSON do not include or license the referenced images.

## Runtime dependencies (installed separately)

The release does not redistribute Python dependency packages, compiled runtimes, or CUDA binaries. The following records describe the directly required distributions inspected in the clean validation environment. Their own licenses and bundled third-party notices continue to apply. This list is not an exhaustive license inventory for a future wheel bundle or container.

| Distribution | Version | Installed license evidence |
|---|---|---|
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| scipy | 1.18.1 | BSD-style; see full distribution license |
| scikit-learn | 1.9.1 | BSD-3-Clause |
| joblib | 1.6.0 | BSD-3-Clause |
| xgboost | 3.4.1 | Apache-2.0 |
| lightgbm | 4.7.0 | MIT |
| threadpoolctl | 3.7.0 | BSD-3-Clause |
| torch | 2.11.0+cu128 | BSD-3-Clause |

NumPy, SciPy and PyTorch distributions can contain additional third-party components and notices. Retain and review those notices if redistributing their distributions, rather than treating the top-level license label as an exhaustive list.

## Project code and data

Project-authored code and explicitly listed software configuration/usage documentation are MIT-licensed by ******* ****; see LICENSE_SCOPE.md and CODE_LICENSE_SCOPE.json. This grant does not replace the third-party notices above. Match-derived arrays, predictions, splits and league mappings remain subject to a separate, unresolved redistribution review; see LICENSING_STATUS_ZH.md.
