# Changelog

All notable changes to the EHRXQA dataset will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-02-06

### Changed
- Migrated to Python 3.12 from Python 3.8.5
- Updated dependencies to latest compatible versions (pandas 3.0.0+, scikit-learn 1.8.0+, dask 2026.1.0+)
- Added UV package manager support with `pyproject.toml` and `uv.lock`
- Restructured README with collapsible sections for better readability
- Added CHANGELOG.md for version tracking

## [1.0.0] - 2024-07-24

### Added
- Released complete EHRXQA dataset on [PhysioNet](https://physionet.org/content/ehrxqa/1.0.0/)
- Official dataset publication with full access for credentialed users

### Changed
- Updated README with PhysioNet dataset link

## [0.1.0] - 2023-11-13

### Added
- Full reproduction scripts for dataset generation
- Integrated database construction (MIMIC-IV + MIMIC-CXR)
- Ground-truth answer generation pipeline
- Complete documentation and usage examples
- Pre-release dataset files (`_train.json`, `_valid.json`, `_test.json`)
- Source dataset download utilities
- Data preprocessing pipeline

## [0.0.1] - 2023-10-28

### Added
- Initial release of research paper on [arXiv](https://arxiv.org/abs/2310.18652)
- Project announcement

[1.1.0]: https://github.com/baeseongsu/ehrxqa/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/baeseongsu/ehrxqa/releases/tag/v1.0.0
[0.1.0]: https://github.com/baeseongsu/ehrxqa/compare/v0.0.1...v1.0.0
[0.0.1]: https://github.com/baeseongsu/ehrxqa/releases/tag/v0.0.1
