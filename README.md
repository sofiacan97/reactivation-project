# Reactivation Project

Model that predicts which inactive customers are likely to be reactivated based on reactevated customers pattern.

## Project structure

```
data/         input data (not included in the repo)
outputs/      trained model and reports (not included in the repo)
src/          data processing, model, scoring and utility code
predict.py    generates predictions on new data
runner.py     runs the full pipeline
release.py    packages the project for release
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Place your input CSV files in `data/raw/`, then run:

```bash
python runner.py
```

To generate predictions only:

```bash
python predict.py
```

Results are saved in `outputs/reports/`.

## Notes

Data files and trained models are excluded from this repository for privacy reasons.