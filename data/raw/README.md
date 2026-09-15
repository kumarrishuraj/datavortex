# data/raw/ — competition dataset (not included in this repository)

This folder is intentionally empty in the public repository.

The Track 1 bundle supplied by the TransOrg AgentIQ Datathon organisers is **not
redistributed** here, for two reasons:

1. The competition asks for code and data dictionaries to be published, not the dataset,
   and its rules prohibit sharing data outside the team.
2. `track1_kyc_records.csv` contains full PAN and Aadhaar values. The data is synthetic,
   but the pipeline is built so that full identifiers never leave this folder — the
   processed layer keeps only masked forms.

## To reproduce the pipeline

Copy the organisers' five files into this folder:

```
data/raw/track1_upi_transactions.csv
data/raw/track1_kyc_records.csv
data/raw/track1_merchants_master.csv
data/raw/track1_chargebacks.json
data/raw/track1_dataset_notes.txt
```

Then, from the repository root:

```bash
python scripts/run_pipeline.py
python scripts/build_analytics.py
```

`run_pipeline.py` asserts the expected raw row counts, so a different or partial bundle fails
loudly rather than silently producing different numbers.

The dashboard does **not** need these files. It reads only the privacy-safe Parquet already
committed under `data/processed/`.
