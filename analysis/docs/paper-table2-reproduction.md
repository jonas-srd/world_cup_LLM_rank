# Reproduce paper Table 2 from the frozen release

[`../scripts/reproduce-paper-table2.py`](../scripts/reproduce-paper-table2.py)
is the standalone runner included in research release
`worldcup2026-2026-07-21-v1`. It recomputes all **35 estimates and 70 confidence
limits** in Table 2 from archived forecasts, without a database, API key, new
LLM call, or search request. The release's original CSV is the input; the live
analytics export and partial deployment seed are not interchangeable with it.

## Download the fixed input

Download [snapshot-original.csv](https://www.llm-soccerarena.com/releases/worldcup2026-2026-07-21-v1/snapshot-original.csv)
and save it as `analysis/inputs/snapshot-original.csv` without opening and
resaving it in a spreadsheet or text editor. Preserve its exact bytes; changes
to line endings, encoding, or a trailing newline will correctly fail the check.
The file is 7,211,402 bytes, with SHA-256:

```text
745c6ff9c38c258204ed4ea4839cd5e1ca4dd0794b18983d6bc7bb3f0f24a681
```

The CSV contains 9,984 match attempts and 480 tournament predictions. Its
tournament rows have a documented missing empty column. The runner validates
the original format and uses only the match rows; use this original CSV rather
than the corrected companion for this exact reproduction.

## Run from the repository root

Use **Python 3.12 and NumPy 2.5.1**. The runner rejects other NumPy versions to
preserve the archived bootstrap results. This small environment is independent
of the full analysis package and does not replace its `uv.lock`.

On Windows (PowerShell):

```powershell
py -3.12 -m venv analysis/inputs/table2-env
analysis/inputs/table2-env/Scripts/python.exe -m pip install -r analysis/requirements-table2.txt
analysis/inputs/table2-env/Scripts/python.exe analysis/scripts/reproduce-paper-table2.py --csv analysis/inputs/snapshot-original.csv --output-dir analysis/artifacts/table2-reproduced
```

On macOS or Linux:

```sh
python3.12 -m venv analysis/inputs/table2-env
analysis/inputs/table2-env/bin/python -m pip install -r analysis/requirements-table2.txt
analysis/inputs/table2-env/bin/python analysis/scripts/reproduce-paper-table2.py --csv analysis/inputs/snapshot-original.csv --output-dir analysis/artifacts/table2-reproduced
```

The input, isolated environment, and generated results are inside paths already
ignored by Git. After downloading the CSV and installing NumPy, execution is
fully offline. Exact results have been verified with CPython 3.12.10 and NumPy
2.5.1 on Windows; other operating systems have not been separately tested.

Expected final output:

```text
PASS: 8,736 evaluated records; 1,456 Table 2 records; all 105 estimates/CI limits match the paper at displayed precision.
```

## What is checked and produced

The script checks the source hash, prediction IDs, the complete 8,736-record
seven-model panel, probability bounds and sums, and recomputed stored metrics.
Table 2 uses 1,456 probabilistic T-24h forecasts: seven models, 104 matches, and
both information-access conditions. Metrics are averaged across the two access
conditions within each match before aggregation across matches. Uncertainty uses
10,000 stratified studentized bootstrap replicates and fixed per-metric seeds.

The output directory contains:

- `table2.csv` and `table2.json`: reproduced estimates, confidence limits, seeds,
  and validation results;
- `evaluated-prediction-ids.txt`: the 8,736-record main panel;
- `table2-prediction-ids.txt`: the exact 1,456 forecasts used for Table 2.

Any integrity, completeness, metric, or reported-value mismatch causes a nonzero
exit. Expected paper values are validation targets; they do not enter the metric
or bootstrap calculations.

## Provenance and full release integrity

The [research release page](https://www.llm-soccerarena.com/data/worldcup2026-2026-07-21-v1)
links the complete archive and its
[versioned manifest](https://www.llm-soccerarena.com/releases/worldcup2026-2026-07-21-v1/manifest.json).
The [full reproduction guide](https://www.llm-soccerarena.com/releases/worldcup2026-2026-07-21-v1/REPRODUCTION.md)
documents the estimand, bootstrap, and historical source references; the
[settings guide](https://www.llm-soccerarena.com/releases/worldcup2026-2026-07-21-v1/SETTINGS.md)
distinguishes recorded metadata from historical defaults and unavailable fields.

To verify the entire downloaded release, extract its ZIP and run this script
with `--verify-checksums PATH_TO_EXTRACTED_RELEASE`. That option requires the
release's `SHA256SUMS` and all inventoried files; it is not a checksum check of
this source-code repository. Checksum verification uses only Python's standard
library.

Stored citation annotations are not complete search execution histories. Reported
completion costs are not an exhaustive retry/repair ledger. Missing metadata
remain missing. These limits are explicit in the release and do not prevent
offline reproduction of Table 2.
