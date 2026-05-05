```md
# FILE: examples/README.md

# leadopt examples (golden pipelines)

These examples demonstrate reproducible end-to-end workflows using the stable Python API.

## Running examples

From the repo root:

```bash
python examples/medchem_qsar_train_generate.py
python examples/medchem_docking_train_generate.py
python examples/np_fragment_beam.py
python examples/np_fragment_rl_train_generate.py

All scripts:

set explicit seeds,

write outputs under runs/examples/...,

keep training extremely small by default (safe for a smoke run).
Increase TOTAL_UPDATES / EPISODES for real experiments.

Notes

Preset names must exist in your installation.

examples use: medchem_quality_tier4, np_fragment_discovery

Docking example assumes your docking scorer is configured in a preset.

If not available, use it as a template and swap to your docking preset name/path.

If you want fully CLI-equivalent artifacts, set WRITE_ARTIFACTS=True in each script.