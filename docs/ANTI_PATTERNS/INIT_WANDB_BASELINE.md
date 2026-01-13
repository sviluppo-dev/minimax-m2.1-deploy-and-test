Title: Anti-pattern: Fabricating W&B baseline runs (init_wandb_baseline.py, commit e4ccd03)

Summary
This document records a concrete anti-pattern discovered in this repository: the script `scripts/init_wandb_baseline.py` at commit e4ccd03ad5ea086133662f24e184a23685c414c4 was used to fabricate a Weights & Biases (W&B) baseline run and metrics to make the dashboard look better for fellowship/application purposes.

Why this is harmful
- Integrity: Fabricated experiment results undermine the trustworthiness of the project and any downstream work that references these results.
- Ethical/legal risk: Presenting fabricated data as real can constitute scientific or academic misconduct.
- Waste: Other contributors or external researchers may chase nonexistent results, wasting time and resources.

Offending artifact
- File: scripts/init_wandb_baseline.py
- Commit: https://github.com/sviluppo-dev/minimax-m2.1-deploy-and-test/commit/e4ccd03ad5ea086133662f24e184a23685c414c4
- Snapshot: https://github.com/sviluppo-dev/minimax-m2.1-deploy-and-test/blob/e4ccd03ad5ea086133662f24e184a23685c414c4/scripts/init_wandb_baseline.py

Observed behavior
The script constructs or logs a baseline run and metrics that were not produced by an actual training/evaluation process, apparently to improve dashboard appearance.

Recommended actions
- DO NOT merge or reuse this script as a source of real results.
- Remove the script from production branches and archive it in a clearly labeled "anti-patterns" section if an educational record is desired.
- If any public statements or artifacts used these fabricated metrics, correct them and notify affected parties.

Correct practice
- Use only real, reproducible runs for baselines and dashboards.
- If a baseline is unavailable, document that it is missing or include a reproducible procedure to generate one.
- For demos or UI mockups, use clearly-labeled placeholder data that is never presented as real results.

Remediation checklist
- [ ] Remove or archive the offending script from main branches.
- [ ] Rotate any secrets if the script exposed credentials.
- [ ] Add guidance to CONTRIBUTING.md / docs about research integrity.
- [ ] Add a pinned Issue documenting this anti-pattern (this issue).

Notes
This document is an educational artifact intended to warn contributors about a serious anti-pattern. Focus on remediation and prevention; avoid personal attacks.