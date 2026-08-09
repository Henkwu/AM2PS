# Validation report

Date: 2026-08-08

The reconstructed repository was checked in the local execution environment before packaging.

## Checks completed

1. `python -m compileall -q .` — PASS
2. `python tests/smoke_test.py` — PASS
   - Instantiated the complete AM2PS fusion graph with a dummy frozen CLIP backend.
   - Verified forward pass, probability normalization, and backward propagation.
3. Synthetic end-to-end training — PASS
   - Constructed a temporary ImageFolder-style binary dataset.
   - Ran `train.py` for one epoch on CPU using the dummy CLIP backend.
   - Saved `best.pt`, configuration metadata, training summary, and test metrics.
   - This test validates the training/evaluation pipeline only; it is not a scientific result.

## Not executed here

Full ChestXRay2017 and ChestXRay-Covid19 training was not executed because the datasets and pretrained CLIP weights were not supplied in this conversation. Consequently, the repository does not claim to have regenerated the paper's 95.65% or 97.21% accuracies.

## Reproduction status

The package is suitable for controlled reproduction once the datasets are placed in the documented directory structure. Exact numerical reproduction remains conditional on resolving manuscript-omitted settings and the formula ambiguities documented in `PAPER_IMPLEMENTATION_NOTES.md`.
