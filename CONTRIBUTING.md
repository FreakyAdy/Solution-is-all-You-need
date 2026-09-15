# Contributing to PHANTOM

Thank you for your interest in contributing to PHANTOM. We welcome contributions that improve inference efficiency, expand hardware support, and enhance mathematical rigor.

---

## 1. Ground Truth & Anti-Regression Invariants

PHANTOM operates under strict empirical ground truth invariants:

1. **No claim without a run**: Never introduce or alter a performance figure in documentation without executing the corresponding benchmark script. Every published number must trace to `benchmarks/results/latest.json`.
2. **Never mark something PASS without execution**: If a test cannot run (e.g. missing GPU hardware), label it `SKIPPED`, not `PASS`.
3. **Zero-Disk model storage policy**: Never commit raw model weights to the repository. Use `tests/ephemeral_test_runner.py` or `phantom plan` for zero-disk verification.
4. **Mandatory PR Gates**:
   - `python tests/correctness/test_reference_parity.py --quick` (Verifies numerical soundess, Top-1 agreement > 99%)
   - `python scripts/check_claims.py` (Ensures 0 unverified or drifting numbers across documentation)
   - `python scripts/generate_results.py --check` (Confirms `RESULTS.md` matches `latest.json`)

---

## 2. Local Development Workflow

### Setting up the Environment

```bash
git clone https://github.com/FreakyAdy/phantom.git
cd phantom
pip install -e python/
```

### Running the Pre-Commit Test Gate

Before submitting a Pull Request, run the local verification suite:

```bash
# 1. Run quick numerical parity gate
python tests/correctness/test_reference_parity.py --quick

# 2. Run documentation claims consistency gate
python scripts/check_claims.py

# 3. Verify generated results ledger
python scripts/generate_results.py --check
```

If your PR introduces a new hardware constant, documented parameter scale, or measured memory allocation, add an entry with description and owner in [`docs/claims_allowlist.yml`](docs/claims_allowlist.yml).

---

## 3. Pull Request Guidelines

- **Focus**: Keep Pull Requests small, atomic, and focused on a single subsystem.
- **Documentation**: If any performance or memory behavior is changed, regenerate `RESULTS.md` via `python scripts/generate_results.py` and commit the updated `benchmarks/results/latest.json`.
- **Commit Messages**: Follow conventional commit conventions (`feat:`, `fix:`, `docs:`, `perf:`, `refactor:`, `test:`).
