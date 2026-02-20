# HOWZAT DD Fix Journal

## Objective
Diagnose why exact DD modes (`howzat-dd@int:gmprat` and `howzat-dd@sp:gmprat`) could disagree with `ppl+hlbl`, fix the core logic bug, and recover performance as close as possible to PPL while preserving exact parity.

Primary repro:
- `data/motif_early/arcadian-accordion__hit_000001_w6_ex+1_sp2_seed6082620689740043133.json`
- Expected width (`ppl+hlbl`): `5`
- Buggy pre-fix width (`howzat-dd@int:gmprat`): `6`

## Core correctness bug (exact modes should never be wrong)
The bug was not numeric. It was DD combinatorial logic in shared core code.

Root cause:
- Adjacency was evaluated on raw parent zero-set intersections, not on the face restricted to `AddedHalfspaces`.
- In generator mode, SAT row IDs may exist for rows outside the currently added halfspace domain, so using the unrestricted face can produce wrong exact adjacency decisions.

Why both exact pathways failed:
- `@int` and `@sp` use different numeric umpires, but both route through the same DD adjacency core.

Implemented correctness fix:
- Force adjacency quick/full checks to operate on `face = (zero(r1) ∩ zero(r2)) ∩ AddedHalfspaces`.
- Keep candidate witness checks in the same domain.
- Added regression test: `dd::tests::adjacency_uses_added_halfspaces_face`.

## PPL/cddlib comparison
Reviewed vendored sources under `target/build-deps`:
- PPL: `target/build-deps/ppl-sys/.../PPL-.../src/Polyhedron_conversion_templates.hh` (adjacency loop around ~746-838).
- cddlib: `target/build-deps/cddlib-sys/.../lib-src/cddcore.c` (`dd_CheckAdjacency`, `dd_AddNewHalfspace1`).

Shared semantic behavior:
- Adjacency decisions are taken in the currently treated-constraint domain.
- Full witness test checks whether another generator saturates the same face.

## Why performance regressed after correctness fix
With correct AddedHalfspaces masking, more pairs are admissible than in the buggy path, so more rays are generated.
This effect is real (semantic), not a random slowdown.

Measured on 137-file corpus (`/tmp/howzat_motif_files.list`) in the corrected build before `@sp` optimization:
- `ppl+hlbl`: ~0.89s
- `howzat-dd@int:gmprat`: ~0.99s
- `howzat-dd@sp:gmprat`: ~1.69s
- `howzat-dd[purify[snap]]:f64[eps[1e-12]]`: ~0.83s

Key observation from instrumentation run (same corrected semantics):
- `@int` and `@sp` had identical DD-level operation counts (`classify`, adjacency checks, created rays, etc.).
- Therefore the large `@sp` gap was per-operation cost, not extra combinatorial work.

## Performance bottlenecks and fixes
### Bottleneck A (`@sp` per-ray numeric work)
`SinglePrecisionUmpire::generate_new_ray` was reclassifying each new ray from scratch across all rows (`classify_vector`), even when parent sign information was already available.

Fix A in `howzat/src/dd/umpire/single_precision.rs`:
- Build inherited zero set: `zero(r1) ∩ zero(r2)` plus current row.
- Build inherited row-sign cache from parent row signs using `combine_nonnegative_signs`.
- Seed the new-ray classifier with this inherited information (`build_ray_from_vector(..., seeded_zero_set=true, preseeded_row_signs=...)`).
- Keep purifier-safe behavior:
  - if purification changes vector, drop inherited sign cache and fall back to seeded-zero-set classification to preserve correctness.

This removed a large amount of repeated dot-product work on `@sp`.

### Bottleneck B (DD core adjacency fast-path divergence from PPL)
After reviewing `Polyhedron_conversion_templates.hh` carefully, two extra divergences remained in our shared DD core:

1. Quick-adjacency predicate too strict.
   - PPL quick-adjacency condition is equivalent to:
     - `min(parent_saturators) == common_saturators + 1`
   - Our code required:
     - `common_saturators == required` **and** `min(parent_saturators) == required + 1`
   - This weaker predicate skipped many safe quick-adjacency accepts and forced more full witness scans.

2. Added-halfspace saturation mask lagging one iteration.
   - We assigned SAT id for current row before pair generation but inserted current row into `added_halfspaces_zero` only after pair generation.
   - This kept `added_domain_full` false in the hot loop and triggered slower masked cardinality paths repeatedly.

Fix B in `howzat/src/dd/engine.rs`:
- Quick-adjacency now matches PPL logic: accept when `min_parent_saturators == common + 1`.
- Insert current row into `added_halfspaces_zero` before pair generation.

## Validation
### Tests
- `cargo test -p howzat dd::tests -- --nocapture` passes.
- `cargo test -p howzat single_precision::tests -- --nocapture` passes.

### Parity (137-file corpus)
Command shape used:
- `target/release/hirsch --noninteractive sandbox stats --quiet --backend <spec> --input <file>`

Result:
- `howzat-dd@int:gmprat`: 0 mismatches vs `ppl+hlbl`
- `howzat-dd@sp:gmprat`: 0 mismatches vs `ppl+hlbl`
- `howzat-dd[purify[snap]]:f64[eps[1e-12]]`: 0 mismatches vs `ppl+hlbl`

## Final performance (clean build, latest 5-run medians)
Measured on Feb 20, 2026 with full 137-file sweep per run (`/tmp/howzat_motif_files.list`).

- `ppl+hlbl`: `0.87s`
- `howzat-dd@int:gmprat`: `0.90s`
- `howzat-dd@sp:gmprat`: `1.13s`
- `howzat-dd[purify[snap]]:f64[eps[1e-12]]`: `0.78s`

Per-file ratio scan against `ppl+hlbl`:
- `@int`: worst observed ratio ~`1.38x` (most files close to `1.0x..1.15x`)
- `@sp`: worst observed ratio ~`1.70x`

Net result versus post-correctness baseline:
- `@int`: recovered from roughly `~0.97s` to `~0.90s`
- `@sp`: recovered from roughly `~1.21s` to `~1.13s`

## Notes
- I attempted kernel `perf` sampling for one last hotspot pass, but this host has `perf_event_paranoid=4`, so hardware profiling is blocked for non-privileged runs.
- Temporary DD instrumentation used during diagnosis was removed from hot paths in the final code.

## Final diagnosis summary
1. Exact errors came from adjacency face-domain semantics, not numeric instability.
2. The bug affected both exact umpires because the shared DD core was wrong.
3. Correct semantics necessarily increase some combinatorial work vs the buggy path.
4. The remaining avoidable slowdown versus PPL came from DD core quick-adjacency handling; matching PPL's predicate and mask timing removed extra full witness scans.

## Additional pass (PPL-oriented adjacency optimization)
After the previous fixes, I re-instrumented `add_new_halfspace_dynamic` to isolate hot sections on the same 137-file corpus:
- Dynamic DD halfspace processing was adjacency-heavy.
- Pair workload was very skewed (`~1.85M` checked pairs, `~53k` accepted), i.e. most pairs fail adjacency early.

Optimization applied in `howzat/src/dd/engine.rs`:
- In `check_adjacency_common_with_candidates`, for the common `added_domain_full` path:
  - compute `common` via count-only intersection first (`count_intersection`);
  - return early for quick non-adj / quick-adj outcomes;
  - materialize `adj_face` only when the full witness test is actually needed.
- Kept PPL-equivalent quick-adj predicate (`min_parent == common + 1`).

This avoids unnecessary `adj_face` materialization for the vast majority of rejected pairs.

Important safety fix while optimizing:
- Candidate bitset capacity can grow as new rays are inserted mid-loop.
- The resize guard in the pair loop is required; removing it causes `IdSet::intersection_inplace requires normalized storage` panics.
- Final code keeps that guard.

Additional diagnostic:
- Temporarily disabling quick-adjacency (`min_parent == common + 1`) did **not** remove the known edge-list mismatch set vs `ppl+hlbl` in `sandbox bench --mode adjacency`.
- That mismatch is therefore not caused by the quick-adj shortcut itself.

### Latest revalidation
- `cargo test -p howzat dd::tests::adjacency_uses_added_halfspaces_face -- --nocapture` passes.
- `cargo test -p howzat single_precision::tests -- --nocapture` passes.
- Corpus parity (`/tmp/howzat_motif_files.list`, 137 files):
  - `howzat-dd@int:gmprat`: `0` mismatches vs `ppl+hlbl`
  - `howzat-dd@sp:gmprat`: `0` mismatches vs `ppl+hlbl`

### Latest performance snapshot (5-run medians, same corpus)
- `ppl+hlbl`: `0.84s`
- `howzat-dd@int:gmprat`: `0.94s` (~`1.12x`)
- `howzat-dd@sp:gmprat`: `1.17s` (~`1.39x`)

Compared to the immediately previous post-fix baseline on this machine, this recovered additional runtime in both exact pathways.

## Deep dive: remaining regression and edge mismatches (final pass)

### Finding 1: `sandbox bench --mode adjacency` mismatches were not DD-core errors
I re-ran deterministic adjacency bench on the same 137-file motif corpus and confirmed the persistent `mismatches=25` pattern for both exact howzat DD pathways while incidence mode had `mismatches=0`.

That isolated the issue to **vertex-adjacency extraction**, not the DD conversion core:
- `ppl+hlbl` builds vertex adjacency via `adjacency_from_incidence_with(..., adj_dim, ...)`.
- `howzat-dd` geometry extraction in `howzat-kit/src/backend/howzat_common.rs` was using `input_adjacency_from_rows_by_node_with(...)`.
- That helper is for input-drums semantics and internally uses a fixed `adj_dim=3` plus special-case edge injections, which is not equivalent to the PPL incidence-adjacency criterion used for benchmark comparison.

Fix:
- Switched howzat-dd vertex adjacency extraction to `adjacency_from_incidence_with` in both representation branches, with the real `adj_dim` (`inequality_matrix.col_count()`) and explicit excluded-node mask where needed.

Result:
- `sandbox bench --mode adjacency` now reports `mismatches=0` for both `howzat-dd@int:gmprat` and `howzat-dd@sp:gmprat` on the 137-file corpus.

### Finding 2: masked AddedHalfspaces path was not the runtime bottleneck
I instrumented DD pair accounting and confirmed:
- `masked_pairs=0` across the corpus in normal operation.

So the residual slowdown was not from repeatedly taking the masked intersection branch; it was elsewhere.

### Finding 3: hot-loop cleanup in exact umpire ray generation
I then optimized parent-sign reuse to avoid a redundant full-row prepass in both:
- `howzat/src/dd/umpire/int.rs`
- `howzat/src/dd/umpire/single_precision.rs`

Change:
- Moved parent-sign hint combination into the single `build_ray_from_vector` row loop via `parent_sign_hints`, removing an extra full `order_vector` pass during `generate_new_ray`.

This reduces per-ray overhead without changing semantics.

## Final revalidation (this pass)

### Correctness
- `cargo test -p howzat dd::tests -- --nocapture` passes.
- `cargo test -p howzat single_precision::tests -- --nocapture` passes.
- Width parity on `/tmp/howzat_motif_files.list` (137 files):
  - `howzat-dd@int:gmprat`: `0` mismatches vs `ppl+hlbl`
  - `howzat-dd@sp:gmprat`: `0` mismatches vs `ppl+hlbl`
- Adjacency bench parity (`sandbox bench --mode adjacency`) on the same 137-file deterministic corpus:
  - `mismatches=0` for both `howzat-dd@int:gmprat` and `howzat-dd@sp:gmprat`.

### Performance snapshot (same deterministic 137-file corpus, noisy single-run snapshots)
- `--mode incidence`:
  - `ppl+hlbl`: avg ~`1.30ms`
  - `howzat-dd@int:gmprat`: avg ~`1.59ms` (`~1.22x`)
  - `howzat-dd@sp:gmprat`: avg ~`3.45ms` (`~2.65x`)
- `--mode adjacency`:
  - `ppl+hlbl`: avg ~`1.51ms`
  - `howzat-dd@int:gmprat`: avg ~`1.85ms` (`~1.23x`)
  - `howzat-dd@sp:gmprat`: avg ~`3.73ms` (`~2.47x`)

Notes:
- There is measurable run-to-run noise on this host, but ratios stayed in the same band after the final fixes.
- The previous adjacency mismatch set is fully resolved by the extraction-path correction.

## Final pass (PPL-equivalent perf focus, rigorous A/B)

### What I re-checked
- Re-ran deterministic parity/perf bench on `file:/tmp/howzat_motif_bench_137` for:
  - `ppl+hlbl:gmpint`
  - `howzat-dd@int:gmprat`
  - `howzat-dd@sp:gmprat`
- Re-ran exact repro width check on:
  - `data/motif_early/arcadian-accordion__hit_000001_w6_ex+1_sp2_seed6082620689740043133.json`
- Re-ran motif fixed harness in `kompute-hirsch/scripts/bench_motif_fixed.sh` with fixed seed.

### Core finding: remaining DD gap vs PPL came from full-test structure
PPL full adjacency witness test scans candidate generators and checks subset on saturation rows.
Our DD core had moved to incidence-index intersections in a way that kept extra overhead in SatRepr exact pathways.

Applied optimization:
- Added a SatRepr-specific branch (`ZeroRepr::USE_INCIDENCE_INDEX_FOR_CANDIDATE_TEST = false`) so degeneracy full witness checks use direct subset tests against candidate rays.
- Kept incidence index for RowRepr path.
- For SatRepr:
  - stop maintaining incidence index in register/unregister/clear paths;
  - skip bitset-candidate fast-path tied to incidence capacity;
  - use PPL-like candidate subset scan in `check_adjacency_common_with_candidates`.
- Added cheap prefilter in subset scan:
  - skip candidates where `zero_set_count < face_cardinality`.

Files:
- `howzat/src/dd/engine.rs`
- `howzat/src/dd/index.rs`
- `howzat/src/dd/zero.rs`
- `howzat/src/dd/sat.rs`

### AddedHalfspaces correctness remains the root fix
Preserved the semantic fix:
- adjacency face is computed in AddedHalfspaces domain (masked face), not unrestricted zero-set intersection.
- This is why exact int and exact sp previously both failed identically: shared core logic bug.

Files:
- `howzat/src/dd/engine.rs`
- `howzat/src/dd/state.rs`
- regression test in `howzat/src/dd/tests.rs` (`adjacency_uses_added_halfspaces_face`).

### Critical stability check: singular pivot panic
During motif fixed harness A/B, I observed oracle errors caused by:
- `panic: exact basis computation failed (singular pivot submatrix)`

This was due to local drift in `int.rs` temporarily undoing the previously committed safety behavior.
I restored `howzat/src/dd/umpire/int.rs` to committed `b64c119` state so singular pivots no longer panic.

Validation (same seed/options):
- `intcheck_short` (200k checks): `errors_total=0`
- `intcheck_mid` (500k checks): `errors_total=0`

### SP umpire A/B and minimization decision
I tested reverting SP umpire local changes to minimize tree size.
Result was a severe regression:
- `howzat-dd@sp:gmprat` degraded to about `~5.3x..5.9x` vs PPL on the 137-file corpus.
Therefore I restored SP umpire changes (`single_precision.rs`, `umpire/mod.rs`) as required for performance.

### Final performance snapshot (latest deterministic corpus run)
Adjacency mode (`/tmp/howzat_motif_bench_137`):
- `ppl+hlbl:gmpint`: avg `1.475ms`
- `howzat-dd@int:gmprat`: avg `1.750ms` (`1.19x`)
- `howzat-dd@sp:gmprat`: avg `3.480ms` (`2.36x`)

Incidence mode (`/tmp/howzat_motif_bench_137`):
- `ppl+hlbl:gmpint`: avg `1.137ms`
- `howzat-dd@int:gmprat`: avg `1.341ms` (`1.18x`)
- `howzat-dd@sp:gmprat`: avg `2.886ms` (`2.54x`)

### Motif fixed harness cross-check (real workload path)
With `checks=200000`, `timeout_ms=5`, `seed=424242`:
- PPL: `attempted_per_s=6056.94`, `errors_total=0`
- howzat int: `attempted_per_s=5879.07`, `errors_total=0`

So on this workload path, int throughput is within a few percent of PPL and no longer exhibits singular-pivot crash errors.

### Final diagnosis
1. Exact wrong width was an implementation bug in shared DD adjacency domain semantics, not numeric instability.
2. The same bug in both exact int and exact sp is expected because both use the same adjacency core.
3. Post-correctness perf regression source was divergence from PPL full-test structure in SatRepr pathways.
4. PPL-equivalent strategy for this hotspot is candidate subset scan on saturation rows; implementing that path recovered most of the exact-int gap while keeping the correctness fix.

## Rigorous slowdown root-cause pass (PPL-side instrumentation + iterative A/B)

### Instrumentation setup
- Rebuilt `ppl-sys` against patched PPL conversion source and captured `PPL_DD_PROFILE` counters on the deterministic 137-file corpus.
- Added gated howzat DD profiling (`HOWZAT_DD_PROFILE`) in `howzat/src/dd/engine.rs` + `howzat/src/dd/state.rs`:
  - counts: `pairs`, quick/nonadj/full/new-ray, dedup drops
  - timings: `full_ns`, `create_ns`
- Kept hot-loop counters disabled unless `HOWZAT_DD_TRACE`/`HOWZAT_DD_PROFILE` is set.

### Measured divergence (before final tweak)
On `/tmp/howzat_motif_bench_137` adjacency mode:
- PPL:
  - `pairs=1,644,186`
  - `full_tests=6,128`
  - `created=50,304`
  - `full_ns=408,819`
  - `create_ns=64,866,581`
- howzat int:
  - `pairs=1,852,144` (~+12.6%)
  - `full_tests=6,123` (near-identical)
  - `new_rays=53,459` (~+6.3%)
  - `full_ns=1,015,877`
  - `create_ns=120,627,331`

Key conclusion:
- Remaining gap is dominated by **ray creation path** cost (`create_ns`), not by full witness checks.
- Full-test count is already close to PPL; pair/ray workload is somewhat higher in howzat, but the largest per-op gap is `create_new_ray` pipeline.

### A/B attempts and outcomes
1. Quick-adjacency predicate tightening (`common==required && ...`):
- Rejected: no pair reduction; just moved accepts from quick path to full tests, and slowed runtime.

2. Row-order policy swap to `MinIndex`:
- Rejected: introduced large adjacency/facet mismatch set against baseline.

3. Aggressive lazy inherited-ray construction (skip full classification):
- Rejected: broke output quality (e.g. repro facet/ridge counts wrong) and failed `sandbox stats` path.

4. Safe int-only early-exit in new-ray classification:
- Accepted:
  - In `IntUmpire::build_ray_from_vector`, for SatRepr/non-cutoff path only, stop per-row classification once first infeasible future row is found.
  - This keeps exact behavior and parity while reducing unnecessary per-new-ray work.

Files:
- `howzat/src/dd/umpire/int.rs`
- `howzat/src/dd/engine.rs`
- `howzat/src/dd/state.rs`

### Current validation snapshot
- Repro width file (`arcadian-accordion...`):
  - `ppl+hlbl:gmpint = 5`
  - `howzat-dd@int:gmprat = 5`
  - `howzat-dd@sp:gmprat = 5`
- Tests:
  - `cargo test -p howzat dd::tests::adjacency_uses_added_halfspaces_face -- --nocapture` pass
  - `cargo test -p howzat single_precision::tests -- --nocapture` pass
- 137-file deterministic parity bench:
  - `mismatches=0` for both `@int` and `@sp`.

### Current performance snapshot (latest run)
Adjacency mode (`/tmp/howzat_motif_bench_137`):
- `ppl+hlbl:gmpint`: avg `1.541ms`
- `howzat-dd@int:gmprat`: avg `1.757ms` (`1.14x`)
- `howzat-dd@sp:gmprat`: avg `3.538ms` (`2.30x`)

Incidence mode (same corpus):
- `ppl+hlbl:gmpint`: avg `1.276ms`
- `howzat-dd@int:gmprat`: avg `1.514ms` (`1.19x`)
- `howzat-dd@sp:gmprat`: avg `3.343ms` (`2.62x`)

### State of diagnosis
- Root correctness issue remains the AddedHalfspaces-domain adjacency bug in shared DD core (already fixed).
- Remaining perf delta is now quantified:
  - moderate extra pair/ray workload vs PPL on this corpus,
  - plus higher per-new-ray cost in howzat, with `create_new_ray` still the primary hotspot.
- Latest accepted tweak reduces the exact-int adjacency gap further without regressions.

## Final optimization pass: equality-mask fast path + int create-ray tightening

### Why this pass
Even after the core correctness fix, remaining exact-int slowdown vs PPL was still concentrated in DD ray creation cost. I targeted only minimal changes that preserved correctness/parity and reduced hot-path work.

### Change 1: make AddedHalfspaces mask include equality rows
File:
- `howzat/src/dd/engine.rs`

What changed:
- `rebuild_added_halfspaces_zero()` now seeds `added_halfspaces_zero` from both:
  - `equality_set`
  - `added_halfspaces`

Why:
- Equality rows already have Sat IDs and are saturated by all rays.
- Excluding them forced the masked-intersection path more often than necessary in adjacency checks.
- Including them keeps the same adjacency semantics while restoring more of the unmasked fast path behavior.

### Change 2: remove int umpire inherited-sign prepass
File:
- `howzat/src/dd/umpire/int.rs`

What changed:
- `generate_new_ray()` no longer does a full `order_vector` prepass to build inherited signs.
- Parent-sign inference is now folded into the existing `build_ray_from_vector()` row loop via `parent_sign_hints`.
- Classification now checks `seeded_zero_set` early (before parent-hint fallback), skipping extra hint work when zero is already known.

Why:
- This removed one full pass over rows per created ray in exact-int mode.

### Change 3: SatRepr/non-cutoff prefix skip in int classification
File:
- `howzat/src/dd/umpire/int.rs`

What changed:
- In SatRepr + non-cutoff + generated-ray path (`parent_sign_hints.is_some()`), `build_ray_from_vector()` begins scanning at `last_row` position in `order_vector`.

Why (invariant used):
- In this preordered path, rows before the split row are already weakly-added prefix rows and are satisfied by active parent rays, so they cannot become the first infeasible row for the generated child.
- This trims unnecessary prefix scanning in exact-int create-ray classification.

### Validation

#### Repro correctness
- `data/motif_early/arcadian-accordion__hit_000001_w6_ex+1_sp2_seed6082620689740043133.json`
  - `ppl+hlbl:gmpint`: `5`
  - `howzat-dd@int:gmprat`: `5`
  - `howzat-dd@sp:gmprat`: `5`

#### Tests
- `cargo test -p howzat dd::tests -- --nocapture` passed.
- `cargo test -p howzat single_precision::tests -- --nocapture` passed.

#### Corpus parity/perf (`file:/tmp/howzat_motif_bench_137`)
Adjacency:
- `ppl+hlbl:gmpint`: avg `1.297ms`
- `howzat-dd@int:gmprat`: avg `1.462ms` (`1.13x`)
- `howzat-dd@sp:gmprat`: avg `3.020ms` (`2.33x`)
- mismatches: `0`

Incidence:
- `ppl+hlbl:gmpint`: avg `0.961ms`
- `howzat-dd@int:gmprat`: avg `1.104ms` (`1.15x`)
- `howzat-dd@sp:gmprat`: avg `2.430ms` (`2.53x`)
- mismatches: `0`

#### Motif fixed workload (real path, fixed seed)
Runs (direct `hirsch motif`, `checks=200000`, `timeout_ms=5`, `seed=424242`):
- `ppl+hlbl:gmpint`: `attempted_per_s = 6247.27`
- `howzat-dd@int:gmprat`: `attempted_per_s = 6056.57`

So this pass leaves exact-int about ~3% behind PPL on this workload path.

### Instrumented gap snapshot (still remaining)
PPL (`PPL_DD_PROFILE`, adjacency corpus aggregate):
- `pairs=1,644,186`
- `created=50,304`
- `create_ns=58,421,174`
- `create_ns_per_ray ≈ 1161`

howzat int (`HOWZAT_DD_PROFILE`, adjacency corpus aggregate):
- `pairs=1,852,144`
- `new_rays=53,459`
- `create_ns_per_ray` in current runs: ~`1.7k..2.0k` ns/ray (host-noise dependent), improved from earlier ~`2.2k`.

Interpretation:
- Core correctness is stable and parity-complete.
- Remaining overhead is now mostly per-ray create/classify cost plus a moderate pair/ray-count surplus versus PPL.

## 2026-02-20: PPL combine-path parity tightening (latest pass)

### Goal
Reduce the remaining exact-int regression without touching DD semantics.

### Source-level comparison (PPL vs howzat)
I re-read and compared:
- `kompute-hirsch/target/build-deps/ppl-sys/.../src/Polyhedron_conversion_templates.hh`
- `kompute-hirsch/target/build-deps/ppl-sys/.../src/Dense_Row.cc`
- `howzat/src/dd/umpire/int.rs`

Key PPL behavior in `create_new_ray`:
- `normalize2(...)` on row scalar products.
- `linear_combine(...)` with coefficient-1 specializations.
- `strong_normalize()` afterward.

Relevant divergence in howzat hot path:
- generated-ray combine loop always paid full multiply path and used per-coordinate `abs()` allocations while accumulating gcd information.

### What I changed (minimal hot-path change)
File:
- `howzat/src/dd/umpire/int.rs`

In `IntUmpire::generate_new_ray()`:
- keep the same exact formula and same normalization semantics;
- add coefficient-fast-path booleans (`a1_is_one`, `a2_is_one`) so we skip unnecessary multiplies when coefficient is `1`;
- skip second-term work when parent coordinate is zero;
- replace per-coordinate `abs()` allocation with scratch reuse:
  - `assign_from(&mut dot_tmp, &dot_acc)`
  - `dot_tmp.abs_mut()`
  - gcd update from `dot_tmp`

No DD semantic change; only arithmetic-path tightening.

### Important intermediate finding
A temporary profiling add-on accidentally kept nnz counters in the hot loop for non-profile runs; this caused a regression. That instrumentation overhead was removed, keeping only the combine-path optimization.

### Correctness revalidation
Repro file:
- `data/motif_early/arcadian-accordion__hit_000001_w6_ex+1_sp2_seed6082620689740043133.json`

Width checks (all still correct):
- `ppl+hlbl:gmpint = 5`
- `howzat-dd@int:gmprat = 5`
- `howzat-dd@sp:gmprat = 5`
- `howzat-dd[purify[snap]]:f64[eps[1e-12]] = 5`

Targeted test:
- `cargo test -p howzat dd::tests::adjacency_uses_added_halfspaces_face -- --nocapture` passed.

Corpus parity:
- `sandbox bench` still reports `mismatches=0`.

### Performance (latest clean pass)
Interleaved 8-run medians on `file:/tmp/howzat_motif_bench_137`:

Adjacency:
- `ppl+hlbl:gmpint`: `1.309 ms`
- `howzat-dd@int:gmprat`: `1.382 ms`
- ratio: `1.055x`

Incidence:
- `ppl+hlbl:gmpint`: `1.111 ms`
- `howzat-dd@int:gmprat`: `1.151 ms`
- ratio: `1.036x`

This is materially tighter than the prior ~`1.13x` / ~`1.15x` exact-int gap on the same corpus.

### Profile snapshot after this pass (adjacency corpus)
- Pair/ray-count deltas vs PPL remain (`pairs` and `new_rays` still above PPL).
- Per-ray create cost improved in non-profile wall-time behavior, but profiling still shows higher create-path cost than PPL.

Interpretation:
- The largest remaining gap now appears to be structural workload delta (extra pair/ray volume) plus residual per-ray arithmetic overhead, not the earlier obvious combine-path allocation issue.

### Final clean rerun after removing temporary instrumentation overhead
After stripping temporary nnz/unit-coefficient counters from the non-profile hot loop, I reran the interleaved benchmark matrix.

Interleaved 8-run medians (`file:/tmp/howzat_motif_bench_137`):
- Adjacency:
  - `ppl+hlbl:gmpint`: `1.225 ms`
  - `howzat-dd@int:gmprat`: `1.284 ms`
  - ratio: `1.048x`
- Incidence:
  - `ppl+hlbl:gmpint`: `1.037 ms`
  - `howzat-dd@int:gmprat`: `1.068 ms`
  - ratio: `1.030x`

Fixed-seed motif single-run spot check (`checks=200000`, `seed=424242`):
- `ppl+hlbl:gmpint`: `attempted_per_s = 6058.22`
- `howzat-dd@int:gmprat`: `attempted_per_s = 6057.85`

So with the latest combine-path optimization and cleanup, exact-int is now very close to PPL on this test box.

## 2026-02-20: SP exact-path deep perf pass (PPL-guided)

### Why this pass
Even after the core DD correctness fix, `howzat-dd@sp:gmprat` remained materially slower than both PPL and `howzat-dd@int:gmprat`.

### Re-measured baseline before edits
On `file:/tmp/howzat_motif_bench_137`:
- adjacency:
  - `ppl+hlbl:gmpint`: `1.176ms`
  - `howzat-dd@int:gmprat`: `1.235ms` (`1.05x`)
  - `howzat-dd@sp:gmprat`: `2.507ms` (`2.13x`)
- incidence:
  - `ppl+hlbl:gmpint`: `1.144ms`
  - `howzat-dd@int:gmprat`: `1.265ms` (`1.11x`)
  - `howzat-dd@sp:gmprat`: `2.760ms` (`2.41x`)

### Instrumentation and diagnosis
I instrumented `SinglePrecisionUmpire::generate_new_ray` (temporarily, under `HOWZAT_DD_PROFILE`) to split create time into eval/combine/norm/build, and compared to existing int-side profile and PPL counters.

Key finding on hard case (`r3-s6__hit_000005_w6_ex+1_sp3_seed...`):
- DD combinatorics are identical between `@int` and `@sp`:
  - `pairs=22627`, `new_rays=518`
- Remaining `@sp` gap was per-ray arithmetic cost (not more DD work).

### Accepted optimizations
1. `howzat/src/dd/umpire/single_precision.rs`
- In `generate_new_ray`, uncached `(val1,val2)` evaluation now uses `linalg::dot2(...)` (single row pass) instead of two separate `dot(...)` calls.

2. `ferramentum/calculo/src/linalg.rs` (`RugRatOps`)
- Added integer-denominator fast paths for `dot`, `dot2`, and `lin_comb2_into`.
- Added/kept coefficient and zero shortcuts in `lin_comb2_into` (`lhs/rhs factor is 0/1`, `lhs/rhs coord is 0`).

3. `ferramentum/calculo/src/num.rs` (`GcdNormalizer`)
- Added all-integral shortcut in normalize pass:
  - avoid LCM churn when denominators are already `1`;
  - normalize by numerator-gcd directly.

### Rejected/reverted experiments
- `gmprat[no]` / `gmprat[min|max]` normalizer alternatives:
  - rejected (`[no]` was catastrophically slower; `min/max` slower than default gcd).
- A deeper `integral_numer` normalizer API experiment:
  - reverted after measured regressions.
- Temporary SP profile-report plumbing:
  - removed from final source (used only for diagnosis).

### Validation
- `cargo test -p howzat dd::tests::adjacency_uses_added_halfspaces_face -- --nocapture` passed.
- `cargo test -p howzat single_precision::tests -- --nocapture` passed.
- `cargo check -p calculo` passed.

### Performance snapshot after accepted changes
(quiet host, deterministic corpus `file:/tmp/howzat_motif_bench_137`)
- adjacency:
  - `ppl+hlbl:gmpint`: `1.371ms`
  - `howzat-dd@int:gmprat`: `1.421ms` (`1.04x`)
  - `howzat-dd@sp:gmprat`: `2.253ms` (`1.64x`)
- incidence:
  - `ppl+hlbl:gmpint`: `1.156ms`
  - `howzat-dd@int:gmprat`: `1.184ms` (`1.02x`)
  - `howzat-dd@sp:gmprat`: `2.009ms` (`1.74x`)

So this pass materially reduced the `@sp` regression from roughly `~2.1x..2.4x` down to roughly `~1.6x..1.8x`, while preserving exact parity and the DD core correctness fix.

### Build note (workspace state)
- This `kompute-hirsch` checkout currently has unrelated borrow-check errors in dirty `bin-hirsch/src/cmd/motif.rs` (around lines 1330 and 1341), so a fresh `cargo build -p bin-hirsch --release` from that tree is blocked.
- Perf runs above were executed with the last successful `target/release/hirsch` binary present in this workspace.
