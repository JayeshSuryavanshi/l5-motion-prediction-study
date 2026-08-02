# l5-motion-prediction-study

Motion prediction for autonomous driving on the Lyft Level 5 Prediction
dataset, run entirely on one laptop: an Apple M1 Pro with 16 GB of RAM and a
strict disk budget. The 2020 Kaggle competition on this dataset was won with
multi-GPU ensembles trained on over a hundred million samples. This study asks
a smaller, and to me more practical, question: with roughly a thousandth of
that compute, what do classical baselines and small neural models actually
achieve, and which design choices matter most under a fixed wall-clock budget?

Work in progress. The results table fills in as runs complete. Every number
in it is produced by a command in this repository, on the hardware above, on
a declared data subset, with the seed listed.

## Study design

All experiments will share one frozen evaluation protocol: a held-out set of
scenes from `train.zarr` (the last ~1,000 scenes, never trained on), chopped
at a fixed frame so each qualifying agent gets one prediction unit with a
5-second (50 frame) future, scored with the multi-modal negative
log-likelihood used by the competition. The metric implementation lives in
`src/l5study/metrics.py` and is unit-tested against the reference
implementation in l5kit. Scores are comparable within this repository, not
with competition leaderboard numbers, which used a different (private) test
set.

The ladder, from no learning to small learned models:

| # | Model | Map | Learned | Status |
|---|-------|-----|---------|--------|
| E0 | Stationary / constant velocity / constant turn | no | no | done |
| E1 | Three-mode kinematic mixture | no | no | done |
| E2 | History-only MLP | no | yes | done |
| E3 | Raster CNN (resnet18, 128 px) | yes | yes | pending |
| E4 | Raster CNN (resnet18, 224 px) | yes | yes | pending |
| E5 | Vectorized model (polyline encoder + attention) | yes | yes | pending |

Planned ablations on the strongest rungs: history length, raster resolution
at fixed wall-clock, number of predicted modes, backbone size, and a
data-scaling curve.

## Results

Frozen holdout: the last 1,000 scenes of `train.zarr` (scenes 15,265 to
16,264, never used for training), chopped at frame 100, giving 6,138
agents with a 5-second future each. The two splits share no frames: the
scene cut is a contiguous index cut, and l5kit builds each sample's
history and future strictly within its own scene (verified; the nearest
holdout ground-truth frame lies more than 10 seconds past the boundary).
Metric columns: multi-modal negative log-likelihood (the competition
metric, lower is better), and best-mode average / final displacement
error in meters. Baselines reproduce with
`uv run python scripts/run_baselines.py --zarr data/scenes/holdout.zarr`
(deterministic, no seeds); E2 with
`uv run python scripts/train_e2.py` (seed 0, ~30 s of training on an
M1 Pro after one-time feature extraction).

| Model | NLL | ADE | FDE |
|-------|----:|----:|----:|
| E2 history MLP (200k samples, no map) | 128.94 | 0.79 | 1.52 |
| E1 kinematic mixture (3 modes) | 288.99 | 1.57 | 3.33 |
| E0 constant velocity | 310.52 | 1.20 | 2.62 |
| E0 constant turn | 701.74 | 1.96 | 4.49 |
| E0 stationary | 7496.93 | 7.57 | 13.26 |

Early observations, to be tested further up the ladder: a mixture of three
kinematic modes beats its own best single mode on NLL purely through mode
diversity, even though constant velocity has the better displacement
errors; fitting a yaw rate from noisy 0.4-second histories actively
hurts (constant turn loses to constant velocity across the board); and a
0.7M-parameter MLP that sees only the same 11-frame history the kinematic
baselines see, still no map, cuts NLL by more than half. One honest
caveat on E2: its in-pool validation NLL is ~48 while the holdout NLL is
~129; the training pool contains only agents with full 11-frame
histories, while the holdout protocol admits partial-history agents
(about 4.5% of the eval set) and a different agent mix at the chop
frame, so part of E2's gap to the baselines' protocol is distribution
shift the learned model has to absorb and the kinematic ones dodge by
construction.

A pipeline-validation run of the same baselines on the public
`sample.zarr` split lives in `results/baselines_sample/metrics.csv`; the
numbers track the holdout closely, which is a useful protocol sanity
check, but the holdout table above is the study reference.

## Reproduce

```
make setup   # uv environment + patched l5kit (see note below)
make test    # metric and baseline unit tests, synthetic data only
```

Data setup instructions will land with the first experiment. The dataset is
not redistributed here and must be obtained from its own source under its
own terms.

Note on l5kit: the library is archived upstream and its pinned dependencies
no longer resolve on current Python or Apple Silicon. `make setup` installs
it without dependencies against a modern pinned stack and applies a small
compatibility patch for removed numpy aliases (`scripts/patch_l5kit.py`).

## Limitations

- Trained on a subset of the dataset, on a single laptop. Numbers are not
  comparable to competition results and no leaderboard claims are made.
- MPS (Apple GPU) training only; fp32 with bf16 autocast, no CUDA.
- Negative results are reported alongside positive ones.

## Data and licensing

The code is licensed Apache-2.0. The Lyft Level 5 Prediction dataset is
licensed CC BY-NC-SA 4.0 by its authors, so any use of this code together
with that data is bound by the dataset's non-commercial terms. No data, in
raw or derived form, is committed to this repository.

If you use the dataset, cite its paper:

```bibtex
@inproceedings{houston2020one,
  title     = {One Thousand and One Hours: Self-driving Motion Prediction Dataset},
  author    = {Houston, John and Zuidhof, Guido and Bergamini, Luca and Ye, Yawei
               and Chen, Long and Jain, Ashesh and Omari, Sammy
               and Iglovikov, Vladimir and Ondruska, Peter},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2020}
}
```

Dataset tooling: [l5kit](https://github.com/woven-by-toyota/l5kit) (Apache-2.0).
