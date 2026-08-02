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
| E0 | Stationary / constant velocity / constant turn | no | no | pending |
| E1 | Three-mode kinematic mixture | no | no | pending |
| E2 | History-only MLP | no | yes | pending |
| E3 | Raster CNN (resnet18, 128 px) | yes | yes | pending |
| E4 | Raster CNN (resnet18, 224 px) | yes | yes | pending |
| E5 | Vectorized model (polyline encoder + attention) | yes | yes | pending |

Planned ablations on the strongest rungs: history length, raster resolution
at fixed wall-clock, number of predicted modes, backbone size, and a
data-scaling curve.

## Results

None yet on the frozen holdout. A pipeline-validation run of the
no-learning baselines on the public `sample.zarr` split (480 agents) lives
in `results/baselines_sample/metrics.csv`; treat those numbers as a smoke
test of the protocol, not study results.

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
