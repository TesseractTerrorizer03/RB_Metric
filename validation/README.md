# Component A validation

## Findings

The experiment fit on **4,387 attempts in 80 games** and tested on **2,355 attempts in 42 different games**. Catch-depth and completion forecasts improved materially. The YAC fit reduced large errors and nearly removed the average underprediction, while the average absolute YAC error barely changed.

| Held-out check | Original | Fitted | Test sample |
| --- | ---: | ---: | ---: |
| Catch depth MAE | 2.02 yd | **1.35 yd** | 1,509 completions |
| Intended depth MAE | 3.07 yd | **1.95 yd** | 2,355 attempts |
| YAC MAE | 3.47 yd | 3.44 yd | 1,509 completions |
| YAC RMSE | 6.18 yd | **5.51 yd** | 1,509 completions |
| Total completed gain MAE | 4.96 yd | **3.72 yd** | 1,509 completions |
| Completion Brier score | 0.240 | **0.193** | 2,355 attempts |
| Completion log loss | 0.754 | **0.573** | 2,355 attempts |

Lower is better. Predicting the training completion rate on every attempt scored 0.230 Brier. The old proxy underpredicted the observed rate by 10.9 percentage points on average, versus a 1.5-point overprediction by the fitted model. Adding the two downfield defender features to the geometry-only YAC fit improved its MAE from 3.54 to 3.44 yards. Compared with the original YAC proxy, the paired game-bootstrap 95% interval for the MAE change was −0.21 to +0.15 yards.

On 2,282 held-out non-turnover attempts, the original signed value had an RMSE of **1.376** on the hand-set point scale. Changing depth alone yielded **1.364**, YAC alone **1.346**, completion alone **1.258**, and all three **1.226**. The combined model removed most of the downward bias (−0.461 to +0.005), while MAE moved from 0.896 to 0.915. The largest gain on this RMSE test came from the completion fit.

### Kupp's two Giants plays

These estimates are taken from the last tracked frame before release, so a value can differ from the earlier analysis of a route's peak frame.

| Play | Observed depth / YAC | Original estimate | Fitted estimate | Original → fitted completion |
| --- | ---: | ---: | ---: | ---: |
| 13-yard TD | 0 / 13 yd | −0.8 / 2.4 yd | 0.1 / 6.2 yd | 60% → 74% |
| 28-yard catch | 27 / 1 yd | 19.5 / 0.0 yd | 27.1 / 10.0 yd | 5% → 33% |

The fit improved depth on both throws. It still underpredicted the long run after the short catch and overpredicted YAC on the deep ball. On the touchdown, the original signed value was −0.477, the fitted inputs with original points gave −0.110, and the same fitted inputs with nflfastR points gave +0.070.

### Point model and rankings

The published booster reproduced nflverse's snap EP to six decimal places on the matched test sample. Among **93 WRs with at least 40 routes in Weeks 6–8**, Kupp was **82nd** with the original score, **84th** with all fitted inputs and original points, and **77th** after the nflfastR point swap. His mean positive peak value went from **0.355** to **0.725** under the same point formula, so the lower rank came from relative changes across receivers. Holding the fitted inputs fixed, the two point models' WR ranks had a Spearman correlation of **0.91**. These ranks use the held-out cohort and its route minimum.

The stronger forecasts support the play-level analysis. They were tested on actual throws at release, where the quarterback selected a target. Applying them earlier in a play or to an untargeted route is an extrapolation. The rank shifts show sensitivity to modeling choices. For a future version, predicting a distribution of completed gains would let the score account for the chances of a first down or touchdown even when the mean gain falls short.

## Method

This experiment matches the NYU 2021 tracking data to [nflverse play-by-play](https://github.com/nflverse/nflverse-data/releases/tag/pbp) by legacy game and play ID, and to the [nflverse player IDs](https://github.com/nflverse/nflverse-data/releases/tag/players) by GSIS receiver ID. It uses Weeks 1–5 for fitting and Weeks 6–8 for testing, with whole games assigned to one split. The models use the last fully tracked frame before release for each matched attempt. Penalties, spikes, sacks, and plays without a matched target or intended air yards are outside the attempt sample. Interceptions count as failed completions and are excluded, along with lost fumbles, from the binary completion-versus-incompletion value benchmark.

The depth model learns air yards from pre-release tracking features. The YAC model learns from completions using projected catch geometry and receiver motion. A second YAC fit adds the closest defender downfield and the number of defenders within 12 yards forward and five yards laterally of the projected catch point. The completion model is a regularized logistic regression using pre-release geometry and the original completion logit. Actual air yards, YAC, and pass outcome serve as labels. Test features come from the tracking available at the scored frame.

The scoring comparison changes depth, YAC, and completion separately with the original point formula held fixed, then changes all three. The final version uses the public [fastrmodels expected-points booster](https://github.com/nflverse/fastrmodels) on the same predicted completion and incompletion states. Hypothetical states retain the snap clock and timeouts and use the same touchdown and safety conventions as the original. The script checks the booster against published snap EP before scoring. Release-frame accuracy evaluates observed attempts. The WR ranking comparison applies those fitted models to all legal pre-release frames in the test games and averages each route's peak positive value. It is a ranking sensitivity check.

## Reproduce

Run from the workspace root with the Component A requirements and `pyarrow`, `scikit-learn`, `scipy`, `threadpoolctl`, and `xgboost` installed. The test used scikit-learn 1.6.1 and XGBoost 2.1.4.

```bash
inputs=$(mktemp -d)
curl -fL 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2021.parquet' -o "$inputs/pbp.parquet"
curl -fL 'https://github.com/nflverse/nflverse-data/releases/download/players/players.csv' -o "$inputs/players.csv"
curl -fL 'https://raw.githubusercontent.com/nflverse/fastrmodels/master/data/ep_model.rda' -o "$inputs/ep_model.rda"
python3 "RB_Metric/validation/run_validation.py" \
  --pbp "$inputs/pbp.parquet" --players "$inputs/players.csv" --ep-model "$inputs/ep_model.rda"
```

The script writes attempt-level predictions, accuracy metrics, completion calibration bins, and WR rank comparisons to `RB_Metric/results/validation/`. Error columns in the metrics CSV are MAE and RMSE except on completion rows, where they are Brier score and log loss. Bias is mean prediction minus mean observed value.
