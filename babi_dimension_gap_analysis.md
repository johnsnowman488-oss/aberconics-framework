# bAbI QA1 Dimension and Slow-Channel Gap Analysis

## Scope

The active 32-dimensional sweep was intentionally terminated after enough results had accumulated. The run used the corrected readout settings: hidden dimension 32, learning rate 0.005, 10 epochs, 900 training stories, and 1,000 test stories. The memory variants were `full`, `no_slow`, and `collapsed_gamma`.

## Preserved 32-D results

| Variant | Seeds completed | Mean accuracy | Sample SD | Approx. 95% CI |
|---|---:|---:|---:|---:|
| `full` | 20 | 43.39% | 1.46 pp | ±0.68 pp |
| `no_slow` | 20 | 47.62% | 1.30 pp | ±0.61 pp |
| `collapsed_gamma` | 10 | 46.59% | 1.66 pp | ±1.19 pp |

The paired comparison uses identical readout seeds. The `no_slow - full` contrast is **+4.23 percentage points**, positive on **20/20** seeds, with an approximate 95% CI of **+3.47 to +4.99 pp**. The first ten `collapsed_gamma - full` differences average **+3.22 points**, positive on **10/10** seeds. Over those same ten seeds, `collapsed_gamma - no_slow` averages **−0.69 points**, with only 1/10 positive differences.

## Comparison with 16-D corrected confirmation

The corrected 16-D confirmation used five seeds at the same readout settings:

| Dimension | `full` | `no_slow` | Gap (`no_slow - full`) |
|---:|---:|---:|---:|
| 16 | 41.40% | 44.26% | +2.86 pp |
| 32 | 43.39% | 47.62% | +4.23 pp |

The 16-D and 32-D seed counts are not identical, so this is directional rather than a fully balanced dimension study. Nevertheless, the preserved results do **not** show the slow channel becoming helpful as total dimension increases. The opposite pattern is observed: `full` improves by about 2.0 points from 16 to 32, while `no_slow` improves by about 3.4 points.

## Interpretation

The readout diagnosis was correct: the original learning rate of 0.05 was too high. At 0.005, QA1 accuracy rises from approximately 18–23% to the low-to-high 40% range. That optimization correction explains most of the original apparent gap.

After correction, the remaining result is stable: the slow channel is not contributing positively under the current representation and readout. The `collapsed_gamma` control is much closer to `no_slow` than to `full`, suggesting that the issue is associated with the distinct slow timescale rather than merely the number of channels.

This is not yet evidence that slow memory is intrinsically harmful. The current experiment has several confounds:

1. The query is a bag-of-words forcing vector rather than a structured query representation.
2. The classifier receives a large gated feature vector and may favor short-timescale features.
3. The diagnostic uses a single online SGD update per story; it does not optimize a batch objective.
4. The 16-D corrected comparison has only five seeds, while 32-D has 20 seeds for `full` and `no_slow` but only 10 for `collapsed_gamma`.
5. The final 32-D JSON bundle was not written because the process was terminated while the diagnostic was still in progress; the values above were preserved from the terminal output.

## Decision for the next experiment

Do not launch another full 20-seed dimension sweep immediately. The next run should be a **quick gap sweep** with 3 seeds per dimension and all three variants, using the corrected readout settings. Use dimensions 16, 32, 64, and optionally 128. This identifies whether the gap trends upward, downward, or changes sign before committing to 20-seed runs.

Only after the quick sweep identifies a dimension where the slow channel plausibly participates should a full 20-seed sweep be run at that dimension.

The most informative follow-up is a feature-path diagnostic at the selected dimension:

- `u` only;
- weighted memory readout only;
- concatenated `[u, memory]`;
- individual channel vectors `chi_1`, `chi_2`, `chi_3`;
- query-gated features.

That experiment can distinguish slow-channel information loss from slow-channel interference in the downstream readout.

## Operational lesson

Use the following policy going forward:

- **Quick gap sweep:** 3 seeds, 5 epochs, one learning rate already validated, all variants and candidate dimensions.
- **Full sweep:** 20 seeds, 10 epochs, only after the quick sweep identifies a meaningful dimension/variant pattern.
- **Never use the original learning rate 0.05** for this readout; use 0.005 as the baseline and test 0.01 only as a controlled alternative.
