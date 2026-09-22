# Results: p_raw vs q* vs p_head on WANLI

First end-to-end run of the pipeline. Everything below is reproducible with the
commands at the bottom.

## Setup

| | |
|---|---|
| base model | `Qwen/Qwen3.5-2B-Base` (`qwen3_5`, frozen, bf16) |
| task | WANLI 3-way NLI as a choice question (Premise + Hypothesis -> entailment / neutral / contradiction) |
| splits | train 3000 / val 500 / test 500 (val carved from the tail of `train`) |
| views | 4 paraphrases x 3 cyclic option orders = 12 per item |
| head | 2-layer MLP on the hidden state at each option's **text-end** token, trained with forward KL to `q*` |
| calibration | one temperature per method, fitted on the training-time validation split |

`p_raw` is the free baseline (one forward pass, frozen label logits).
`q*` is the mean distribution over the 12 views (12 forward passes, no training).
`p_head` is the distilled head, evaluated on a single (canonical) view.

## Test results (n = 500)

| method | acc | bal_acc | nll | brier | ece | ece (cal) | conf |
|---|---|---|---|---|---|---|---|
| `p_raw` (1 pass, no train) | 0.612 | 0.473 | 0.813 | 0.507 | **0.077** | 0.069 | 0.674 |
| `q*` (12 passes, no train) | 0.614 | **0.570** | 0.864 | 0.522 | 0.092 | 0.072 | 0.524 |
| `p_head` (1 pass, trained) | **0.624** | 0.560 | 0.856 | 0.518 | 0.111 | 0.073 | 0.513 |
| `p_head_avg` (12 passes) | 0.624 | 0.515 | 0.854 | 0.516 | 0.109 | – | 0.515 |
| majority-class prior | 0.498 | 0.333 | 0.995 | 0.604 | 0.026 | – | 0.472 |

### Order consistency across the 12 permuted views

| readout | mean total-variation | argmax agreement |
|---|---|---|
| `p_raw` | 0.488 | **0.449** |
| `p_head` | **0.047** | **0.898** |

## Paired significance (vs `p_raw`, n = 500)

| method | dAcc | 95% CI | dBalAcc | discordant (b, c) | McNemar p |
|---|---|---|---|---|---|
| `q*` | +0.002 | [-0.042, +0.048] | **+0.097** | (66, 65) | 1.00 |
| `p_head` | +0.012 | [-0.028, +0.052] | **+0.087** | (56, 50) | 0.63 |

## What this says

1. **Raw accuracy does not improve significantly.** Both `q*` and `p_head` land
   inside the noise band of `p_raw`. On this task and this 2B backbone, the frozen
   model's own logits already carry the decision signal — consistent with the
   observation that a no-training logit readout (`SemIf`) gets ~0.845 agreement
   with Jev on TypeSafe's public cases.
2. **Balanced accuracy does improve, by ~9 points.** `p_raw` is majority-biased
   (it collapses onto `neutral`: bal_acc 0.473 vs acc 0.612). Averaging over
   paraphrases/orders and distilling that into a single pass largely removes the
   bias (`q*` 0.570, `p_head` 0.560), and the head keeps ~90% of `q*`'s gain at
   1/12 of the forward passes.
3. **Order invariance is the headline win.** Shuffling the options flips `p_raw`'s
   argmax more than half the time (agreement 0.449, TV 0.488). The head agrees
   0.898 across the same permutations (TV 0.047). This is the concrete, defensible
   benefit of the re-parameterised readout, and it is not visible in an
   accuracy-only table.
4. **Calibration is a wash.** `p_raw` is the best calibrated before any correction
   (ECE 0.077). After a single fitted temperature, all three land at ECE ~0.07.
   Self-consistency distillation did **not** buy better calibration here — which is
   the expected outcome when no calibrated teacher is available.

## High-cardinality task: Banking77 (26-way)

The 3-way NLI result is mostly a bias-correction story.  The hypothesis behind the
next step is that **order sensitivity costs more as the option set grows**, so the
head's invariance should convert into a real accuracy win.  Task: pick the correct
customer intent out of the 26 most frequent Banking77 intents (labels `A`..`Z`,
which is the ceiling for a single-token logit baseline).  2000/400/500 items,
6 views each (2 paraphrases x 3 permutations).

| method | acc | bal_acc | nll | brier | ece | ece (cal) | conf |
|---|---|---|---|---|---|---|---|
| `p_raw` (1 pass, no train) | 0.644 | 0.632 | 1.418 | 0.500 | 0.122 | 0.067 | 0.522 |
| `q*` (6 passes, no train) | **0.772** | **0.765** | 1.159 | 0.444 | 0.313 | 0.046 | 0.459 |
| `p_head` (1 pass, trained) | 0.762 | 0.751 | **1.124** | **0.436** | 0.278 | 0.049 | 0.484 |
| `p_head_avg` (6 passes) | 0.762 | 0.751 | 1.131 | 0.436 | 0.282 | – | 0.480 |
| majority-class prior | 0.040 | 0.038 | 3.275 | 0.963 | 0.019 | – | 0.059 |

Order consistency across the 6 permuted views:

| readout | mean total-variation | argmax agreement |
|---|---|---|
| `p_raw` | 0.453 | **0.556** |
| `p_head` | **0.083** | **0.928** |

Paired significance vs `p_raw` (n = 500):

| method | dAcc | 95% CI | dBalAcc | discordant (b, c) | McNemar p |
|---|---|---|---|---|---|
| `q*` | **+0.128** | [+0.090, +0.168] | +0.132 | (86, 22) | <0.0001 |
| `p_head` | **+0.118** | [+0.078, +0.160] | +0.119 | (85, 26) | <0.0001 |

### What changed relative to 3-way

1. **The accuracy gain is now real and large**: +11.8 points for the single-pass
   head, highly significant.  With 26 options, an unstable readout has more room to
   lose, and the head's order invariance converts directly into correctness.
2. **The head recovers ~92% of the multi-pass `q*` ceiling** (0.762 vs 0.772) at
   one sixth of the forward passes.
3. **Aggregation makes the model badly overconfident** — `q*` raw ECE 0.313.  A
   single fitted temperature brings every method to ECE ~0.05, so post-hoc
   calibration is not optional here; it is the difference between a usable and an
   unusable confidence signal.
4. Order invariance is the same story as before but stronger: argmax agreement
   0.556 -> 0.928.

For context, the Laya reproduction reports 42.5% on a 77-way choice where Jev gets
87.0%.  Different model and task, but the same shape: cardinality is where the
readout design starts to matter.

## Label-free debiasing: the prior was the dominant error

Our aggregation removed most of the **position** bias but never touched the
**label prior** -- and the WANLI result above is exactly a prior failure: `p_raw`
scores 0.612 accuracy but 0.473 balanced accuracy, i.e. it dumps mass on the
majority class.  This is the second of the two readout biases, and it is fixed by
two label-free estimators from the calibration literature:

* **batch prior** -- the mean predicted distribution over real inputs; divide it
  out and renormalize.  No labels, no extra forward passes (Zhou et al., ICLR 2024).
* **content-free prior** -- the distribution on inputs whose *state* has been
  replaced by a probe (`N/A`, `""`, `[MASK]`) (Zhao et al., ICML 2021).

Both are implemented in [`src/jev/debias.py`](../src/jev/debias.py).  Our arrays
store canonical order plus the permutation, so position space is reconstructed
from `view_order`; the prior is therefore estimated and applied *in position
space*, before marginalization.  We also switch the
marginalization to a **geometric mean** (`combine="logmean"`), which cancels an
additive per-slot bias exactly.  WANLI (K=3) uses all K rotations, so the
cancellation is exact there; at K=26 we use 3 seeded permutations, and the
all-K-versus-subset question is answered below.

### The rule that matters: debias the target, not the head

Applying a prior estimated from the *raw* readout to an already-distilled head is
wrong and visibly so: on 26-way Banking77 the frozen head drops from 0.762 to
**0.268**.  The head has already absorbed whatever bias is in `q*`, so the
correction is applied twice.  The fix is to fold the debiasing into the
distillation target -- `20_train_head.py --debias batch --combine logmean` -- so
the head learns the debiased mapping directly and needs no post-hoc correction.

### 26-way Banking77 (n=500)

| method | acc | bal_acc | nll | ece (cal) | cov@5% | dAcc vs `p_raw` |
|---|---|---|---|---|---|---|
| `p_raw` (1 pass, no train) | 0.644 | 0.632 | 1.418 | 0.067 | 0.220 | – |
| `p_raw` + batch prior (still 1 pass) | 0.810 | 0.810 | 0.973 | 0.056 | 0.250 | +0.166 |
| `q*` (V passes, no prior) | 0.772 | 0.765 | 1.159 | 0.046 | 0.420 | +0.128 |
| `q*` + batch prior + logmean | 0.850 | 0.852 | 0.857 | 0.045 | 0.570 | +0.206 |
| `p_head`, old (biased) target | 0.762 | 0.751 | 1.124 | 0.049 | 0.440 | +0.118 |
| **`p_head`, debiased target** | **0.844** | **0.846** | **0.827** | 0.044 | **0.600** | **+0.200** |
| `p_head_avg`, debiased target | **0.852** | **0.853** | 0.815 | 0.052 | 0.582 | +0.208 |

`p_head - p_raw = +0.200`, 95% CI `[+0.158, +0.244]`, McNemar `p = 3.3e-19`.
Order agreement across the 6 views: 0.556 (`p_raw`) -> 0.955 (debiased head);
the permutation-only flip rate goes 0.485 -> 0.054.

Two things are worth stating plainly:

1. **The prior correction is the largest single effect in this repo.**  It is free
   (no labels, and for the batch estimator not even a forward pass) and worth
   +0.166 accuracy on its own, before any training.
2. **A single forward pass now matches the multi-pass ceiling.**  The debiased
   head (0.844-0.852) is statistically indistinguishable from the debiased `q*`
   (0.850), so the compression from V passes to 1 works on the corrected target
   too -- and it is now +0.20 over the free baseline instead of +0.12.

### 3-way WANLI (n=500)

| method | acc | bal_acc | ece (cal) | cov@5% |
|---|---|---|---|---|
| `p_raw` | 0.612 | 0.473 | 0.069 | 0.000 |
| `p_raw` + batch prior | 0.608 | **0.640** | 0.098 | 0.008 |
| `q*` + logmean | **0.634** | 0.579 | 0.096 | 0.008 |
| `p_head`, old target | 0.624 | 0.560 | 0.073 | 0.022 |
| `p_head`, debiased (batch, logmean) | 0.612 | 0.639 | 0.100 | 0.016 |
| `p_head`, debiased (batch, mean) | 0.626 | 0.591 | **0.054** | 0.016 |

On WANLI the correction buys **balanced accuracy, not accuracy**: the model was
already picking the majority class, and 3-way NLI has no cardinality pressure, so
there is little headroom.  This is the honest reading of the two datasets: the
prior fix matters in proportion to how much the model's marginal output
distribution departs from the label marginal, which on a 26-way task with a
near-uniform gold marginal is enormous and on 3-way NLI is mild.

### Batch vs content-free

Both estimators were run on 26-way Banking77 (full train/test coverage) and 3-way
WANLI (content-free on a 1000-item prefix).  Batch wins on both, which matches the
literature's view of batch calibration as the low-variance estimator:

| task | method | `p_raw` acc / bal | `q*` acc / bal (logmean) | cov@5% (`p_raw`) |
|---|---|---|---|---|
| banking 26-way | none | 0.644 / 0.632 | 0.798 / 0.791 | 0.220 |
| | **batch** | **0.810 / 0.810** | **0.850 / 0.852** | **0.250** |
| | content_free | 0.746 / 0.749 | 0.822 / 0.826 | 0.296 |
| WANLI 3-way | none | 0.612 / 0.473 | 0.634 / 0.579 | 0.000 |
| | **batch** | 0.608 / **0.640** | 0.596 / 0.626 | 0.008 |
| | content_free | 0.608 / 0.597 | 0.586 / 0.605 | 0.008 |

The content-free probe is also the more expensive estimator (one forward pass per
item x view x probe), so there is no setting here where it is the right default.

### Does all-K cyclic marginalization help at high cardinality?

Permutation marginalization (Zheng et al., ICLR 2024) shows the options in all K
cyclic shifts, on the argument that an additive per-slot bias cancels exactly under
a geometric mean.
That is affine with our `max_orders` setting, so it is testable: WANLI (K=3)
already uses all 3 rotations, and the 26-way run uses 3 seeded random
permutations.  To compare the two at K=26 we re-induced the test split with
`MAX_ORDERS=0` (26 cyclic shifts x 2 paraphrases = 52 views, `--no-hidden` since
this is a `q*`-only question), both sides using a batch prior estimated from the
same split so only the marginalization differs:

| views (Banking77 26-way, n=500) | prior | combine | acc | bal_acc | cov@5% |
|---|---|---|---|---|---|
| 6 (3 random perms x 2) | none | mean | 0.772 | 0.765 | 0.420 |
| | none | **logmean** | **0.798** | **0.791** | **0.520** |
| | batch | mean | **0.842** | **0.845** | **0.644** |
| | batch | logmean | 0.842 | 0.845 | 0.576 |
| 52 (all 26 cyclic x 2) | none | logmean | 0.760 | 0.754 | 0.498 |
| | batch | mean | 0.834 | 0.833 | 0.490 |
| | batch | logmean | 0.830 | 0.830 | 0.490 |

**More views, all-K and theoretically exact, is slightly worse.**  9x the compute
buys -0.012 accuracy and -0.086 coverage against the cheap 3-permutation subset.
The likely reason is that a cyclic shift preserves *adjacency*: option ``i+1``
stays next to option ``i`` in every rotation, so any bias that depends on relative
order (or on a "recency"/suffix preference that is not purely additive per slot)
is not averaged out, while a random permutation destroys that structure.  The
practical consequence is that the costly all-K schedule is not worth it here, and
the repo keeps ``MAX_ORDERS`` as a knob with the default subset.

Two honest qualifications: the 52-view run has no hidden states, so this compares
the `q*` path only; and the theory behind cyclic shifts assumes a purely additive
slot bias, which this model, task and K evidently violate.

### A caveat about the batch estimator

The batch prior assumes the label marginal of the batch resembles the test
marginal, and it is estimated on the training split (never the test split).  We
report the gold marginals alongside every run for this reason, and they agree
closely on both datasets.  It is not a substitute for a proper calibration set if
the deployment marginal is genuinely different.

## LoRA: adapting the backbone, not just the readout

The frozen head is capped by the frozen representation.  `scripts/25_train_lora.py`
attaches LoRA (`r=8`, all linear layers, 8.4M trainable params, 0.44% of the model)
and trains it jointly with the head against the same `q*` targets, on the 26-way
Banking77 task.

| method | acc | bal_acc | nll | ece | ece (cal) |
|---|---|---|---|---|---|
| `p_raw` (frozen LM, 1 pass) | 0.644 | 0.632 | 1.418 | 0.122 | 0.067 |
| `q*` (frozen LM, 6 passes) | 0.772 | 0.765 | 1.159 | 0.313 | 0.046 |
| `p_head` (frozen + head, 1 pass) | 0.762 | 0.751 | 1.124 | 0.278 | 0.049 |
| `p_lora_head` (LoRA + head, 1 pass) | **0.780** | **0.769** | **1.124** | 0.311 | **0.046** |

Order consistency of the LoRA head across the 6 views: argmax agreement **0.966**
(mean TV 0.036), versus 0.556 for the frozen `p_raw` and 0.928 for the frozen head.

### Reading

1. **LoRA exceeds the frozen multi-pass ceiling.**  `p_lora_head` (0.780) is above
   `q*` (0.772), which a frozen readout provably cannot be.  Letting the
   representation move buys ~1.8 points over the frozen head and ~0.8 over the
   best frozen aggregate — the expected direction, since the frozen head was capped.
2. **It is the best method on every metric once calibrated**: best accuracy, best
   balanced accuracy, best NLL, and an ECE (0.046) matching `q*` and better than
   `p_raw` (0.067).
3. **Order invariance gets even better** (0.966 vs 0.928), so the adaptation did
   not come at the cost of stability.
4. **Do not ship the raw logits path with LoRA.**  `p_lora_raw` collapses to 0.092:
   LoRA training only ever optimises the head's KL loss, so nothing preserves the
   base model's own label-logit readout.  The head is not optional in this variant.
   (Getting good calibration from it required fitting the temperature on the head's
   raw scores rather than its softmax output — an earlier run passed probabilities
   into the scaler and reported a spurious ECE of 0.276.)

## Noul formulation, 77 options (arbitrary cardinality)

The choice format is capped at 26 options by the single-token label requirement.
`scripts/11_induce_noul.py` instead renders **one prompt per option** and reads the
final-token Yes/No contrast; that supports any K and is order-invariant by
construction.  Full Banking77 (2000/400/500, 2 paraphrases = 2 views, 77 options),
run on two base models to separate "the primitive works" from "the model can answer
an independent yes/no".

| base model | method | acc | bal_acc | nll | ece | ece (cal) | conf | order agreement |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-2B-**Base** | `p_raw` | 0.376 | 0.386 | 3.874 | 0.351 | 0.072 | 0.025 | 0.372 |
| | `q*` | 0.422 | 0.437 | 3.812 | 0.396 | 0.082 | 0.026 | – |
| | `p_head` | 0.472 | 0.483 | 3.812 | 0.447 | 0.131 | 0.025 | 0.702 |
| Qwen2.5-1.5B-**Instruct** | `p_raw` | 0.472 | 0.498 | 2.519 | 0.321 | 0.041 | 0.151 | 0.694 |
| | `q*` | 0.504 | 0.530 | 2.144 | 0.255 | 0.060 | 0.250 | – |
| | `p_head` | **0.556** | **0.577** | **2.133** | 0.321 | 0.044 | 0.235 | **0.782** |

### Reading

1. **Instruction tuning is what makes the primitive usable, and it is not about
   size.**  On the *base* model the per-option `p_yes` are barely separated: `nll`
   3.81 against `log 77 = 4.34` and `conf` 0.025, about 2x uniform — the ranking
   carries signal while the probabilities do not.  The instruct model is *smaller*
   (1.5B vs 2B) yet roughly halves the NLL (2.52) and lifts top-1 confidence 6x
   (0.151), giving ECE 0.041-0.060 after one fitted temperature.
2. **Heads help on both**: +9.6 points of accuracy on the base model and +8.4 on the
   instruct one, over `p_raw`; and in this formulation the single-pass head also
   beats the multi-pass `q*` (0.472 vs 0.422, and 0.556 vs 0.504).
3. **Order invariance improves in both** (base 0.372 -> 0.702; instruct 0.694 ->
   0.782).  The noul prompt never lists the options, so this is paraphrase
   stability rather than permutation stability.
4. **Best 77-way result: 0.556 acc / 0.577 bal_acc with ECE 0.044**, from a 1.5B
   model, one forward pass per option at inference and a head that costs nothing
   measurable.
5. **Caveat that generalises**: a noul's probability is only as good as the model's
   willingness to answer an independent yes/no crisply.  On a base model the noul
   interface is a ranker, not a source of calibrated confidence.


## Bugs that mattered

The first version read the hidden state at each option's **label** token (`"A"`).
In a causal LM that position has not yet read the option text, so the head was
scoring options it could not see; it scored at chance while `p_raw` did the real
work. Reading at the **last token of the option text** fixed it. See
`src/jev/prompts.py::render_options` and `OptionReader.read(hidden_offsets=...)`.

The second was in the LoRA trainer: head scores come out in **display order** while
`q*` is in **canonical order**, so the KL loss was trained against a permuted
target.  With the identity view first, this bug is invisible in accuracy and only
shows up as a collapse in order consistency (agreement 0.274).  Fix: permute the
target with `view.order` before the loss.

Both bugs are the same class: a per-option readout is easy to mis-align silently,
and an accuracy table alone will not tell you.

## Reproduce

See [`REPRODUCE.md`](REPRODUCE.md) for the environment, data, choice / noul / LoRA
and analysis commands; `run.sh` drives the whole pipeline on one GPU.

## Next steps

- **Select LoRA checkpoints on validation NLL**, not the last epoch, and refit the
  temperature afterwards.
- **A calibrated reference**: score against TypeSafe's published workflow cases to
  measure the real calibration gap rather than only comparing to `p_raw`.
- **Higher-capacity heads / cross-option attention** where cardinality is large,
  instead of the pointwise per-option MLP.
- **Fix the noul phrasing for binary questions**: an option-conditioned paraphrase
  is a poor framing for a 2-option Bernoulli, where a direct yes/no question is the
  natural primitive.
- **Learn the selective-prediction threshold** rather than the fixed 5% cut, since
  that is where the head's advantage actually shows up.
