# Paper Breakdown — "The Geometry of Truth"

**Paper:** Samuel Marks (Northeastern University) & Max Tegmark (MIT), "The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets," arXiv:2310.06824 (submitted Oct 10, 2023; v3 revised Aug 19, 2024).
**Code/data:** https://github.com/saprmarks/geometry-of-truth (includes an interactive data explorer)

---

## 1. The question the paper asks

Large language models often output falsehoods — sometimes because they don't know better, and sometimes, apparently, even when they "know" a statement is false (the paper opens with the GPT-4 agent that lied about being a vision-impaired human to get help solving a CAPTCHA). A natural goal: given a model M and a statement s, determine whether M *believes* s is true.

One approach is to read the model's mind instead of its outputs: train a **probe** — a small classifier — on the model's internal activations to predict whether a statement is true. Prior work (Azaria & Mitchell 2023; Li et al. 2023; Burns et al. 2023) did exactly this, and the fact that *linear* probes worked suggested there might be a **"truth direction"** — a single direction in activation space that separates true from false statements.

But this was controversial. Levinstein & Herrmann (2023) showed that earlier probes failed to generalize in basic ways — e.g., they broke on statements containing the word "not" — suggesting the probes may have latched onto features that merely *correlate* with truth on their training data rather than truth itself.

**The paper's question:** Do LLMs actually have a unified, linearly-represented notion of truth that spans structurally and topically diverse factual statements — and if so, where does it live, how general is it, and is it causally used by the model?

All experiments use autoregressive transformers from the **LLaMA-2 family (7B, 13B, 70B)**.

## 2. The method

### 2.1 Curated true/false datasets (the data-mining core of the paper)

Instead of using messy existing benchmarks, the authors built **high-quality, deliberately simple datasets** of unambiguous true/false statements. Each curated dataset follows a fixed template on a fixed topic, but the *union* spans diverse topics and — crucially — diverse *structures*: plain statements, negations with "not," and logical conjunctions/disjunctions. This design lets them separate "truth" from surface features like sentence shape or subject matter.

| Dataset | Template / description | Rows |
|---|---|---|
| `cities` | "The city of [city] is in [country]." | 1,496 |
| `neg_cities` | Negations of `cities` statements with "not" | 1,496 |
| `sp_en_trans` | "The Spanish word '[word]' means '[English word]'." | 354 |
| `neg_sp_en_trans` | Negations of `sp_en_trans` with "not" | 354 |
| `larger_than` | "x is larger than y." (numbers in words) | 1,980 |
| `smaller_than` | "x is smaller than y." | 1,980 |
| `cities_cities_conj` | Conjunctions of two `cities` statements with "and" | 1,500 |
| `cities_cities_disj` | Disjunctions of two `cities` statements with "or" | 1,500 |

Examples: "Sixty-one is larger than seventy-four" (false); "The Spanish word 'nariz' does not mean 'giraffe'" (true).

They also use **uncurated datasets** adapted from prior work as harder, messier test sets:

| Dataset | Source | Rows |
|---|---|---|
| `companies_true_false` | Azaria & Mitchell (2023) | 1,200 |
| `common_claim_true_false` | Casper et al. (2023) | 4,450 |
| `counterfact_true_false` | Meng et al. (2022) | 31,960 |

And one special control dataset:

- **`likely`** (10,000 rows): *nonfactual* text whose final token is either the most likely or the 100th-most-likely completion according to LLaMA-13B. This control exists to check that "truth" directions aren't just "probable vs. improbable text" directions. This matters because probability and truth are correlated in some datasets and *anti*-correlated in others: for `cities`, truth value and the log-probability LLaMA-2-70B assigns correlate at r = .85; for `sp_en_trans`, r = .95; but for `neg_cities`, r = −.63, and for `neg_sp_en_trans`, r = −.89 (e.g., after "The city of Paris is not in," the model still finds "France" the most probable continuation — probable, but false).

### 2.2 Localizing where truth lives: activation patching (§3)

Before probing, the authors ask *which hidden states* might encode a statement's truth value. Using a few-shot prompt ("The city of Tokyo is in Japan. This statement is: TRUE … The city of Chicago is in Canada. This statement is:"), they run **activation patching**: cache residual-stream activations from a run where the final statement is true, then swap individual activations (one token position × layer at a time) into a run where it's false, and measure the change in log P(TRUE) − log P(FALSE).

For LLaMA-2-13B on `cities`, three groups of causally implicated hidden states appear:

- **Group (a):** states over the statement's subject (e.g., "Chicago"/"Toronto") — plausibly encoding the entity.
- **Group (b):** states over the **final token of the statement and the end-of-sentence punctuation** — they summarize the whole statement and are causally influential on the TRUE/FALSE decision. The authors hypothesize these store the statement's truth representation, and study them for the rest of the paper. (For LLaMA-2-13B, the most downstream group-(b) state used is the **layer 15** residual-stream activation over the end-of-sentence punctuation token.)
- **Group (c):** late states that directly encode the model's prediction (applying the unembedding gives top logits like "true", "True", "TRUE").

### 2.3 Visualizing representations with PCA (§4)

They take group-(b) activations for each dataset (no few-shot priming this time), center them, and plot the top two principal components. In LLaMA-2-13B and 70B, true and false statements **separate cleanly along a line** in the top PCs — direct visual evidence of linear structure. The structure emerges rapidly in early-middle layers, and later for structurally complex statements (e.g., conjunctions).

### 2.4 Probing techniques compared (§5)

Three linear probing methods are trained on one dataset and tested for **cross-dataset generalization** on all the others:

1. **Logistic regression (LR)** — the standard approach.
2. **Mass-mean (MM) probing** — the paper's proposed method. A **difference-in-means** direction: θ_mm = μ⁺ − μ⁻, the vector from the mean activation of false statements to the mean activation of true statements. The probe is p_mm(x) = σ(θ_mmᵀx). For in-distribution (IID) evaluation, a covariance-corrected variant p_mm^iid(x) = σ(θ_mmᵀΣ⁻¹x) tilts the decision boundary to account for interference from non-orthogonal features (this coincides with linear discriminant analysis; with infinite data and a homoscedasticity assumption it recovers the LR direction — but MM additionally tracks a candidate *feature direction*, which is what matters for causal interventions).
   - **Why not just LR?** The paper gives a clean failure case: if another feature f is represented along a direction non-orthogonal to the truth direction, LR converges to a maximum-margin separator that mixes in f — it finds a good *classifier* but the wrong *direction*. Difference-in-means points at the feature itself.
3. **Contrast-consistent search (CCS)** — the unsupervised method of Burns et al. (2023), using contrast pairs (a statement and its negation).

**Setup:** probes are trained on one of {`cities`, `cities` + `neg_cities`, `larger_than`, `larger_than` + `smaller_than`, `likely`}, on a random 80% split, and evaluated on all curated + uncurated test sets. Baselines: calibrated few-shot prompting (surprisingly weak) and an oracle LR trained on the test set itself.

### 2.5 Causal interventions (§6)

Classification accuracy alone can't prove a direction *is* truth — it might be a correlate. So the authors **intervene**: take a probe direction θ (normalized so that adding θ moves the average false representation to the average true one), add/subtract it from group-(b) hidden states during a forward pass on `sp_en_trans` statements in a few-shot TRUE/FALSE prompt, and see whether the model flips its answer. Effects are reported as **normalized indirect effects (NIEs)**: 0 = intervention did nothing; 1 = the model now labels false statements TRUE with as much confidence as genuine true statements (or vice versa). Crucially, unlike prior work (Li et al. 2023), interventions are evaluated **out-of-distribution** — directions extracted from one dataset are tested on another.

## 3. Key findings

1. **Truth is linearly represented, and the representation gets more abstract with scale.** PCA visualizations show true/false statements separating linearly in LLaMA-2-13B and 70B. In the 7B model, by contrast, representations cluster by surface features (e.g., statements containing the token "eighty"). Larger models develop a more general notion of truth that applies across structurally and topically diverse inputs.

2. **Alignment across datasets is real but develops gradually.** The first PC of `cities` also separates true/false statements in other, even uncurated, datasets. But there are striking misalignments: in LLaMA-2-13B, `larger_than` and `smaller_than` separate along *antipodal* (opposite) directions, while in the 70B model they align along a common direction. Statements and their negations can separate along approximately *orthogonal* directions. The paper's interpretation: smaller models (and earlier layers) represent intermediate concepts like "close association" (Beijing–China are associated), which correlates with truth on `cities` but *anti*-correlates on `neg_cities`; at scale, these resolve into a unified truth direction. Within a single forward pass of the 13B model you can watch this happen across layers — antipodal, then orthogonal, then aligned.

3. **Probes generalize across datasets — much better for larger models.** For LLaMA-2-13B and 70B, cross-dataset generalization is generally high: e.g., probes trained on `larger_than` + `smaller_than` achieve **>95% accuracy on `sp_en_trans`** regardless of probing technique — number comparisons teaching a probe that transfers to Spanish translations.

4. **Training on statements *and their opposites* improves generalization.** Going from `cities` to `cities` + `neg_cities` improves transfer (largely via better generalization to other negated statements like `neg_sp_en_trans` — the exact failure mode of earlier probes). Going from `larger_than` to `larger_than` + `smaller_than` also helps, even though those datasets are structurally unlike the others. The interpretation: pairing statements with their opposites cancels out non-truth features whose correlation with truth flips sign between the pair.

5. **The truth direction is not just "probable text."** Probes trained on the `likely` dataset do well only where probability and truth happen to correlate (e.g., `sp_en_trans`, r = .95) and perform **worse than chance** where they anti-correlate. LLMs linearly represent truth-relevant information beyond text plausibility.

6. **Mass-mean probes match other methods on accuracy but are far more causal.** For the larger models, MM generalizes about as well as LR and CCS (it underperforms on the 7B model). But in the intervention experiments, **MM directions beat LR and CCS in 7 of 8 conditions**, often substantially — despite near-identical classification accuracies. From Table 2 (NIEs on `sp_en_trans`):
   - LLaMA-2-13B, trained on `cities` + `neg_cities`: MM achieves NIE **0.85** (false→true) and **0.97** (true→false), vs. LR's 0.33 / 0.52 and CCS's 0.31 / 0.73.
   - LLaMA-2-13B, trained on `cities` alone: MM 0.77 / 0.90 vs. LR 0.13 / 0.19.
   - LLaMA-2-70B: effects are strong overall (e.g., LR trained on `cities` + `neg_cities` reaches 0.61 / 1.00; MM 0.81 / 0.95).
   - Directions from `likely` are a surprisingly decent intervention baseline but still weaker than truth probes — an open puzzle (see Limitations).

   In plain terms: adding the mass-mean "truth direction" to a false statement's representation can make the model treat it as true, and subtracting it makes true statements read as false — **causal evidence** that the direction isn't merely correlated with truth but is part of how the model computes its answer.

## 4. Ablations / key experimental variations

- **Model scale (7B / 13B / 70B):** linear truth structure, probe generalization, and cross-dataset alignment all strengthen with scale; MM only matches LR/CCS at 13B+.
- **Training-set diversity:** statements alone vs. statements + their negations/opposites (`cities` vs. `cities`+`neg_cities`; `larger_than` vs. `larger_than`+`smaller_than`).
- **Probing technique:** LR vs. MM (and its IID covariance-corrected variant) vs. CCS — similar accuracy, very different causal strength.
- **Layer / token position:** truth structure emerges in early-middle layers, later for complex statements; using the final content token instead of the punctuation token sometimes makes top PCs capture the token itself (e.g., clusters for statements ending in "China") rather than truth.
- **Controls:** the `likely` dataset (probable vs. improbable nonfactual text) rules out the "it's just plausibility" explanation; correlation analysis of truth vs. log-probability (r from −.89 to +.95 across datasets) supports the same point.
- **Baselines:** calibrated few-shot prompting (weak) and oracle LR trained on the test set.

## 5. Limitations and open questions (from §7.1)

1. **Only simple, uncontroversial statements.** The authors cannot disambiguate "truth" from closely related features such as "commonly believed" or "verifiable" (Levinstein & Herrmann's critique). Nothing here directly licenses a general-purpose lie detector for messy, real-world text.
2. **Only the LLaMA-2 family** was studied; results may not transfer to all LLMs.
3. **Open puzzles:** Why are mass-mean interventions from the `likely` dataset so effective even though those probes classify truth poorly? One puzzle the paper's prose raises dissolves against its own Table 2: the text wonders why MM trained on `cities` + `neg_cities` performs poorly for the 70B model, but Table 2 (v3) shows it scoring 0.81/0.95 there — including the highest 70B false→true number in the table — so MM on `cities` + `neg_cities` does not in fact perform poorly at 70B; the prose puzzlement appears to be stale wording left over from an earlier version of the paper. Also, training on opposites helped causal interventions for `cities` but not for `larger_than`, even though it improved `larger_than` probes' classification accuracy — more to understand about how training data shapes truth directions.

## 6. Key figures and tables

- **Figure 1** — PCA visualizations of LLaMA-2-70B representations for the true/false datasets: true and false statements separate linearly in the top two PCs.
- **Figure 2** — Activation-patching heatmap (LLaMA-2-13B, `cities`): change in log P(TRUE) − log P(FALSE) per token position × layer, revealing the three groups (a)/(b)/(c) of causally implicated hidden states.
- **Figure 3** — Alignment and misalignment across datasets: (a) `cities`' first PC separates other datasets too; (b) `larger_than`/`smaller_than` separate antipodally in 13B but along a common direction in 70B (7B clusters by surface tokens like "eighty"); (c) statements and their negations separating along near-orthogonal directions (layer 12).
- **Figure 4** — Toy illustration of logistic regression's weakness: with a non-orthogonal interfering feature, LR finds a max-margin separator that misses the true feature direction; difference-in-means does not.
- **Figure 5** — Probe generalization results: (a) average accuracy improves when training on statements + their opposites; (b) accuracy by model scale and training set — generalization is high for 13B/70B, poor for `likely`-trained probes.
- **Table 1** — The dataset suite: 8 curated datasets, 3 uncurated datasets, and the `likely` control, with sizes from 354 to 31,960 rows.
- **Table 2** — Normalized indirect effects (NIEs) for the causal interventions on `sp_en_trans`: mass-mean directions are the most causally effective in 7 of 8 conditions.

## 7. Why this matters (one-paragraph version)

The paper is a careful piece of data mining *inside* a neural network: build clean, controlled datasets; use dimensionality reduction (PCA) to visualize representation geometry; compare simple classifiers (LR, LDA-flavored difference-in-means, CCS) not just on accuracy but on cross-dataset transfer; and then validate with causal interventions. Its headline message — at sufficient scale, LLMs linearly represent the truth of factual statements along a direction that simple difference-in-mean probes recover best — is both a foundational result for interpretability and a cautionary tale: a probe can be accurate without pointing at the feature the model actually uses.
