# The Geometry of Truth — CMPE-255 Short Story

**Student:** Bernie Miao
**Course:** CMPE-255 Data Mining, Fall 2026 (San Jose State University)
**Assignment:** Short story assignment — an explainer of one approved research paper, with a Medium article, a slide deck, and a narrated video.

## The paper

Samuel Marks & Max Tegmark (2023; v3 2024), **"The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets,"** arXiv:2310.06824
https://arxiv.org/abs/2310.06824

Topic approval: this paper is Bernie's approved topic in the instructor's selection spreadsheet —
[Fall 2026 CMPE-255 Short Story Selection spreadsheet](https://docs.google.com/spreadsheets/d/1dgFf_VLUsn00PcexC8l3uXd5lKOSDMXS66x70DCNL_g)

## Artifacts

| Artifact | Where |
|---|---|
| Medium article | PENDING — link added when published |
| Slide deck (SlideShare) | PENDING — link added when published |
| Narrated video (10–15 min, slides + voiceover) | `geometry_of_truth_video.mp4` in this repo (added when rendered) |
| Slide deck files | `geometry_of_truth_deck.pptx` / `geometry_of_truth_deck.pdf` in this repo |
| Article source | `article_draft.md` |
| Video narration script | `video_script.md` |
| Paper breakdown (notes) | `paper_breakdown.md` |

## What the paper shows, in one paragraph

Marks and Tegmark built small, unambiguous true/false datasets (cities, comparisons, translations, and their negations), then probed the internal activations of LLaMA-2 models. At sufficient scale, true and false statements separate linearly in activation space — a single "truth direction" — and simple probes trained on one dataset generalize to very different ones. Their simplest probe, mass-mean (difference-in-means), matches logistic regression and CCS on accuracy but is far more causal: adding or subtracting its direction flips the model's own true/false judgments in 7 of 8 out-of-distribution intervention conditions. The claims are limited to simple, uncontroversial statements and the LLaMA-2 family.
