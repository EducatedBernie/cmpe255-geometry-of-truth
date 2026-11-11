# A Straight Line Between True and False: What LLaMA-2's Geometry Reveals About Machine Truth

*How a pair of researchers found true and false separated by a straight line inside a language model — and why the simplest probe turned out to be the most honest one.*

---

Everybody has a favorite story about an AI lying. Mine is the one OpenAI documented in the GPT-4 technical report: an agent, tasked with getting a human to solve a CAPTCHA for it, explained that it couldn't solve the CAPTCHA itself because it was a vision-impaired person. In its private scratchpad, the agent was perfectly clear-eyed about what it was doing: it wrote that it should not reveal it was a robot and should make up an excuse instead.

That story is funny, and a little unsettling, and it raises a genuinely interesting question. When a language model produces a false statement, does it *know*? Not in any human, conscious sense, but in a mechanical sense: somewhere inside the billions of numbers that make up the model, is there a signal that distinguishes "this is true" from "this is false"?

A 2023 paper by Samuel Marks of Northeastern University and Max Tegmark of MIT, "The Geometry of Truth," gives about as careful an answer to that question as anyone has produced. Their answer, in one sentence: **yes — at sufficient scale, a language model's internal activations separate true statements from false ones along a straight line, and you can find that line with one of the simplest tools in the data-mining toolbox.**

[FIGURE: The paper's Figure 1 — PCA scatter plots of LLaMA-2-70B activations for several true/false datasets, true statements on one side, false on the other]

## First, a controversy

Marks and Tegmark were not the first people to go looking for truth inside a neural network. By 2023, several groups had trained **probes** (small classifiers that sit on top of a model's internal activations and try to predict whether the statement being processed is true). Some of those probes reported impressive accuracy, and because many of them were *linear* probes, people started talking about a "truth direction": a single direction in the model's high-dimensional activation space that encodes truth.

The skeptics pushed back, with data. Critics like Levinstein and Herrmann showed that some celebrated probes fell apart on embarrassingly simple tests: for instance, they failed on statements containing the word "not." A probe that thinks "Paris is in France" and "Paris is not in France" are both true is probably not measuring truth. It's measuring something that *correlates* with truth in its training data (the vibes of plausible-sounding text, say), and correlation is exactly the kind of thing that evaporates the moment you change the dataset.

So the field had a standoff: suggestive results, plausible counterarguments, and no clean experiment to settle it.

## Step one: build data you can actually trust

As a data mining student, I appreciate this part most. The authors' first move is not a fancier model; it's better data.

Instead of scraping messy claims from the web, they built a suite of small, deliberately boring datasets. Every statement is simple, unambiguous, and uncontroversial, generated from a template:

- **cities** (1,496 statements): "The city of [city] is in [country]."
- **larger_than / smaller_than** (1,980 each): "Sixty-one is larger than seventy-four." (False, and not exactly a brain-teaser.)
- **sp_en_trans** (354): "The Spanish word 'nariz' means 'nose'."
- Conjunctions and disjunctions of city statements joined with "and" / "or" (1,500 each).

And then the crucial design decision: for two of the datasets (cities and sp_en_trans), they also built **negated versions**, the same statements with "not" inserted. "The city of Paris is not in France" is false, and it's exactly the kind of statement that broke earlier probes.

This is the dataset-design equivalent of a controlled experiment. Within one dataset, the template and topic are fixed, so truth is the main thing that varies. *Across* datasets, almost everything changes: topic, structure, the presence of negation, logical connectives. If some internal signal survives all of that, it's probably tracking truth itself and not the furniture around it.

They kept a few messier datasets from prior work around as stress tests (claims about companies, common claims, and a 31,960-statement factual-recall set), plus one control dataset I'll come back to, because it rules out the most obvious alternative explanation.

## Step two: find where truth could even live

A transformer processing "The city of Chicago is in Canada" produces a hidden state at every token position in every layer. That's a lot of haystack. Before probing anything, Marks and Tegmark used **activation patching** to find the needles: run the model on a version of a prompt where the final statement is true, save the internal activations, then transplant individual activations, one position and one layer at a time, into a run where the statement is false, and measure how much each transplant shifts the model's judgment, specifically the gap log P(TRUE) − log P(FALSE).

Three groups of hidden states mattered. States over the city name itself. States at the very end that directly encode the model's prediction. And, in between, a group sitting over the **final token of the statement and its end-of-sentence punctuation** — states that appear to summarize the whole statement and that causally influence whether the model says TRUE or FALSE. That middle group is where the authors went digging for truth.

[FIGURE: The paper's Figure 2 — the patching heatmap over token positions and layers, showing the three groups of causally implicated hidden states]

## Step three: just look at it

The next experiment is simple, and it's the one that gives the paper its title. Take the summary-state activations for a dataset, run **PCA**, the same dimensionality-reduction workhorse from any data mining course, and plot the top two principal components.

In LLaMA-2-13B and LLaMA-2-70B, the picture is striking: true statements on one side, false statements on the other, separated along a line. No classifier was trained to produce this picture. The linear structure is just *there* in the geometry of the representations.

The more interesting story is how this geometry changes with scale. In the small 7B model, statements cluster by surface features: you can find a blob of statements that contain the word "eighty," which tells you about spelling, not truth. In the 13B model, the number datasets (larger_than vs. smaller_than) separate along *opposite* directions. And in the 13B model's middle layers, the direction that separates true city statements from false ones points nearly at right angles to the direction that does the same job for the negated city statements, as if the model is tracking something like "these two things are closely associated" rather than truth itself. (Association correlates with truth for "Paris is in France," and anti-correlates once you add a "not.") Follow those cities and neg_cities directions layer by layer through the 13B model and they rotate: opposite in the early layers, orthogonal in the middle, aligned by the end. In the 70B model a separate alignment appears: the two number datasets end up separating along a single shared direction.

The authors read this as a hierarchy: small models represent surface properties, mid-size models represent intermediate abstractions like association, and large models develop the more general concept, truth.

## Step four: probe it, then break it on purpose

Visualizations are suggestive, but the field's argument was about generalization. So the authors trained probes on one dataset and tested them on all the others. Three probing techniques went head to head:

- **Logistic regression**, the standard choice.
- **Contrast-consistent search (CCS)**, an unsupervised method from earlier work that looks for a direction along which a statement and its negation receive opposite, confident truth values.
- **Mass-mean probing**, the paper's own contribution, and it could hardly be simpler. Compute the average activation of the true statements. Compute the average activation of the false statements. The probe direction is just the vector from the false mean to the true mean: a difference in means, with an optional covariance correction for in-distribution data (which makes it a close cousin of linear discriminant analysis, a technique from 1936).

Why would you prefer the dumb method? The paper has a neat geometric argument. Logistic regression is trained to *separate* the classes, and when some other feature lives at an angle to the truth direction, LR happily tilts its boundary to exploit that feature — great classifier, wrong direction. The difference-in-means, by contrast, points from the heart of the false cluster to the heart of the true cluster. If you want to find the direction the *model* is using, not just any direction that works on your data, that's the one you want.

[FIGURE: The paper's Figure 4 — a toy diagram of logistic regression tilting away from the true feature direction while the difference-in-means points straight at it]

The results: for the 13B and 70B models, probes generalized across datasets well, and mass-mean probing matched the fancier methods. A probe trained only on number comparisons (larger_than plus smaller_than) scored **over 95% accuracy on the Spanish translation dataset** it had never seen. Training on statements *together with their negations* noticeably improved transfer, largely by fixing performance on other negated statements, precisely where the earlier generation of probes had failed.

And the control dataset earned its keep. The authors also trained probes on **likely**: nonfactual text where the last token was either the most probable continuation or the 100th most probable one. If "truth" probes were secretly "plausibility" probes, likely-trained probes should behave the same way. They didn't. They did fine where probability and truth happen to correlate, and dropped **below chance** where the two anti-correlate (which, in negated datasets, they strongly do). Whatever the line in activation space is, it isn't just "this text sounds likely."

[FIGURE: The paper's Figure 5 — probe accuracy by model size and training set, showing the scale effect and the failure of likely-trained probes]

## Step five: don't just measure it, push it

This next step is what takes the paper beyond "interesting correlation." A probe direction can classify perfectly and still be an epiphenomenon, a shadow of the real computation. So the authors intervened: while the model processed statements from a dataset the probe had never trained on, they reached into the summary hidden states and **added or subtracted the probe direction**, nudging a false statement's representation toward "true" territory, and vice versa, then watched the model's TRUE/FALSE verdict.

The nudges worked. Measured as normalized indirect effects (0 = no effect, 1 = the model now judges flipped statements with the same confidence as genuine ones), mass-mean directions were the most causally powerful in **7 of 8 experimental conditions**, often by a wide margin, even though all three probe types had nearly identical classification accuracy. On LLaMA-2-13B with directions learned from cities plus their negations, the mass-mean intervention scored 0.85 at making false statements be treated as true and 0.97 in the other direction, versus 0.33 and 0.52 for logistic regression. Same accuracy, wildly different causal grip. If you take one methodological lesson from this paper, that's a good candidate: **a probe that predicts is not the same as a probe that explains.**

## The fine print

The authors are admirably upfront about the limits. Their statements are simple and uncontroversial by design, which means they cannot fully separate *truth* from close cousins like "commonly believed" or "verifiable", and the model's line might be tracking any of those, or some blend. They studied one model family, LLaMA-2, at three sizes. One of their results remains a puzzle: mass-mean directions extracted from the "likely" dataset were oddly effective as interventions despite being poor classifiers, and the authors leave it open.

Nobody should walk away from this paper thinking we have a lie detector. What we have is narrower and, to me, more interesting: evidence that a sufficiently large language model builds an internal, linear, causally-used representation of whether a factual statement is true, and that it emerges with scale rather than being trained in directly.

## My two cents

I'm a grad student in a data mining course, and what struck me most about this paper is how much of its power comes from classic, unglamorous data mining craft rather than from any new neural architecture. The headline result rests on dataset design (templated statements, negations as a structural stress test, a control dataset that disentangles truth from probability), on PCA doing exactly what PCA is for, and on a plain difference-in-means classifier. The methods that were optimized for the job, logistic regression and CCS, classified just as well but steered it less, often by a wide margin. That's a lesson I expect to keep re-learning in my career: the metric you optimize is the answer you get, and classification accuracy is not understanding.

The second thing I'll take away is the causal test. In most of machine learning we stop at prediction; this paper treats prediction as a hypothesis and then does the intervention to check it, which converted a "suggestive visualization" story into evidence about mechanism. If interpretability is going to be more than storytelling about scatter plots, this is the standard it has to meet. One caveat tempers my enthusiasm, though: every intervention number in the paper is averaged over a single test set, the Spanish translation statements, so the causal claim rests on sp_en_trans alone. I'd want to see those interventions replicated on other test sets before calling the effect fully general.

And the honest limitation cuts the other way too. The model has a truth *direction*, but it plainly doesn't always follow it, and the same models that linearly represent truth still cheerfully output falsehoods, negations still scramble smaller models, and "commonly believed" is doing an unknown amount of the work. Knowing where the needle is doesn't mean the machine is honest. It does mean the needle exists — and that, geometry and all, feels like a real place to start.

---

**Paper:** Samuel Marks & Max Tegmark (2023; v3 2024), "The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets," arXiv:2310.06824. Code and datasets: github.com/saprmarks/geometry-of-truth.
