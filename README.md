# StoryForge — Does structured tagging improve stylistic control in a fine-tuned LLM?

A data-centric pipeline and controlled experiment: take a raw corpus of interactive-fiction
scenes, build a multi-facet tagging system with a closed vocabulary, and train two QLoRA
adapters — one on plain scenes, one on the same scenes conditioned on control tags — to
measure whether tagging buys controllable, higher-quality stylistic generation.

The interesting work here is **not** the fine-tuning (two runs, one config). It's the
**data and the experimental design**: how scenes are selected, deduplicated, and diversified;
how they're consistently annotated by an LLM against a curated vocabulary; and how the A/B is
set up so the two models differ in exactly one variable.

---

## The question

Style LoRAs are usually trained on raw text. This project asks: if you annotate each scene
with structured control tags (situation, tone, setting, dynamics, …) and train on
`instruction → scene`, does the model become *steerable* — can you ask for
`Tone: tender, Setting: docks` and get it — in a way a plain-text LoRA can't? To answer
honestly you need a controlled comparison, not a vibe check.

---

## Pipeline

```
raw SQLite scenes
      │
      ▼
[1] profile + filter      keyword filter, length gating, per-scene scoring
      │
      ▼
[2] dedup + diversify     MinHash/LSH near-dup removal, MMR selection
      │
      ▼
[3] plain corpus + manifest   scene = one example (no splitting); manifest records
      │                        scene_id + split + text for later alignment
      ▼
[4] tag the SAME scenes   discover → curate closed vocabulary → closed classification
      │                    (LLM tagger, guided-JSON decoding, ~210 labels / 7 facets)
      ▼
[5] tagged corpus          {prompt: instruction-from-tags, completion: scene}
      │                     aligned 1:1 to the plain corpus, same train/val split
      ▼
[6] train two QLoRAs       identical hyperparameters + seed; only the corpus differs
      │
      ▼
[7] compare + export       checkpoint comparison, merge, GGUF for local (ollama) use
```

---

## Design decisions worth noting

- **Scene = one training example (no chunk splitting).** Makes the tag↔example mapping 1:1
  and avoids splice artifacts from stitching partial scenes. Scenes are length-gated
  (≤ ~1700 est. tokens) so scene + tag-prompt fits a 2048 window.

- **Tagging is *classification against a curated vocabulary*, not free extraction.**
  Workflow: an open **discovery** pass harvests candidate descriptors → the vocabulary is
  **curated by hand** → a **closed** pass assigns only in-vocabulary labels. Guided-JSON
  decoding guarantees valid, schema-conforming output. This keeps labels consistent and
  learnable instead of a long tail of one-off phrasings.

- **Clean A/B via a shared manifest.** Both corpora are built from the same selected scenes
  in the same train/val split (aligned by `scene_id`), with the same random seed, so the
  trained models differ in exactly one thing: the presence of tags.

- **Regularization against overfitting.** Attention-only LoRA targets (q/k/v/o — not MLP,
  which memorizes surface content), `alpha == r`, dropout 0.05, one epoch, lr 1e-4, with
  validation enabled to watch the train/val gap. (A prior 2–3 epoch run overfit and broke
  scene logic; this configuration did not — val loss decreased monotonically to a plateau.)

- **Things that were tried and dropped.** An earlier *coherence-scoring* stage was removed
  after a control test (injected-splice detection) showed it caught 0/10 — it added no
  signal, so it was cut rather than kept for appearance.

---

## Honest limitations

- **The A/B is not perfectly sterile.** The plain corpus trains as raw `text`; the tagged
  corpus trains as `instruction` with prompt masking. So the two models differ in *format*
  as well as *tags*. This is a comparison of two **approaches** (raw style LoRA vs.
  tag-conditioned LoRA), not a pure ablation of tags alone. A stricter version would wrap the
  plain corpus in a generic `Write a scene.` instruction so only the tag content varies.

- **Evaluation is currently qualitative.** Results are judged by reading generations. The
  clear next step is a quantitative metric (e.g. a classifier checking whether generations
  obey the requested tags, or blind win-rate between the two models).

- **Modest scale.** ~1900 scenes — enough for style, not "big data".

---

## Results (qualitative)

Training was healthy for both runs: train and validation loss fell together and val loss
plateaued without rising, i.e. the model learned style without the overfitting seen in
earlier multi-epoch attempts. Peak VRAM ~7.5 GB (8B QLoRA, 4-bit, 2048 window on a 12 GB
RTX 3060). Whether the tagged model is meaningfully more *steerable* than the plain model is
assessed by generation, and best conditioned with the exact tag phrasing seen in training
(the vocabulary labels), optionally combined with free-text elaboration.

> Note: plain vs. tagged **loss values are not directly comparable** — the tagged model is
> trained with prompt masking (loss over the scene only, given the tags), the plain model over
> the whole string. Compare by generation, not by loss.

---

## Repository layout

Data pipeline (`dataset_builder/`):
- `build_corpus.py` — select/filter/score/dedup/diversify scenes → plain corpus + manifest
- (analysis / keyword / metrics / ranking / splitter helpers)

Tagging (`lora_trainer/`):
- `tag_discover.py` — open discovery pass → candidate vocabulary
- `vocab.txt` — hand-curated closed vocabulary (~210 labels / 7 facets)
- `tag_closed.py` — closed classification (transformers + guided JSON), checkpoint/resume + ETA
- `tag_closed_vllm.py` — same, on vLLM (Linux/WSL) for a fast batched pass
- `build_tagged_corpus.py` — manifest + tags → `{prompt, completion}` tagged corpus
- `tag_viewer.html` — drag-drop viewer to eyeball tag assignments

Training / export:
- `config.yaml` (+ `config.plain.yaml`, `config.tagged.yaml` via `extends`) — one base config,
  two thin overrides so the A/B differs only in name + corpus
- `train.py`, `model.py`, `dataset.py`, `trainer.py` — Unsloth QLoRA training
- `compare_checkpoints.py` — run one prompt across all checkpoints to pick the best stage
- `export.py` — merge LoRA + GGUF export

---

## Running it

**Build both corpora**
```bash
# plain corpus + manifest (scene = example, length-gated)
python build_corpus.py coc2.db --analysis dataset_analysis.csv \
    --no-split --max-tokens 1700 --max-examples 0 --out corpus

# tag EXACTLY the corpus scenes (100% coverage, no drift)
python tag_closed.py coc2.db --vocab vocab.txt --ids-from corpus.manifest.jsonl
#   (or tag_closed_vllm.py on Linux/WSL for a fast batched pass)

# tagged corpus, same scenes / same splits
python build_tagged_corpus.py --manifest corpus.manifest.jsonl \
    --tags tag_closed.jsonl --out corpus_tagged
```

**Train the two adapters (identical config, only the corpus differs)**
```bash
python train.py --config config.plain.yaml
python train.py --config config.tagged.yaml
python runs.py                     # table of runs for comparison
```

**Export for local use (ollama)**
```bash
python export.py --run tagged-1900-r16-e1 --merge
# then convert the merged model to GGUF with llama.cpp and load via an ollama Modelfile
```

---

## Environment

Two separate environments (do not mix):
- **Training** (Windows): Unsloth + PyTorch (CUDA build) + bitsandbytes. See
  `requirements-train.txt`.
- **Fast tagging / GGUF** (Linux/WSL): vLLM + llama.cpp. See `requirements-infer.txt`.

> PyTorch must be installed from the CUDA index, not from requirements directly, e.g.
> `pip install torch --index-url https://download.pytorch.org/whl/cu130` (match your driver).
> On Windows, install Unsloth with `--no-deps` so it doesn't replace the CUDA torch build with
> a CPU one. vLLM is Linux/WSL only.

---

## What this project demonstrates

Data-centric ML: corpus construction and curation, LLM-as-annotator with quality control
(closed vocabulary + guided decoding), controlled experiment design with an isolated variable,
overfitting-aware fine-tuning validated against a held-out set, and an end-to-end path from raw
data to a deployable quantized model — plus the judgment to cut a component (coherence scoring)
that measurably didn't work.
