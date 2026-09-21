# STC-RGCN

**Spatio-Temporal Constrained Relational Graph Convolutional Network** for urban
origin–destination (OD) prediction.

STC-RGCN is a multi-task relational graph convolutional network that jointly predicts,
for every OD pair:

- the **trip-purpose distribution** — a 15-dimensional probability vector over trip
  purposes, trained with KL divergence; and
- the **total flow volume** — a non-negative regression target, trained with an L1 or
  Tweedie loss.

Instead of discrete relation types, the relational convolutions consume **continuous
4-mode transport shares** (bus / driving / subway / non-vehicle) as edge relation
weights. Node features fuse a **POI embedding** (Doc2Vec over each grid cell's POI
composition) with a **trajectory embedding** (Word2Vec over constrained random-walk
trajectories synthesized from behavioural constraints). The two tasks can be balanced
adaptively with [GradNorm](https://arxiv.org/abs/1711.02257).

---

## Installation

Python **3.9+**.

```bash
# PyTorch first, matching your CUDA version (see https://pytorch.org/get-started/)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then the package
pip install -e .
```

Optional extras:

```bash
pip install -e ".[preprocessing]"   # gensim, geopandas, scikit-learn
pip install -e ".[baselines]"       # xgboost, lightgbm, scikit-learn
pip install -e ".[dev]"             # ruff
```

`torch_geometric` is a core dependency and installs as a normal wheel.
`torch_scatter` is **not** required — the scatter operations are written with
`index_add_`, so there is no wheel to match against your exact torch/CUDA build.
GMEL additionally needs [`dgl`](https://www.dgl.ai/pages/start.html), which has its
own install matrix.

---

## Quick start

Train the full multi-task model on the shipped sample dataset:

```bash
stc-rgcn-train --variant full --task multi --patience 20
```

The console script is a thin wrapper around `stc_rgcn/run.py`, which is equally
runnable as a module:

```bash
python -m stc_rgcn.run --variant full --task multi --patience 20
```

Every run writes a self-contained directory:

```
runs/full-multi/
├── config.json       the resolved arguments
├── best_model.pt     best-validation checkpoint
├── predictions.txt   test predictions, 34 columns
└── metrics.json      final test metrics
```

Then score it:

```bash
stc-rgcn-eval score --folder runs/full-multi
```

Other variants:

```bash
# Tweedie flow loss with log-normalized flows
stc-rgcn-train --variant full_tweedie --task multi --log-transform --patience 20

# single-task ablations
stc-rgcn-train --variant purpose --task purpose --patience 20
stc-rgcn-train --variant base    --task multi   --patience 20

# adaptive loss balancing
stc-rgcn-train --variant full --task multi --gradnorm --patience 20
```

---

## Repository structure

The package root holds what defines the method; `utils/` holds the pieces it is
built out of.

```
stc-rgcn/
├── pyproject.toml · README.md · LICENSE · requirements.txt
├── stc_rgcn/
│   ├── run.py                  training entry point (stc-rgcn-train)
│   ├── config.py               model variants, task selection, default paths
│   ├── data.py                 dataset loading and split parsing
│   ├── model.py                the StcRgcn network
│   ├── training.py             train step, evaluation, prediction writing
│   ├── utils/
│   │   ├── schema.py           on-disk dataset layout, purpose codes, file names
│   │   ├── graph.py            full-graph construction and subgraph sampling
│   │   ├── layers.py           MeanAggregationConv, RelationWeightedConv
│   │   ├── losses.py           KL / L1 / Tweedie losses
│   │   ├── gradnorm.py         GradNormBalancer
│   │   ├── metrics.py          flow and distribution metrics (one shared copy)
│   │   ├── transforms.py       LogMinMaxScaler for the flow target
│   │   └── early_stopping.py   EarlyStopper and the monitored-metric rules
│   ├── evaluation/             scoring + ablation contribution analysis
│   ├── preprocessing/          the five-stage feature pipeline
│   └── baselines/              comparison models and their shared helpers
└── data/
    ├── sample/                 shipped synthetic dataset (60 zones)
    └── constraints/            behavioural constraint tables
```

---

## Model variants

`--variant` selects which inputs and which prediction heads are built.

| Variant | Node features | Mode weights | Heads | Notes |
|---|:---:|:---:|---|---|
| `base` | – | – | purpose + flow | plain GCN ablation |
| `feature` | ✓ | – | purpose + flow | |
| `relation` | – | ✓ | purpose + flow | |
| `purpose` | ✓ | ✓ | purpose | single-task, early-stops on KL |
| `flow` | ✓ | ✓ | flow | single-task, early-stops on MAE |
| `full` | ✓ | ✓ | purpose + flow | **default** |
| `full_tweedie` | ✓ | ✓ | purpose + flow | Tweedie flow loss, power auto-estimated |

`--task` (`purpose` / `flow` / `multi`) selects which loss drives training and early
stopping when a variant has both heads. `multi` uses the summed multi-task loss.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `data/sample` | dataset directory |
| `--feature-file` | `features.txt` | node feature file inside `--data-dir` |
| `--output-dir` | `runs` | parent directory for run directories |
| `--run-name` | `<variant>-<task>` | run directory name |
| `--variant` | `full` | model variant (table above) |
| `--task` | `multi` | which loss drives training |
| `--num-bases` | `4` | basis rank of the relation-weighted convolution |
| `--dropout` | `0.1` | dropout in the convolution stack and heads |
| `--tweedie-power` | auto | Tweedie power; estimated from training flows when omitted |
| `--epochs` | `3000` | maximum epochs |
| `--lr` | `2e-4` | Adam learning rate |
| `--sample-size` | `3000` | edges sampled per training step |
| `--graph-split-ratio` | `0.5` | fraction of sampled edges used for message passing and supervision |
| `--reg-weight` | `0.005` | weight of the parameter-norm penalty; `0` disables it |
| `--clip-grad-norm` | `1.0` | gradient-clipping max norm (*not* the GradNorm algorithm) |
| `--log-transform` | off | train on `log(flow)` rescaled to `[0, 1]` |
| `--gradnorm` | off | balance the two task losses with GradNorm |
| `--eval-every` | `2` | validate every N epochs |
| `--patience` | `20` | validations without improvement before stopping |
| `--delta` | `0.01` | minimum improvement that counts |
| `--eval-on-gpu` | off | keep validation on the training device (default: score the full graph on CPU) |
| `--device` | `auto` | `auto`, `cpu`, `cuda` or `cuda:<index>` |
| `--seed` | none | seed for python, numpy and torch |

---

## Data format

All model inputs live in one directory (default `data/sample`), tab-separated, no
headers.

| File | Format |
|---|---|
| `entities.dict` | one row per zone: `id\tzone_name`, ids `0..N-1` |
| `relations_mode.dict` | 4 rows: `id\tmode` (bus, driving, subway, non_vehicle) |
| `relations_purpose.dict` | 15 rows: `id\tpurpose_code` |
| `{train,valid,test}.txt` | 40 columns (see below) |
| `features.txt` | one row per entity (row *i* = entity *i*); the first 2 columns are metadata and are dropped, so the shipped 146-column file yields 144 features |
| `grid_distance.csv` | *optional*, real OD distances in km — used by the baselines |

The 40 split columns, 0-indexed:

```
[0]      origin entity string
[1..4]   4 transport-mode shares   — continuous, used as edge relation weights
[5]      destination entity string
[6..20]  15 trip-purpose probabilities (sum to 1)
[21..24] 4 transport-mode flows    — unused; the shares above are what the model reads
[25..39] 15 trip-purpose flows     — total flow is their sum
```

Probabilities are stored to 3 decimals, so a row's sum may drift from 1 by a few
thousandths; the loader accepts up to `1e-2` and renormalizes.

> **Legacy names.** Releases before 1.0 used `{train,valid,test}_with_flows.txt` and
> `relations_method.dict`. Both spellings are accepted, so an existing private dataset
> needs no renaming.

### Shipped dataset is synthetic

The real study-area dataset is not public. The repository ships a **generated** sample
with the same file names and layout: 60 zones and 500 / 100 / 100 train / valid / test
rows. No real OD flow, POI embedding or zone location appears in this repository.

### Trip-purpose codes

`relations_purpose.dict` uses pinyin abbreviations of the Chinese survey categories.
The same Chinese strings key the behavioural constraint tables under `data/constraints/`.

| # | Code | Chinese | English |
|---|---|---|---|
| 0 | `sx` | 上学 | school |
| 1 | `dxsk` | 大学上课 | university class |
| 2 | `jypx` | 教育培训 | education and training |
| 3 | `gw` | 购物 | shopping |
| 4 | `shfw` | 生活服务 | daily services |
| 5 | `xxyy` | 休闲娱乐 | leisure and entertainment |
| 6 | `ms` | 美食 | dining |
| 7 | `sw` | 商务 | business |
| 8 | `gz` | 工作 | work |
| 9 | `hj` | 回家 | going home |
| 10 | `yl` | 医疗 | healthcare |
| 11 | `tqfy` | 探亲访友 | visiting friends and relatives |
| 12 | `tccx` | 同城出行 | intra-city travel |
| 13 | `kccx` | 跨城出行 | inter-city travel |
| 14 | `ly` | 旅游 | tourism |

This table is available programmatically as `stc_rgcn.utils.schema.PURPOSE_LABELS`.

### OD distances

The baselines need a distance per OD pair. Sources are tried in order, and every run
prints which one it used:

1. `grid_distance.csv` in the data directory — real OD distances in km, not shipped.
   Both the original Chinese headers (`起点网格ID,终点网格ID,distance`) and English
   aliases (`origin,destination,distance`) are accepted.
2. Coordinates embedded in the entity names — `entities.dict` stores names like
   `HYID8133500|17797000`, and the two numbers are parsed as grid coordinates.
3. A constant fallback, which makes any distance-dependent model degenerate.

---

## Prediction file format

Every model **and** every baseline writes the same 34-column layout, which is what
lets `stc-rgcn-eval score` read a whole results directory without knowing which model
produced which file:

```
origin  destination  pred_prob_<15 codes>  true_prob_<15 codes>  pred_flow  true_flow
```

Columns belonging to a head a variant does not have are left empty.

---

## Evaluation

```bash
# score a folder of prediction files (mean ± std, grouped by model)
stc-rgcn-eval score --folder runs/baselines

# module-contribution analysis over an ablation table
stc-rgcn-eval contribution --csv ablation_results.csv
```

`score` groups files that differ only by a trailing run number (`_run_2`, `-3`, `_4`),
so repeated runs of one model are aggregated automatically.

---

## Feature pipeline

Five stages, run in order. Each is a console script. Stages 1–3 need raw inputs that
are **not** part of this repository (a POI shapefile, trip records, processed grid
data), so they cannot be re-run as shipped; the constraint tables and the resulting
sample dataset are included.

| Stage | Console script | Input → Output |
|---|---|---|
| 1 | `stc-rgcn-poi-embedding` | POI shapefile → 72-d grid POI embeddings |
| 2 | `stc-rgcn-build-dataset` | trip records + entities + POI features → entity dict and splits |
| 3 | `stc-rgcn-synthesize` | grid data + constraint tables → synthesized trajectories |
| 4 | `stc-rgcn-traj-embedding` | trajectories → 72-d grid trajectory embeddings |
| 5 | `stc-rgcn-fuse-features` | POI + trajectory embeddings → fused `features.txt` |

### Behavioural constraints

Stage 3 reads six tables from `data/constraints/`, named after the three mechanism
groups of the paper:

| Group | Files | Switched by |
|---|---|---|
| `PPC` — trip purpose priors | `ppc_generation.csv` (POI condition, minimum stay, time windows, per-period factors), `ppc_transitions.csv` (forbidden purpose transitions) | `ConstraintConfig.enable_ppc` |
| `DCC` — destination choice | `dcc_poi_weights.csv` (purpose → POI affinity), `dcc_distance_decay.csv` (per-purpose distance decay) | `ConstraintConfig.enable_dcc` |
| `TMC` — trip mode choice | `tmc_poi_weights.csv` (mode → POI affinity), `tmc_mode_availability.csv` (per-mode max distance and time window) | `ConstraintConfig.enable_tmc` |

`ConstraintConfig` exposes the three progressive variants compared in the paper:
`ppc_only()` (`GCN_fun-PPC`), `ppc_dcc()` (`GCN_fun-PPC-DCC`) and `full()`
(`STC-RGCN w/o Rel`).

```bash
stc-rgcn-synthesize --grid-data <grid_data.csv> --constraints full
```

```python
from stc_rgcn.preprocessing.trajectory_synthesis import (
    ConstraintConfig, TrajectorySynthesizer,
)

synthesizer = TrajectorySynthesizer(
    grid_data_path="grid_data.csv",
    config=ConstraintConfig.full(),
)
trajectories = synthesizer.synthesize_all()
```

---

## Baselines

| Model | Console script | Needs beyond the core install |
|---|---|---|
| Gravity (GM-O / GM-P / GM-E), radiation | `stc-rgcn-baseline-physics --model gm-o` | – (numpy only; `scikit-learn` is used for the fit when present, with a `numpy.linalg.lstsq` fallback otherwise) |
| XGBoost / LightGBM / Random Forest | `stc-rgcn-baseline-trees --model xgboost` | `xgboost` / `lightgbm` / `scikit-learn` |
| SI-GCN | `stc-rgcn-baseline-sigcn` | – |
| GMEL | `stc-rgcn-baseline-gmel` | `dgl` |

All baselines read `data/sample` by default and write to `runs/baselines/`. `--split`
selects the evaluation split (`test` by default, falling back to `valid` when the test
file is absent). SI-GCN predicts total flow only; its purpose distribution is the
training-set flow-weighted average.

Shared loading, feature, metric, IO and CLI helpers live in
`stc_rgcn.baselines.common`, and the 40-column and 34-column layouts come from
`stc_rgcn.utils.schema`, so no baseline re-implements either.

**Attribution.** `baselines/gmel.py` is a single-file consolidation of the open-source
[GMEL](https://github.com/jackmiemie/GMEL) implementation (Liu et al., AAAI 2020,
*Learning Geo-Contextual Embeddings for Commuting Flow Prediction*); the data pipeline
was replaced with project adapters and the GBRT fine-tuning stage removed.
`baselines/sigcn.py` is a PyTorch re-implementation of SI-GCN (Yao et al., 2020,
*Spatial Origin-Destination Flow Imputation Using Graph Convolutional Neural
Networks*), whose [reference code](https://github.com/susurrant/flow-imputation) is
TensorFlow 1.x. The gravity, radiation and tree-ensemble baselines were written for
this project.

---

## Using the package as a library

```python
from stc_rgcn import StcRgcn, Variant, build_full_graph, load_dataset
from stc_rgcn.training import train_step

dataset = load_dataset("data/sample", Variant.FULL)
model = StcRgcn(
    variant=Variant.FULL,
    num_entities=dataset.num_entities,
    num_relations=dataset.num_relations,
    feature_dim=dataset.feature_dim,
)

losses = train_step(
    model, dataset.train,
    num_entities=dataset.num_entities,
    num_relations=dataset.num_relations,
    sample_size=3000, graph_split_ratio=0.5, reg_weight=0.005,
    node_features=dataset.node_features,
)
losses.total.backward()
```

---

## Development

```bash
pip install -e ".[dev]"
ruff check .
```

Conventions worth keeping to:

- **The package root holds what defines the method** — `config`, `data`, `model`,
  `training` and the `run` entry point. `utils/` holds the pieces they are built out
  of: graph construction, layers, losses, GradNorm, metrics, the flow scaler, early
  stopping and the dataset schema.
- **`Variant` in `config.py` is the single place that decides what a variant does.**
  Everything else reads its capability properties rather than testing the name, so a
  new variant usually means one enum member plus one property update.
- **A new baseline** reads the dataset through `stc_rgcn.baselines.common` and writes
  results with `write_predictions`, keeping the 34-column layout uniform so
  `stc-rgcn-eval score` can score it alongside every other model.
- Code, comments and documentation are English; data keeps its original Chinese domain
  vocabulary, referenced through named constants rather than inline literals.

---

## Citation

```bibtex
@misc{stcrgcn2026,
  title  = {STC-RGCN: Spatio-Temporal Constrained Relational Graph Convolutional
            Network for OD Flow and Trip Purpose Prediction},
  author = {Zhuo, Xingyu},
  year   = {2026},
  note   = {\url{https://github.com/zqiexin-maker/STC-RGCN}}
}
```

## License

Released under the [MIT License](LICENSE). Vendored baseline code (GMEL, SI-GCN)
remains subject to its upstream licenses.
