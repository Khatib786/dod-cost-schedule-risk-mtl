# Shared-Trunk Multi-Task Network for Cost, Schedule, and Risk Prediction

This project develops a shared-trunk multi-task neural network for predicting **cost growth**, **schedule overrun**, and **overall risk** in construction contracts. The model is trained and evaluated using real U.S. DoD FPDS-NG construction contract data. The dataset contains 37,326 projects awarded during FY2009–2014 for training and 19,374 projects awarded during FY2015–2017 for testing. A temporal split is used between the training and testing data.

## Results

The multi-task network performs better than the equivalent single-task neural network on all three tasks. This supports the use of a shared model with a joint loss for these prediction tasks.

For the two classification tasks, the single-task Random Forest achieves higher AUC values than both neural network models. This shows the difference between the neural network and tree-based models on the tabular dataset.

| Task                   | Multi-Task Net | Single-Task RF | Single-Task MLP |
| ---------------------- | -------------: | -------------: | --------------: |
| Cost growth (R²)       |      **0.034** |         -0.015 |          -0.113 |
| Schedule overrun (AUC) |          0.652 |      **0.707** |           0.648 |
| Overall risk (AUC)     |          0.654 |      **0.702** |           0.651 |

The complete report, including data engineering, transaction linkage, right-censoring, target transformations, model architecture, related work, and discussion of the results, is available in **PAPER.md**.

## Repository Structure

```text
├── README.md                 ← project overview
├── PAPER.md                  ← full report (methodology, results, discussion)
├── src/                      ← source code
│   ├── config.py             # column mappings, thresholds, and split configuration
│   ├── prepare_real_data.py  # converts raw transactions into project-level rows
│   ├── data_pipeline.py     # cleaning, encoding, scaling, and temporal split
│   ├── mtl_model.py         # multi-task neural network (NumPy)
│   ├── baselines.py         # single-task Random Forest and MLP
│   ├── run_experiment.py    # main experiment entry point
│   ├── make_figures.py      # regenerates figures from the results log
│   └── projects_clean.csv   # cleaned and aggregated dataset (73,298 rows)
├── figures/                  ← loss curve and results heatmap
├── requirements.txt
└── real_results_log.txt     ← console output from the final run
```

## Data Source

The project uses the DoD FPDS-NG construction contract dataset from Hendrix et al., Air Force Institute of Technology.

**Mendeley Data DOI:** 10.17632/yk4s7pdsvk.1
**License:** CC BY 4.0

The dataset is real rather than synthetic. Anyone reusing the dataset should cite the original source directly.

## Reproduction

Install the required packages using:

```bash
pip install -r requirements.txt
```

Then move into the source directory and run the experiment:

```bash
cd src
python run_experiment.py --data projects_clean.csv --prepared
python make_figures.py
```

The second command regenerates the figures from the logged results.

The `projects_clean.csv` file contains the aggregated project-level dataset and is included in the `src/` directory. This allows the experiment to run without downloading additional data.

The dataset can also be rebuilt from the original FPDS-NG transaction export using `src/prepare_real_data.py`. The training process uses a fixed seed through `mtl_model.MultiTaskNet.fit(..., seed=42)`, while the scikit-learn baseline models may introduce some stochastic variation. As a result, repeated runs should reproduce the reported results closely rather than necessarily producing identical values.

More details about the model architecture, joint loss, data-cleaning decisions, and results are provided in **PAPER.md**.

## License

The code is released under the **MIT License**.

The underlying dataset is licensed under **CC BY 4.0** by its original authors. If the dataset is reused, the original data source should be cited directly.
