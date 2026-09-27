A Shared-Trunk Multi-Task Network for Joint Cost, Schedule, and Risk Prediction in DoD Construction Contracts

ABSTRACT

Construction cost overruns, schedule delays, and overall project risk are often treated as separate prediction problems, even though they can be related to the same project characteristics. This project develops a shared-trunk multi-task neural network to predict cost growth, schedule overrun, and overall risk using one shared representation. A joint loss is used so that all three prediction tasks contribute to the training of the shared layers.

The model is trained and tested using real U.S. Department of Defense construction contract data from FPDS-NG, compiled by Hendrix et al. and available through Mendeley Data (DOI: 10.17632/yk4s7pdsvk.1). A temporal split is used, with 37,326 projects awarded during FY2009–2014 used for training and 19,374 projects awarded during FY2015–2017 used for testing.

The multi-task network performs better than the equivalent single-task neural network on all three tasks. This supports the hypothesis that joint training can help a neural network learn information shared between cost, schedule, and risk outcomes. However, the single-task Random Forest performs better than the neural models on the two classification tasks. This is consistent with previous findings that tree-based models can perform strongly on small and medium-sized tabular datasets.

The study also covers several data-related issues found in the government procurement data, including transaction-level linkage, right-censoring, and the heavy-tailed nature of the cost-growth target. The steps used to handle these issues are described along with the model and evaluation results.

1. INTRODUCTION

Cost overruns, schedule delays, and increased risk in construction contracts are often related. A contract with problems in its scope, competition, or other project characteristics may face problems in more than one of these areas. However, these outcomes are commonly predicted using separate models. Training each model independently can miss information that is shared between the targets.

Multi-task learning provides a way to address this issue. In this approach, a shared trunk learns a common representation of the project data, while separate task-specific heads are used for cost, schedule, and risk prediction. Since the tasks share model parameters, the training signals from each task contribute to the shared layers.

This project implements a shared-trunk multi-task network and compares it with single-task Random Forest and single-task MLP models. The models are evaluated using real DoD construction contract data rather than synthetic data.

The main finding is that the multi-task network performs better than the single-task neural network on all three tasks. This supports the main hypothesis that joint training can improve the neural network's ability to learn information shared across the three outcomes.

Random Forest performs better than the multi-task network on the two classification tasks. This result is consistent with previous work on tabular data, including Grinsztajn et al. (2022), and is included as part of the model comparison rather than treated as a failure of the multi-task approach.

2. DATA

2.1 Source

The dataset used in this project is the DoD FPDS-NG (Federal Procurement Data System – Next Generation) construction contract dataset compiled by Hendrix et al. at the Air Force Institute of Technology.

The dataset is available through Mendeley Data under DOI 10.17632/yk4s7pdsvk.1 and is released under the CC BY 4.0 license. The data is real government procurement data rather than synthetic data and should be cited directly when it is reused.

2.2 From Raw Transactions to Project-Level Rows

The original FPDS-NG export contains 360,029 transaction-level rows. A row represents either a base contract award or a later modification to that contract.

There is no single primary key that can reliably identify a contract across all agencies. Therefore, prepare_real_data.py creates a composite key using the Contracting Agency ID, Contracting Office ID, and PIID.

This produced 88,201 unique project keys. Among the projects containing multiple rows, 80% span three years or less between their first and last transaction. This is consistent with most groups representing actual modification chains rather than unrelated contracts being grouped together. The linkage is not guaranteed to be perfect and is therefore treated as a limitation of the data-processing method.

The transaction records are then aggregated into one row per project. This produces 73,298 project-level rows, which are stored in projects_clean.csv so that the aggregation process does not have to be repeated for every experiment.

2.3 Right-Censoring

The original data file was extracted from FPDS-NG in mid-2020. Projects awarded during the later years of the dataset had less time to receive contract modifications before the extraction date.

For example, the mean number of transactions per project decreases from approximately 22 for FY2009 awards to approximately 1.7 for FY2019 awards. A project with few modifications could therefore appear to have lower risk simply because there was less time for later transactions to be recorded.

Projects awarded after FY2017 are therefore excluded using MAX_MATURE_FISCAL_YEAR in config.py. They are not used for either training or testing.

This removes 16,598 rows and leaves 56,700 mature projects for the analysis.

2.4 Targets

Three targets are created for each project.

Cost Growth Ratio

Cost growth is calculated as:

(current total value − base value) / base value

The resulting ratio is winsorized at the 1st and 99th percentiles. It is then transformed using a sign-preserving log transformation:

sign(x) · log1p(|x|)

The transformed values are standardized before being used for training. The raw cost-growth ratio has a heavy-tailed distribution because some government contracts contain very large modifications. The transformation is used to make the target more suitable for MSE-based training.

Schedule Overrun

Schedule overrun is treated as a binary classification problem. A project receives a value of 1 when its ultimate completion date is more than 30 days later than its current completion date. Otherwise, the value is 0.

A regression model for the exact number of schedule-slip days produced a strongly negative R² on this data. Schedule prediction was therefore changed to a binary classification task, in the same way that the risk target was already defined.

Overall Risk

A project is classified as having overall risk when it meets either of the following conditions:

• Cost growth is greater than 10%.
• Schedule overrun is greater than the defined threshold.

Projects that meet neither condition are assigned a risk value of 0.

2.5 Features

The model uses nine features:

• Base contract value
• Planned duration in days
• Number of offers received
• Military branch
• NAICS code
• PSC code
• Contract type
• Extent competed
• Place-of-performance state

The categorical variables are frequency-encoded. This is particularly useful for NAICS and PSC codes because they contain hundreds of different categories. Frequency encoding keeps the feature space smaller than one-hot encoding.

2.6 Train/Test Split

The data is divided using a temporal split rather than a random split.

Training data consists of projects awarded during FY2009–2014, giving 37,326 projects.

Testing data consists of projects awarded during FY2015–2017, giving 19,374 projects.

The temporal split represents a more realistic prediction setting because the model is trained on earlier projects and evaluated on projects from later years. A random split could allow projects from the same time period to appear in both training and testing data.

3. METHOD

3.1 Shared-Trunk Multi-Task Network

The multi-task network is implemented in mtl_model.py.

The model contains a shared trunk followed by three task-specific heads.

The shared trunk consists of two dense layers with ReLU activation functions. The layers contain 64 and 32 units and take the nine-dimensional feature vector as input.

The cost head contains a dense layer with ReLU activation followed by a linear output layer. It is trained using mean squared error on the transformed cost-growth target.

The schedule head contains a dense layer with ReLU activation followed by a sigmoid output layer. It is trained using binary cross-entropy.

The risk head has the same structure as the schedule head and is also trained using binary cross-entropy.

All three heads use the same 32-dimensional representation produced by the shared trunk.

The total training loss is calculated as:

L = w_cost × MSE_cost + w_sched × BCE_sched + w_risk × BCE_risk

The weights used in the final experiment are:

(w_cost, w_sched, w_risk) = (0.5, 1.5, 1.5)

The classification tasks receive higher weights than the cost task because the cost-growth prediction remains more difficult and noisy.

During training, gradients from the three task heads are combined at the shared trunk and propagated through the common layers. This allows the network to learn information that is useful across the three prediction tasks instead of training three independent models.

The network uses the Adam optimizer with a learning rate of 2e-3 for each parameter. The batch size is 1024 and training runs for 150 epochs.

Implementation Note

The network is implemented directly in NumPy. The forward pass, backpropagation, and Adam optimizer are implemented manually. PyTorch and TensorFlow were not available in the sandbox used for development.

The implementation follows the same basic architecture and joint-loss approach that would be used in a PyTorch model. The current code can therefore be ported to PyTorch if GPU acceleration is required later.

3.2 Baselines

Three single-task baselines are trained independently using the same nine features.

Random Forest

A separate Random Forest model is trained for each task. The cost target uses a regression model, while schedule and risk use classification models.

Single-Task MLP

The single-task MLP uses scikit-learn MLPRegressor and MLPClassifier models. It provides a neural-network comparison to the multi-task model and helps separate the effect of joint training from the effect of using a neural network.

3.3 Evaluation

Cost growth is evaluated using MAE, RMSE, and R². The predictions are inverse-transformed before these metrics are calculated.

Schedule overrun and overall risk are evaluated using accuracy, F1 score, and AUC. A probability threshold of 0.5 is used when calculating accuracy and F1 score.

4. RESULTS

The complete console output from the final experiment is stored in real_results_log.txt. The same results are shown visually in figures/results_heatmap.png.

Table 1. Model Performance

Task                         Multi-Task Net    Single-Task RF    Single-Task MLP
Cost growth (R²)             0.034             -0.015            -0.113
Schedule overrun (AUC)       0.652              0.707             0.648
Overall risk (AUC)           0.654              0.702             0.651

Figure 1. Model performance heatmap

Figure 2. Multi-task training loss curve

Training converged gradually. The joint loss decreased from 2.52 at epoch 1 to 2.08 by epoch 140 and had largely flattened by approximately epoch 100.

4.1 Multi-Task Network vs. Single-Task Neural Network

The multi-task network performs better than the single-task MLP on all three tasks.

For cost growth, the multi-task network obtains an R² of 0.034 compared with -0.113 for the single-task MLP.

For schedule overrun, the multi-task network obtains an AUC of 0.652 compared with 0.648.

For overall risk, the multi-task network obtains an AUC of 0.654 compared with 0.651.

Both models use the same input features and similar neural-network structures. The main difference is that the multi-task model shares its representation and trains the three tasks together. The results therefore provide evidence that joint training improves the neural model on this dataset.

4.2 Neural Networks vs. Random Forest

Random Forest performs better than the multi-task network on both classification tasks. Its AUC is 0.707 for schedule overrun and 0.702 for overall risk, compared with 0.652 and 0.654 for the multi-task network.

For cost growth, the multi-task network obtains an R² of 0.034, compared with -0.015 for Random Forest.

The results therefore differ between the tasks. Random Forest performs better on the two classification problems, while the multi-task network performs better on the cost-growth regression task.

This result is consistent with the broader literature on tabular data, including the findings reported by Grinsztajn, Oyallon, and Varoquaux in their 2022 NeurIPS paper on tree-based models and deep learning.

5. DISCUSSION

The results support a narrower claim than saying that neural networks outperform Random Forest on this dataset. The comparison between the multi-task network and the single-task MLP shows that joint training improves the neural network across all three tasks.

At the same time, Random Forest achieves higher AUC values on the schedule and risk classification tasks. Therefore, the multi-task network does not provide the highest absolute performance for every task.

The cost-growth R² of the multi-task network is also modest at 0.034. Cost growth in real government contracts can be affected by factors that are not included in the nine features used in this study. These may include scope changes, price increases, weather, and disputes.

The purpose of the experiment is to compare different training strategies using the same real-world feature set. It is not intended to claim maximum possible predictive accuracy.

6. LIMITATIONS

Linkage Imperfection

The composite key made from Contracting Agency ID, Contracting Office ID, and PIID is not guaranteed to be unique across all agencies. Some grouped records may therefore represent different contracts, while some records from the same contract may not be grouped perfectly.

Right-Censoring Exclusion

Projects awarded after FY2017 are excluded because they had less time to accumulate modification records. This removes 16,598 rows, or about 23% of the aggregated dataset, and means that the model is not evaluated on the most recent projects in the original source file.

Narrow Feature Set

The model uses structured numeric and categorical fields only. Free-text information, including the Description of Requirement field, is not used in this version. Such fields may contain additional useful information.

NumPy Implementation

The multi-task network is implemented manually in NumPy instead of using a standard deep-learning framework. This is sufficient for the current dataset size but would be less suitable for much larger datasets or more complex architectures without moving to a framework such as PyTorch or TensorFlow.

Threshold Choices

The 10% cost-growth threshold and 30-day schedule-overrun threshold were selected as reasonable defaults. They were not systematically tested across different values or validated through domain-expert judgment.

7. FUTURE WORK

8. Tune the Joint-Loss Weights

The current weights are (0.5, 1.5, 1.5) and were selected without a systematic parameter search. A small grid search could be used to test different combinations of task weights.

2. Add Text-Derived Features

The Description of Requirement field is currently excluded. Features derived from this text, such as TF-IDF or keyword indicators, could be added in a future version.

3. Increase Shared-Trunk Capacity

The current shared trunk uses 64 and 32 units. A larger shared representation could be tested now that the basic multi-task architecture has been evaluated on real data.

REFERENCES

Hendrix, R. et al. DoD Construction Contracts Dataset (FPDS-NG). Air Force Institute of Technology. Mendeley Data. DOI: 10.17632/yk4s7pdsvk.1. CC BY 4.0.

Grinsztajn, L., E. Oyallon, and G. Varoquaux. “Why do tree-based models still outperform deep learning on tabular data?” NeurIPS 2022 Datasets and Benchmarks Track.
