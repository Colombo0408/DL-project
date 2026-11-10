# Gym Workout Recognition & Reps Counting with IMU + HBC

Deep Learning course final project — **Fabio Colombo**, University of Pavia.

A PyTorch re-implementation of [*Hybrid CNN-Dilated Self-attention Model Using Inertial and Body-Area Electrostatic Sensing for Gym Workout Recognition, Counting, and User Authentification*](https://arxiv.org/abs/2503.06311), comparing an inertial sensor (IMU) with the human body capacitance (HBC) sensor, alone and combined, on the [RecGym](https://www.kaggle.com/datasets/zhaxidelebsz/10-gym-exercises-with-615-abstracted-features/data) dataset.

## Pipeline

1. **Data**: RecGym, 10 subjects, 20 Hz, IMU (3-axis acc + 3-axis gyro) + HBC channel, sensor worn at wrist / leg / pocket.
2. **Windowing**: 4 s windows (80 samples), 2 s overlap, label by majority vote, per subject.
3. **Model**: CNN block → multi-head self-attention → dilated convolution, with sliding-window ensembling (post-fusion scheme, Fig. 5 of the paper).
4. **Training**: Leave-One-Subject-Out CV, inverse-frequency class weights, early stopping on validation accuracy.
5. **Evaluation**: accuracy, macro F1, classification reports, confusion matrices, ROC/PR curves.
6. **Repetition counting**: FFT low-pass smoothing + peak detection (`peakutils`) on Acc, Gyro and HBC signals.

## Results (FAST_MODE run: 5 subjects, 8 classes)

LOSO mean accuracy / macro F1:

| Position | HBC | IMU | IMU + HBC |
|---|---|---|---|
| Wrist  | 0.708 / 0.615 | 0.968 / 0.952 | **0.977 / 0.967** |
| Leg    | 0.733 / 0.633 | 0.779 / 0.656 | **0.881 / 0.796** |
| Pocket | 0.623 / 0.400 | **0.810 / 0.691** | 0.741 / 0.590 |

Repetition counting accuracy (mean):

| Position | Acc | Gyro | HBC |
|---|---|---|---|
| Wrist  | 0.891 | 0.859 | **0.909** |
| Leg    | 0.888 | 0.871 | **0.891** |
| Pocket | **0.882** | 0.877 | 0.876 |

## Repository structure

```
colab/
  Gym_Workout_Recognition_ID_580655.ipynb   # main notebook (final version)
data/
  df_processed.parquet, metadata.json       # preprocessed dataset + label map
  hbc_model_<position>_<cap|imu|combine>.pth # best checkpoint per configuration
  all_results.pkl                           # LOSO predictions / probabilities
  counting_df.csv, diag_df.csv              # repetition-counting results
  training.txt                              # training log
images/                                     # figures used in the notebook
```

## How to run

1. Open `colab/Gym_Workout_Recognition_ID_580655.ipynb` in Google Colab (GPU runtime).
2. Place the project folder at `MyDrive/Deep_Learning_Project` (outputs are saved to `data/`).
3. The dataset is downloaded from Kaggle (needs a Kaggle API token), or reloaded from `df_processed.parquet`.
4. Set `FAST_MODE = False` (Section 1.3) for the full run: all 10 subjects and 12 classes.

## Differences from the paper

- **FAST_MODE** reduces subjects, classes (drops Riding, Running, StairClimber, Walking), the `Null` class (30% kept), and uses mixed precision. The results above come from this mode.
- Fewer epochs (150 / patience 20 vs. 1000 / 100).
- Conceptual port from Keras to PyTorch, not a layer-for-layer clone. No average pooling in the CNN block, matching the authors' code rather than the paper text.
- Ground truth for repetition counting is **assumed** (3×10 reps per segment, from the paper's protocol), since the dataset has no rep labels.
- User authentication (Sec. IV-E) is not implemented.

*For research and educational purposes only.*
