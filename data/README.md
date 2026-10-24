# Data Directory Documentation

This directory manages both the raw clinical ICU databases and the extracted feature matrices.

## 1. Directory Structure

- `data/raw/`: Contains symlinks or local references to the original clinical databases:
  - `eicu-collaborative-research-database-demo-2.0.1/` (2,520 ICU stays, PhysioNet ODbL license)
  - `mimic-iv-clinical-database-demo-2.2/` (140 ICU stays, PhysioNet Open Access license)
- `data/processed/`:
  - `cohort_features.csv`: The complete 148-column feature matrix extracted for the 24-hour observation cohort ($N = 1,623$ stays). Contains baseline demographics, periodic vital statistics, aperiodic blood pressure dynamics, laboratory biomarkers, trend velocities, volatility metrics, missingness indicators, and target labels.
  - `sample_trajectories.csv`: Cached 5-minute periodic vital observations for sample test stays, enabling instantaneous interactive exploration in the Streamlit dashboard.

## 2. Access Conditions & Credentialing

- **Demo Databases:** The demo versions included in this repository are publicly available under open licenses for educational and research prototyping.
- **Full Clinical Databases:** Accessing the complete eICU (200,000+ stays) or MIMIC-IV (50,000+ patients) databases requires completing the CITI Data or Specimens Only Research human-subjects training course and obtaining an approved credentialed researcher account on PhysioNet.
