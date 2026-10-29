# Student Performance Prediction System

Internship mini project (AI / ML) — **Week 1: Data Understanding & Analysis** (done).

## Structure
- `generate_dataset.py` – creates the synthetic dataset
- `data/student_performance.csv` – raw dataset (1,015 rows incl. duplicates)
- `data/cleaned_student_data.csv` – cleaned dataset
- `data/model_ready_data.csv` – selected features + targets, input for Week 2
- `Week1_Data_Understanding_and_Analysis.ipynb` – cleaning, EDA, visualizations, feature selection
- `visualizations/` – 9 saved charts

## Week 1 results
- Removed 15 duplicates, standardised Gender text, fixed invalid attendance/study-hour values, filled missing values (median / mode).
- Strongest drivers of final score: study hours, previous exam score, attendance, assignments completed.
- Selected features: Study_Hours_Per_Week, Previous_Exam_Score, Attendance_Percentage, Assignments_Completed_Pct, Sleep_Hours, Tutoring, Parental_Education.

## Run
```
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
python generate_dataset.py      # optional – dataset already included
jupyter notebook Week1_Data_Understanding_and_Analysis.ipynb
```
