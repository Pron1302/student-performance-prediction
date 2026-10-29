"""Creates a realistic synthetic student dataset (with deliberate data-quality issues
so that the cleaning step in the notebook has real work to do)."""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
n = 1000

df = pd.DataFrame({
    "Student_ID": [f"S{1000+i}" for i in range(n)],
    "Gender": rng.choice(["Male", "Female"], n),
    "Age": rng.integers(15, 20, n),
    "Study_Hours_Per_Week": np.clip(rng.normal(14, 6, n), 1, 40).round(1),
    "Attendance_Percentage": np.clip(rng.normal(82, 12, n), 40, 100).round(1),
    "Previous_Exam_Score": np.clip(rng.normal(62, 15, n), 15, 100).round(1),
    "Assignments_Completed_Pct": np.clip(rng.normal(75, 18, n), 10, 100).round(1),
    "Sleep_Hours": np.clip(rng.normal(7, 1.2, n), 3.5, 10).round(1),
    "Internet_Access": rng.choice(["Yes", "No"], n, p=[0.85, 0.15]),
    "Parental_Education": rng.choice(["High School", "Bachelor", "Master", "PhD"], n, p=[0.40, 0.35, 0.18, 0.07]),
    "Extracurricular": rng.choice(["Yes", "No"], n, p=[0.55, 0.45]),
    "Tutoring": rng.choice(["Yes", "No"], n, p=[0.30, 0.70]),
})

edu_bonus = df["Parental_Education"].map({"High School": 0, "Bachelor": 1.5, "Master": 3, "PhD": 4})
score = (
    0.9 * df["Study_Hours_Per_Week"]
    + 0.25 * df["Attendance_Percentage"]
    + 0.30 * df["Previous_Exam_Score"]
    + 0.12 * df["Assignments_Completed_Pct"]
    + 1.0 * (df["Sleep_Hours"] - 7).clip(-3, 1)
    + edu_bonus
    + 3 * (df["Tutoring"] == "Yes")
    + 2 * (df["Internet_Access"] == "Yes")
    + rng.normal(0, 4, n)
)
score = (score - score.min()) / (score.max() - score.min()) * 70 + 25   # scale to ~25-95
df["Final_Score"] = score.round(1)
df["Performance"] = pd.cut(df["Final_Score"], bins=[0, 50, 70, 100],
                           labels=["Low", "Average", "High"]).astype(str)

# ---- inject realistic data-quality problems ----
for col, frac in [("Study_Hours_Per_Week", .05), ("Attendance_Percentage", .06),
                  ("Sleep_Hours", .04), ("Parental_Education", .05),
                  ("Previous_Exam_Score", .03), ("Internet_Access", .02)]:
    df.loc[rng.choice(n, int(n * frac), replace=False), col] = np.nan

df.loc[rng.choice(n, 6, replace=False), "Attendance_Percentage"] = [105, 110, 120, 101, 150, 999]  # invalid
df.loc[rng.choice(n, 4, replace=False), "Study_Hours_Per_Week"] = [80, 95, 120, 100]                # outliers
df.loc[rng.choice(n, 30, replace=False), "Gender"] = ["male", "FEMALE", "female", "MALE", "Male "] * 6  # inconsistent text
df = pd.concat([df, df.sample(15, random_state=1)], ignore_index=True)                              # duplicates
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("data/student_performance.csv", index=False)
print(df.shape)
