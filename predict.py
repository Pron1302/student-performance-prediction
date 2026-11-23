"""Student Performance Prediction System - command line version.
Usage:  python predict.py
Needs the saved models in the models/ folder (created by the Week 2 notebook)."""
import os
import joblib
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
reg = joblib.load(os.path.join(BASE, "models", "final_regression_model.joblib"))
clf = joblib.load(os.path.join(BASE, "models", "final_classifier.joblib"))
features = joblib.load(os.path.join(BASE, "models", "feature_order.joblib"))
EDU_MAP = {"high school": 0, "bachelor": 1, "master": 2, "phd": 3}


def ask_number(prompt, low, high):
    while True:
        try:
            value = float(input(prompt))
            if low <= value <= high:
                return value
        except ValueError:
            pass
        print(f"  Please enter a number between {low} and {high}.")


def ask_choice(prompt, options):
    while True:
        value = input(prompt).strip().lower()
        if value in options:
            return value
        print("  Please choose one of:", ", ".join(options))


def predict(study_hours, previous_score, attendance, assignments, sleep, tutoring, education):
    row = pd.DataFrame([{
        "Study_Hours_Per_Week": study_hours,
        "Previous_Exam_Score": previous_score,
        "Attendance_Percentage": attendance,
        "Assignments_Completed_Pct": assignments,
        "Sleep_Hours": sleep,
        "Tutoring": 1 if tutoring == "yes" else 0,
        "Parental_Education": EDU_MAP[education],
    }])[features]
    score = float(np.clip(reg.predict(row)[0], 0, 100))
    category = clf.predict(row)[0]
    proba = dict(zip(clf.classes_, clf.predict_proba(row)[0]))
    return score, category, proba


def main():
    print("=== Student Performance Prediction System ===\n")
    study = ask_number("Study hours per week (0-60): ", 0, 60)
    prev = ask_number("Previous exam score (0-100): ", 0, 100)
    att = ask_number("Attendance % (0-100): ", 0, 100)
    assign = ask_number("Assignments completed % (0-100): ", 0, 100)
    sleep = ask_number("Sleep hours per night (0-12): ", 0, 12)
    tutoring = ask_choice("Has tutoring? (yes/no): ", ["yes", "no"])
    edu = ask_choice("Parental education (high school/bachelor/master/phd): ", list(EDU_MAP))

    score, category, proba = predict(study, prev, att, assign, sleep, tutoring, edu)
    print("\n--- Result ---")
    print(f"Predicted final score : {score:.1f} / 100")
    print(f"Predicted category    : {category}")
    print("Probabilities         :", {k: f"{v:.0%}" for k, v in proba.items()})


if __name__ == "__main__":
    main()
