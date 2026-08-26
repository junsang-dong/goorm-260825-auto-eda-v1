"""Generate a synthetic demo dataset (employee salary) for trying out the EDA app.

Run: python sample_data/generate_sample.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 500

department = rng.choice(["Engineering", "Sales", "Marketing", "HR", "Finance"], size=N, p=[0.35, 0.25, 0.15, 0.1, 0.15])
education = rng.choice(["Bachelor", "Master", "PhD"], size=N, p=[0.55, 0.35, 0.10])

experience_years = np.clip(rng.normal(7, 4, N), 0, 30)
age = np.clip(experience_years + rng.normal(27, 3, N), 21, 65)
hours_per_week = np.clip(rng.normal(42, 6, N), 20, 70)
num_projects = rng.poisson(4, N)
performance_score = np.clip(rng.normal(70, 15, N), 0, 100)
distance_from_office_km = np.clip(rng.exponential(8, N), 0.5, 60)
remote_days_per_week = rng.integers(0, 4, N)

edu_bonus = pd.Series(education).map({"Bachelor": 0, "Master": 8000, "PhD": 18000}).to_numpy()
dept_bonus = pd.Series(department).map(
    {"Engineering": 12000, "Sales": 4000, "Marketing": 2000, "HR": 0, "Finance": 6000}
).to_numpy()

noise = rng.normal(0, 5000, N)
salary = (
    35000
    + experience_years * 3200
    + performance_score * 250
    + edu_bonus
    + dept_bonus
    + noise
)

random_noise_1 = rng.normal(0, 1, N)
random_noise_2 = rng.integers(0, 100, N)

df = pd.DataFrame({
    "employee_id": [f"E{i:05d}" for i in range(N)],
    "department": department,
    "education": education,
    "age": age.round(1),
    "experience_years": experience_years.round(1),
    "hours_per_week": hours_per_week.round(1),
    "num_projects": num_projects,
    "performance_score": performance_score.round(1),
    "distance_from_office_km": distance_from_office_km.round(2),
    "remote_days_per_week": remote_days_per_week,
    "random_noise_1": random_noise_1.round(3),
    "random_noise_2": random_noise_2,
    "salary": salary.round(0),
})

# Inject some missing values to exercise the profiling/importance missing-value handling.
for col, frac in [("performance_score", 0.03), ("distance_from_office_km", 0.05), ("education", 0.02)]:
    idx = rng.choice(df.index, size=int(len(df) * frac), replace=False)
    df.loc[idx, col] = np.nan

out_path = __file__.replace("generate_sample.py", "sample_dataset.csv")
df.to_csv(out_path, index=False)
print(f"Wrote {len(df)} rows to {out_path}")
