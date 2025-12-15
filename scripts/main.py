import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

# ==============================
# Load and clean data
# ==============================
df = pd.read_csv("data/NBA 2023-2024 Dataset (Combined) V2 - Sheet1.csv")

# Convert attendance per game into numeric
df["Attend./G"] = df["Attend./G"].replace(",", "", regex=True).astype(float)

# Keep only selected columns
cols = ["3PA", "BLK", "PTS", "W", "Attend./G", "All-NBA Player"]
data = df[cols].copy()

# ==============================
# Correlation matrix
# ==============================
print(data.corr(numeric_only=True))

# ==============================
# Histograms in one figure
# ==============================
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
variables = ["W", "3PA", "BLK", "PTS", "Attend./G"]
titles = ["Wins", "3-Point Attempts", "Blocks", "Points", "Attendance"]

for ax, var, title in zip(axes.flatten(), variables + [None], titles + [None]):
    if var is not None:
        ax.hist(data[var], bins=20, color="skyblue", edgecolor="black")
        ax.set_title(title)
    else:
        ax.axis("off")

plt.tight_layout()
plt.show()

# ==============================
# Boxplots in one figure
# ==============================
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, var, title in zip(axes.flatten(), variables + [None], titles + [None]):
    if var is not None:
        sns.boxplot(y=data[var], ax=ax, color="lightgreen")
        ax.set_title(title)
    else:
        ax.axis("off")

plt.tight_layout()
plt.show()

# ==============================
# Typical values (mean, median, sd)
# ==============================

print(data.describe())

# ==============================
# Regression models
# ==============================

# Target variable
y = pd.to_numeric(data["Attend./G"], errors="coerce")

# Predictors
X = data.drop(columns=["Attend./G"])

# Encode categorical into dummy (0/1)
X = pd.get_dummies(X, drop_first=True)

# Force all predictors to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Drop any rows with missing values in y or X
X = X.loc[y.notna()]
y = y.loc[y.notna()]

# Add constant for intercept
X = sm.add_constant(X)

# Fit model
full_model = sm.OLS(y.astype(float), X.astype(float)).fit()
print(full_model.summary())



# Partial regression plots
sm.graphics.plot_partregress_grid(full_model)
plt.show()

# Example confounding test
model = smf.ols("Q('Attend./G') ~ PTS + W + I(W**2) + PTS:W + PTS:I(W**2)", data=data).fit()
print(model.summary())

# Shapiro-Wilk test
shapiro_test = stats.shapiro(full_model.resid)
print("Shapiro-Wilk test:", shapiro_test)

# ==============================
# Collinearity (VIF)
# ==============================
# Drop constant column for VIF
X_no_const = X.drop(columns=["const"])

# Make absolutely sure everything is float
X_no_const = X_no_const.apply(pd.to_numeric, errors="coerce").astype(float)

# Drop any columns with NaNs after conversion
X_no_const = X_no_const.dropna(axis=1, how="any")
vif_data = pd.DataFrame()
vif_data["Variable"] = X_no_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_no_const.values, i)
                   for i in range(X_no_const.shape[1])]
print(vif_data)

# ==============================
# Fitted values vs jackknife residuals
# ==============================
from statsmodels.stats.outliers_influence import OLSInfluence

influence = OLSInfluence(full_model)
stud_resid = influence.resid_studentized_external

plt.scatter(full_model.fittedvalues.astype(float), stud_resid.astype(float))
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Fitted values")
plt.ylabel("Studentized Residuals")
plt.title("Jacknife student residuals")
plt.show()


# ==============================
# Collinearity check (VIF instead of olsrr)
# ==============================
X_no_const = X.drop(columns="const")

# Force numeric and drop any problematic cols
X_no_const = X_no_const.apply(pd.to_numeric, errors="coerce").astype(float)
X_no_const = X_no_const.dropna(axis=1, how="any")

vif_data = pd.DataFrame()
vif_data["Variable"] = X_no_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_no_const.values, i)
                   for i in range(X_no_const.shape[1])]
print(vif_data)

# ==============================
# Standardize variables
# ==============================
data["3PA"] = pd.to_numeric(data["3PA"], errors="coerce")
data["PTS"] = pd.to_numeric(data["PTS"], errors="coerce")

data["3PA"] = (data["3PA"] - data["3PA"].mean()) / data["3PA"].std()
data["PTS"] = (data["PTS"] - data["PTS"].mean()) / data["PTS"].std()

# Refit full model with standardized variables
X_std = data.drop(columns=["Attend./G"])
X_std = pd.get_dummies(X_std, drop_first=True)
X_std = X_std.apply(pd.to_numeric, errors="coerce").astype(float)
X_std = sm.add_constant(X_std)

full_model_std = sm.OLS(y.astype(float), X_std.astype(float)).fit()

# VIF again
X_std_no_const = X_std.drop(columns="const")
X_std_no_const = X_std_no_const.apply(pd.to_numeric, errors="coerce").astype(float)
X_std_no_const = X_std_no_const.dropna(axis=1, how="any")

vif_data_std = pd.DataFrame()
vif_data_std["Variable"] = X_std_no_const.columns
vif_data_std["VIF"] = [variance_inflation_factor(X_std_no_const.values, i)
                       for i in range(X_std_no_const.shape[1])]
print(vif_data_std)

final_model = smf.ols("Q('Attend./G') ~ W + I(W**2)", data=data).fit()

print(final_model.summary())
