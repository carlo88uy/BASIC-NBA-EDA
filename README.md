# NBA 2023–2024 Statistical Exploratory Data Analysis

A curated dataset and supporting scripts for NBA 2023–2024 season statistics, created as part of a class assignment.

## Project Overview

This project analyzes factors influencing **NBA game attendance** during the 2023–2024 season using team-level performance data from Basketball Reference. The study examines whether team success, style of play, and star presence meaningfully affect average fan attendance per game.

We wanted to find, what aspects of NBA team performance most significantly influence average game attendance?

Specifically, this study investigates whether:
- Team wins
- Points per game
- Three-point attempts
- Blocks per game
- Presence of an All-NBA player  

have a measurable impact on attendance.

---

## Repository Contents

Below is a summary of the main files and directories:

| Path | Description |
|------|-------------|
| `Visualizations + Output/`| Results created after running either of the scripts within the 'scripts' folder|
| `NBA 2023-2024 Dataset (Combined) V2 - Sheet1.csv` | Main dataset containing player and team statistics for the 2023–2024 NBA season. |
| `scripts/` | Folder containing Python and/or R scripts used for processing, analysis, or visualization of the dataset. |
| `LICENSE` | MIT License file. |
| `README.md` | This project documentation file. |

---

## Dataset Description

The dataset includes a wide variety of game and player statistics for the 2023–2024 NBA season, such as:

- **Points (PTS)**  
- **Assists (AST)**  
- **Rebounds (REB / OREB / DREB)**  
- **Field Goal, 3-Point, and Free Throw Stats**  
- **Turnovers, Steals, Blocks**  
- **Team identifiers and metadata for each game**  

This consolidated dataset is suitable for exploratory data analysis (EDA), modeling, and visualization.

---

## Installation & Setup

To work with the dataset locally, follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/carlo88uy/NBA.git
   cd NBA

2. **Install python dependencies**
  ```bash
  pip install numpy pandas matplotlib seaborn
  ```

3. **Load the dataset**

---

## Methodology

- Exploratory Data Analysis (EDA) using correlation matrices and partial regression plots  
- Multiple linear regression modeling  
- Backward elimination for model selection  
- Diagnostic checks for:
  - Normality
  - Collinearity
  - Influential points  

Collinearity between points and three-point attempts was addressed through standardization.

---

## Key Findings

- **Team wins are the strongest predictor of attendance**
- Attendance increases with wins, but at a **diminishing rate**
- **Points per game showed a negative relationship with attendance when controlling for wins**
- Three-point attempts, blocks, and the presence of an All-NBA player **did not significantly affect attendance**

### Final Model:
\[
\text{Attendance} = 16299.978 - 416.182P + 52.672W - 0.070W^2
\]

Where:
- \( P \) = points per game  
- \( W \) = team wins  

This suggests that fans primarily respond to **winning**, not specific styles of play or star recognition alone.

---

## ⚠️ Limitations

- Arena capacity differences were not accounted for  
- Socioeconomic factors (city population, ticket prices) were excluded  
- Attendance was modeled as a raw count rather than a percentage of capacity  

As a result, the model explains a modest portion of attendance variability.

---

## 🎯 Conclusions

The analysis indicates that **team success outweighs entertainment-based performance metrics** in driving NBA attendance. While exciting play styles and star players are often assumed to attract fans, their effects were not statistically significant when controlling for wins.

Future research should incorporate market-level and economic factors to build more comprehensive attendance models.

---

## 📚 Data Sources

- Basketball Reference — 2023–2024 NBA Season  
  https://www.basketball-reference.com/leagues/NBA_2024.html  
- Statista — NBA ticket revenue statistics  

---
