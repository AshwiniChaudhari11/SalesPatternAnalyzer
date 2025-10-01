# SalesPatternAnalyzer

## 📌 Description

This project utilizes **linear regression** to build a machine learning model that analyzes sales data from a retail store. The dataset comprises daily sales figures for the year 2022, including indicators for holidays, promotions, and the day of the week.  
The goal is to identify **trends, relationships among variables, and patterns of sales growth**.

---

## ✨ Features

- Analyzes **daily sales data for 365 days in 2022**.
- Utilizes a **Linear Regression model** to predict sales based on holidays, promotions, and the day of the week.
- Visualizes sales trends using **histograms and line plots**.
- Explores relationships between sales and independent variables (weekdays, promotions, holidays) using **histograms, boxplots, and bar charts**.
- Compares performance of multiple regression models:
  - Linear Regression
  - Decision Tree
  - Random Forest

---

## ⚙️ Getting Started

### ✅ Prerequisites

Ensure you have the following installed:

- Python 3.x
- Required libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `jupyter`

---

## 📊 Data

The project uses a CSV file named **`Sales.csv`**, located in the same folder as the Jupyter Notebook.

**Columns:**

- `Date` → Date of the sales
- `Weekday` → Day of the week (1–7)
- `Promotions` → Promotion indicator (1 = promotion, 0 = no promotion)
- `Holiday` → Holiday indicator (1 = holiday, 0 = non-holiday)
- `Sales` → Number of sales made

---

## 🚀 Usage

1. Clone the repository or download the files.
2. Place **`Groceries data.csv`** in the same directory as the notebook.
3. Open the Jupyter Notebook and run cells sequentially.
4. The notebook will:
   - Load and preprocess the data (normalize using MinMaxScaler).
   - Split the dataset into training & testing sets.
   - Train regression models and show accuracy.
   - Visualize actual vs. predicted sales.

---

## 📌 Example

Run the Jupyter Notebook step by step to view:

- Graphs of sales trends
- Model evaluation
- Comparison of regression techniques

---

## Contributing

Contributions are welcome!

- Open an issue for bugs or feature requests
- Submit a pull request with improvements

---
