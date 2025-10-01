# SalesTrendAnalyzer

## Description
This project utilizes linear regression to create a machine learning model that analyzes sales data from a retail store. The dataset comprises daily sales figures for the year 2022, including indicators for holidays, promotions, and the day of the week. The goal is to identify trends, relationships among variables, and patterns of sales growth.

## Features
- Analyzes daily sales data for 365 days in 2022.
- Utilizes a linear regression model to predict sales based on holidays, promotions, and the day of the week.
- Visualizes sales trends using histograms and line plots.
- Explores relationships between sales and independent variables (weekdays, promotions/discounts, holidays) using histograms,     boxplots and bar charts.
- Compares the performance of multiple regression models, including Linear Regression, Decision Tree, and Random Forest.

## Getting Started

### Prerequisites
Ensure you have the following software installed:
- Python 3.x
- Required Python libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
- Jupyter Notebook 7.x

## Data
The project uses a CSV file named Sales.csv, which should be located in the same folder as the Jupyter notebook. The file contains the following columns:

  - Date: Date of the sales.
  - Weekday: Day of the week (1-7).
  - Promotions: Promotion indicator (1 if there is a promotion, 0 if not).
  - Holiday: Holiday indicator (1 if it is a holiday, 0 if not).
  - Sales: Number of sales made.

## Usage
  1. Clone the repository or download the necessary files.
  2. Ensure that the Sales.csv file is in the same directory as the Jupyter notebook.
  3. Open the Jupyter notebook and run the cells in order.
  4. The code will perform the following actions:
     - Load the data from the CSV file.
     - Normalize the data using MinMaxScaler.
     - Split the data into training and testing sets.
     - Train several regression models and display their accuracy.
     - Evaluate the linear regression model, showing graphs of actual vs. predicted sales.

## Example
An example of how to run the analysis is available in the Jupyter notebook. Simply execute the cells one by one to see the results.

## Contributing
Contributions are welcome. If you would like to improve the project, please open an issue or a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more information.

### Feel Free to Customize
You can further customize this README as needed. If there are specific steps you want to include in the **Usage** section.

### Acknowledgments
Thank you Federico Garay.
