# Airline Passenger Satisfaction Analysis

## 1. Project Overview

This project performs an exploratory data analysis (EDA) of airline passenger satisfaction data. Using Python and the Pandas library, the project loads, inspects, cleans, and analyses the dataset to uncover patterns and trends related to passenger satisfaction across multiple demographic and service-related factors.

The analysis is written in a single beginner-friendly Python script (`AryanChawla_AirlinePassengerSatisfaction.py`) with detailed comments explaining each step.

---

## 2. Project Objectives

- Load and explore the airline passenger satisfaction dataset.
- Identify and handle missing values using median imputation.
- Check for duplicate records.
- Understand the distribution of the target variable (`satisfaction`).
- Analyse satisfaction levels across key passenger segments:
  - Customer type (loyal vs. disloyal)
  - Travel class (Business, Eco, Eco Plus)
  - Type of travel (Business Travel vs. Personal Travel)
  - Gender
  - Inflight Wi-Fi service rating
  - Departure delay status
- Visualise the overall satisfaction distribution using a bar chart.
- Summarise key findings in a consolidated report.
- Present findings through both the command-line analysis script and an interactive Streamlit dashboard.

---

## 3. Dataset Information

| Detail | Information |
|--------|-------------|
| **Dataset Name** | Airline Passenger Satisfaction |
| **Source** | [Kaggle — teejmahal20](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) |
| **Files Used** | `train.csv` (primary analysis), `test.csv` (available for future use) |

The dataset contains passenger records with demographic information, flight details, service ratings, and a satisfaction label (`satisfied` or `neutral or dissatisfied`).

---

## 4. Tools and Libraries Used

| Tool / Library | Purpose |
|----------------|---------|
| **Python 3** | Programming language used for the entire analysis |
| **pandas** | Data loading, inspection, cleaning, and analysis |
| **numpy** | Creating the departure delay status column using `np.where()` |
| **matplotlib** | Plotting the satisfaction distribution bar chart |
Streamlit	Creating the interactive dashboard with KPI cards, filters, charts, and insights

---

## 5. Data Analysis Steps

The script `AryanChawla_AirlinePassengerSatisfaction.py` carries out the following steps in order:

| Step | Description |
|------|-------------|
| **Step 1** | Import the Pandas library |
| **Step 2** | Load `train.csv` into a DataFrame named `df` |
| **Step 3** | Display the first 5 rows |
| **Step 4** | Display the number of rows and columns |
| **Step 5** | Display all column names |
| **Step 6** | Check for missing values in each column |
| **Step 7** | Check for duplicate rows |
| **Step 8** | Display data types of all columns |
| **Step 9** | Display summary statistics for numerical columns |
| **Step 10** | Investigate missing values in `Arrival Delay in Minutes` |
| **Step 11** | Fill missing arrival delay values using median imputation (stored in `df_clean`) |
| **Step 12** | Analyse overall satisfaction distribution |
| **Step 13** | Analyse satisfaction by customer type |
| **Step 14** | Analyse satisfaction by travel class |
| **Step 15** | Analyse satisfaction by type of travel |
| **Step 16** | Analyse satisfaction by gender |
| **Step 17** | Analyse satisfaction by inflight Wi-Fi service rating |
| **Step 18** | Create a departure delay status column and analyse satisfaction |
| **Step 19** | Visualise the satisfaction distribution using a bar chart |
| **Step 20** | Print a consolidated summary of all key findings |

---

## 6. Key Findings

### Overall Satisfaction
- **56.67%** of passengers were **neutral or dissatisfied**.
- **43.33%** of passengers were **satisfied**.

### Satisfaction by Customer Type
- **Loyal customers** had a notably higher satisfaction rate than disloyal customers.
- **Disloyal customers** were predominantly neutral or dissatisfied, suggesting that loyalty programmes or first-time experience quality may influence overall satisfaction.

### Satisfaction by Travel Class
- **Business class** passengers showed a markedly higher satisfaction rate compared to economy passengers.
- **Eco** and **Eco Plus** passengers were more likely to be neutral or dissatisfied, indicating that service quality differences across classes are reflected in satisfaction outcomes.

### Satisfaction by Type of Travel
- **Business travellers** were more likely to be satisfied than personal travellers.
- **Personal (leisure) travellers** were predominantly neutral or dissatisfied, which may reflect higher personal expectations or sensitivity to service quality on non-work trips.

### Satisfaction by Gender
- Satisfaction rates were broadly similar between male and female passengers.
- Gender alone does not appear to be a strong differentiating factor in this dataset.

### Satisfaction by Inflight Wi-Fi Service Rating
- Passengers who gave higher ratings to the inflight Wi-Fi service tended to be found more frequently in the satisfied category.
- Passengers who rated the Wi-Fi service poorly (ratings 1–2) were more likely to be neutral or dissatisfied.
- Note: this is an observed association, not a claim of causation.

### Satisfaction by Departure Delay Status
- Passengers whose flights departed on time (**No Delay**) had a higher proportion of satisfied passengers compared to those who experienced a delay.
- Delayed passengers were more likely to be neutral or dissatisfied, consistent with the expectation that departure delays negatively affect the travel experience.

---

## 7. How to Run the Project

### Step 1 — Install dependencies

Make sure Python 3 is installed on your system. Then install the required libraries by running:

```bash
pip install -r requirements.txt
```

### Step 2 — Run the analysis script

Ensure that `train.csv` and `test.csv` are in the same folder as `AryanChawla_AirlinePassengerSatisfaction.py`, then run:

```bash
python AryanChawla_AirlinePassengerSatisfaction.py
```

The script will print all analysis results to the console and display a bar chart for Step 19.

---

## 8. Project Files

| File | Description |
|------|-------------|
| `AryanChawla_AirlinePassengerSatisfaction.py` | Main Python script containing all 20 analysis steps |
| `train.csv` | Training dataset used for the analysis |
| `test.csv` | Test dataset (available for future use or model evaluation) |
| `requirements.txt` | List of Python libraries required to run the project |
| `README.md` | Project documentation (this file) |

---

## 9. Conclusion

This project demonstrates a structured, beginner-friendly approach to exploratory data analysis using Python and Pandas. By working through 20 clearly commented steps, the analysis uncovers meaningful patterns in airline passenger satisfaction data.

The findings suggest that travel class, customer loyalty, and type of travel are among the most differentiated factors associated with passenger satisfaction in this dataset. Inflight Wi-Fi ratings and departure delay status also show observable associations with satisfaction levels. These insights can inform airline service improvement priorities and provide a strong foundation for further analysis or predictive modelling.


---

## 10. Interactive Dashboard

A Streamlit-based interactive dashboard (`dashboard.py`) has been added to the project.
It provides KPI cards, interactive sidebar filters, charts, service-rating analysis,
delay analysis, and an insights & recommendations section — all driven by `train.csv`.

### What the dashboard includes

| Section | Details |
|---------|---------|
| **KPI Cards** | Total passengers, satisfied count, neutral/dissatisfied count, satisfaction rate |
| **Passenger Group Filters** | Sidebar filters for Customer Type, Type of Travel, Travel Class, and Gender |
| **Satisfaction by Group** | Tabs showing counts and rates for Customer Type, Travel Type, Class, Gender, and Age Group |
| **Service Ratings** | Mean rating comparison (satisfied vs neutral/dissatisfied) for all 14 service attributes; single-service drill-down |
| **Delay Analysis** | Satisfaction by delay status, delay distribution histogram, mean delays by group |
| **Insights & Recommendations** | Dynamic observation table + six practical recommendations |

### Prerequisites

Install the required libraries (streamlit is now included in `requirements.txt`):

```bash
pip install -r requirements.txt
```

### How to run the dashboard

Open a terminal, navigate to the project folder, and run:

```bash
streamlit run dashboard.py
```

Streamlit will open the dashboard automatically in your default web browser
(usually at **http://localhost:8501**).

> **Important:** Run the command from inside the project folder (the folder that
> contains `train.csv` and `dashboard.py`). If you run it from a different
> directory, the dashboard will still work because the script resolves the
> data path relative to its own location.

### Stopping the dashboard

Press `Ctrl + C` in the terminal to stop the Streamlit server.
