# Step 1: Import the Pandas library
# Pandas is used for loading and working with data in Python
import pandas as pd

# Step 2: Load the dataset
# We read the train.csv file and store it in a variable called df (short for DataFrame)
df = pd.read_csv("train.csv")

# Step 3: Display the first 5 rows of the dataset
# This gives us a quick peek at what the data looks like
print("First 5 rows of the dataset:")
print(df.head())

# Step 4: Display the number of rows and columns
# df.shape returns a tuple: (number of rows, number of columns)
print("\nNumber of rows and columns:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Step 5: Display all column names
# This shows us every feature (variable) available in the dataset
print("\nColumn names:")
print(df.columns.tolist())

# Step 6: Check for missing values in each column
# Missing values (also called NaN) are empty cells with no data
# df.isnull() checks every cell - True if empty, False if not
# .sum() then counts how many True values exist in each column
print("\nMissing values in each column:")
print(df.isnull().sum())

# Show only columns that actually have missing values (count > 0)
# This makes it easier to spot which columns need attention later
missing = df.isnull().sum()
missing_cols = missing[missing > 0]

if missing_cols.empty:
    print("No missing values found in the dataset!")
else:
    print("\nColumns with missing values:")
    print(missing_cols)
# Step 7: Check for duplicate rows
# Duplicate rows are records that appear more than once.

duplicate_count = df.duplicated().sum()

print("\nNumber of duplicate rows:", duplicate_count)

# Step 8: Display the data types of all columns
# A data type tells us what kind of information is stored in each column
# For example: int64 = whole numbers, float64 = decimal numbers, object = text
# Knowing data types helps us understand how to work with each column later
print("\nData types of each column:")
print(df.dtypes)

# Step 9: Display summary statistics for the dataset
# df.describe() automatically calculates useful statistics for every numerical column
# It shows: count (non-missing values), mean (average), std (spread of values),
# min (smallest value), 25%/50%/75% (quartiles), and max (largest value)
# This gives a quick overview of the range and distribution of each number column
print("\nSummary statistics of the dataset:")
print(df.describe())

# Step 10: Investigate missing values in the "Arrival Delay in Minutes" column
# Before deciding what to do with missing data, it is important to understand
# how many values are missing and what those rows look like

# Count how many values are missing in the "Arrival Delay in Minutes" column
# df["column"].isnull() returns True for each row where the value is missing
# .sum() counts all those True values to give us a total
arrival_missing_count = df["Arrival Delay in Minutes"].isnull().sum()
print("\nNumber of missing values in 'Arrival Delay in Minutes':")
print(arrival_missing_count)

# Display the first 5 rows where "Arrival Delay in Minutes" is missing
# df[df["column"].isnull()] filters the DataFrame to only show rows with missing values
# .head() then shows just the first 5 of those rows
print("\nFirst 5 rows where 'Arrival Delay in Minutes' is missing:")
print(df[df["Arrival Delay in Minutes"].isnull()].head())

# Step 11: Handle missing values in the "Arrival Delay in Minutes" column
# We will use a technique called "median imputation" to fill in the missing values
# Median imputation means replacing missing values with the median (middle value)
# of all the existing values in that column
# The median is preferred over the mean (average) because it is not affected
# by very large or very small outlier values

# We create a separate COPY of df called df_clean
# This keeps the original df completely unchanged so we can still refer to it
df_clean = df.copy()

# Calculate the median of the "Arrival Delay in Minutes" column
# .median() automatically ignores missing (NaN) values when calculating
arrival_median = df_clean["Arrival Delay in Minutes"].median()
print("\nMedian value used to fill missing 'Arrival Delay in Minutes':")
print(arrival_median)

# Show the number of missing values BEFORE filling
print("\nMissing values in 'Arrival Delay in Minutes' BEFORE filling:")
print(df_clean["Arrival Delay in Minutes"].isnull().sum())

# Fill the missing values in df_clean with the calculated median
# fillna() replaces every NaN in the column with the value we provide
# inplace=True means the change is applied directly to df_clean (no new variable needed)
df_clean["Arrival Delay in Minutes"] = df_clean["Arrival Delay in Minutes"].fillna(arrival_median)

# Show the number of missing values AFTER filling (should be 0)
print("\nMissing values in 'Arrival Delay in Minutes' AFTER filling:")
print(df_clean["Arrival Delay in Minutes"].isnull().sum())

# Step 12: Satisfaction Distribution
# Here we look at how many passengers are "satisfied" vs "neutral or dissatisfied"
# This is called the target variable - the main outcome we want to understand
# Knowing the distribution helps us spot imbalances (e.g. far more of one category)
# which is important context before doing any deeper analysis or modelling

print("\n--- Step 12: Satisfaction Distribution ---")

# Count how many passengers fall into each satisfaction category
# value_counts() tallies up every unique value in the column
satisfaction_counts = df_clean["satisfaction"].value_counts()
print("\nNumber of passengers in each satisfaction category:")
print(satisfaction_counts)

# Calculate the percentage share of each category
# normalize=True makes value_counts() return proportions (0 to 1) instead of raw counts
# Multiplying by 100 converts proportions to percentages
# round(2) keeps the result to 2 decimal places for readability
satisfaction_pct = df_clean["satisfaction"].value_counts(normalize=True) * 100
satisfaction_pct = satisfaction_pct.round(2)
print("\nPercentage of passengers in each satisfaction category:")
print(satisfaction_pct.astype(str) + " %")

# Step 13: Satisfaction by Customer Type
# Here we compare satisfaction levels between different types of customers:
# "Loyal Customer" (passengers who regularly fly with this airline) and
# "disloyal Customer" (passengers who do not regularly fly with this airline)
# A crosstab (cross-tabulation) is a table that shows how two categories
# relate to each other — like a tally chart with rows and columns
# This helps us understand whether loyal customers are more satisfied

print("\n--- Step 13: Satisfaction by Customer Type ---")

# pd.crosstab() counts how many passengers fall into each combination
# of Customer Type (rows) and satisfaction (columns)
crosstab_counts = pd.crosstab(df_clean["Customer Type"], df_clean["satisfaction"])
print("\nSatisfaction counts by Customer Type:")
print(crosstab_counts)

# Calculate row-wise percentages
# normalize='index' divides each row's values by the row total
# so each row adds up to 1.0 (a proportion)
# Multiplying by 100 converts proportions to percentages
# round(2) keeps the result tidy to 2 decimal places
crosstab_pct = pd.crosstab(df_clean["Customer Type"], df_clean["satisfaction"], normalize="index") * 100
crosstab_pct = crosstab_pct.round(2)
print("\nSatisfaction percentages by Customer Type (row-wise %):")
print(crosstab_pct)

# Step 14: Satisfaction by Travel Class
# Airlines typically offer different travel classes: Business, Eco, and Eco Plus
# Here we check whether passengers in higher classes (e.g. Business) tend to
# be more satisfied than those in economy classes
# This is useful because it reveals whether class of travel is linked to satisfaction

print("\n--- Step 14: Satisfaction by Travel Class ---")

# pd.crosstab() counts how many passengers in each travel Class
# fall into each satisfaction category (the columns)
crosstab_class_counts = pd.crosstab(df_clean["Class"], df_clean["satisfaction"])
print("\nSatisfaction counts by Travel Class:")
print(crosstab_class_counts)

# Calculate row-wise percentages
# normalize='index' divides each value by the total for that row (travel class)
# so each row's percentages add up to 100%
# This makes it easy to compare satisfaction rates across classes fairly
crosstab_class_pct = pd.crosstab(df_clean["Class"], df_clean["satisfaction"], normalize="index") * 100
crosstab_class_pct = crosstab_class_pct.round(2)
print("\nSatisfaction percentages by Travel Class (row-wise %):")
print(crosstab_class_pct)

# Step 15: Satisfaction by Type of Travel
# Passengers travel for different reasons - either for business or personal (leisure) trips
# Business travellers and personal travellers may have very different expectations
# and experiences, so their satisfaction levels could differ significantly
# This analysis helps us understand whether the purpose of travel is linked to satisfaction

print("\n--- Step 15: Satisfaction by Type of Travel ---")

# pd.crosstab() counts how many passengers of each travel type
# (Business Travel or Personal Travel) fall into each satisfaction category
crosstab_travel_counts = pd.crosstab(df_clean["Type of Travel"], df_clean["satisfaction"])
print("\nSatisfaction counts by Type of Travel:")
print(crosstab_travel_counts)

# Calculate row-wise percentages
# normalize='index' divides each value by the total for that row (travel type)
# so each row's percentages add up to 100%
# This allows a fair comparison between business and personal travellers
crosstab_travel_pct = pd.crosstab(df_clean["Type of Travel"], df_clean["satisfaction"], normalize="index") * 100
crosstab_travel_pct = crosstab_travel_pct.round(2)
print("\nSatisfaction percentages by Type of Travel (row-wise %):")
print(crosstab_travel_pct)

# Step 16: Satisfaction by Gender
# Here we check whether satisfaction levels differ between male and female passengers
# While gender may not be the strongest predictor, it is useful to examine
# all demographic factors to get a complete picture of what drives satisfaction

print("\n--- Step 16: Satisfaction by Gender ---")

# pd.crosstab() counts how many passengers of each gender
# fall into each satisfaction category
crosstab_gender_counts = pd.crosstab(df_clean["Gender"], df_clean["satisfaction"])
print("\nSatisfaction counts by Gender:")
print(crosstab_gender_counts)

# Calculate row-wise percentages
# normalize='index' divides each value by the total for that row (gender)
# so each row's percentages add up to 100%
# This allows a fair comparison between male and female passengers
crosstab_gender_pct = pd.crosstab(df_clean["Gender"], df_clean["satisfaction"], normalize="index") * 100
crosstab_gender_pct = crosstab_gender_pct.round(2)
print("\nSatisfaction percentages by Gender (row-wise %):")
print(crosstab_gender_pct)

# Step 17: Satisfaction by Inflight Wifi Service Rating
# Inflight wifi is a key service that passengers rate on a scale from 0 to 5
# (0 = not applicable, 1 = very poor, 5 = excellent)
# This analysis checks whether passengers who rated wifi higher tend to be
# more satisfied overall — helping identify if wifi quality is a driver of satisfaction

print("\n--- Step 17: Satisfaction by Inflight Wifi Service Rating ---")

# pd.crosstab() counts how many passengers gave each wifi rating (rows)
# and how they split between satisfaction categories (columns)
crosstab_wifi_counts = pd.crosstab(df_clean["Inflight wifi service"], df_clean["satisfaction"])
print("\nSatisfaction counts by Inflight Wifi Service Rating:")
print(crosstab_wifi_counts)

# Calculate row-wise percentages
# normalize='index' divides each value by the total for that rating row
# so each row's percentages add up to 100%
# This lets us compare satisfaction rates fairly across all rating levels
crosstab_wifi_pct = pd.crosstab(df_clean["Inflight wifi service"], df_clean["satisfaction"], normalize="index") * 100
crosstab_wifi_pct = crosstab_wifi_pct.round(2)
print("\nSatisfaction percentages by Inflight Wifi Service Rating (row-wise %):")
print(crosstab_wifi_pct)

# Step 18: Satisfaction by Departure Delay Status
# Instead of looking at the exact number of delay minutes, we simplify the data
# by grouping passengers into just two categories: "Delayed" or "No Delay"
# This makes it easier to compare satisfaction between delayed and on-time flights
# np.where() is a quick way to assign labels based on a condition

print("\n--- Step 18: Satisfaction by Departure Delay Status ---")

# Create a new column "Departure Delay Status" in df_clean
# If "Departure Delay in Minutes" is 0, label the row "No Delay"
# If it is greater than 0, label the row "Delayed"
# np.where(condition, value_if_true, value_if_false) works like an IF statement
import numpy as np
df_clean["Departure Delay Status"] = np.where(
    df_clean["Departure Delay in Minutes"] == 0, "No Delay", "Delayed"
)

# pd.crosstab() counts how many passengers in each delay status (rows)
# fall into each satisfaction category (columns)
crosstab_delay_counts = pd.crosstab(df_clean["Departure Delay Status"], df_clean["satisfaction"])
print("\nSatisfaction counts by Departure Delay Status:")
print(crosstab_delay_counts)

# Calculate row-wise percentages
# normalize='index' divides each value by the row total
# so each row's percentages add up to 100%
# This lets us fairly compare satisfaction between delayed and on-time passengers
crosstab_delay_pct = pd.crosstab(df_clean["Departure Delay Status"], df_clean["satisfaction"], normalize="index") * 100
crosstab_delay_pct = crosstab_delay_pct.round(2)
print("\nSatisfaction percentages by Departure Delay Status (row-wise %):")
print(crosstab_delay_pct)

# Step 19: Satisfaction Distribution Visualization
# Numbers in a table are useful, but a bar chart makes patterns much easier to see at a glance
# Here we create a bar chart showing how many passengers are in each satisfaction category
# This gives a clear visual picture of the overall balance of satisfied vs dissatisfied passengers

print("\n--- Step 19: Satisfaction Distribution Visualization ---")

# Import matplotlib - the most popular Python library for creating charts and graphs
import matplotlib.pyplot as plt

# Count the number of passengers in each satisfaction category
# value_counts() returns the counts sorted from highest to lowest
satisfaction_counts = df_clean["satisfaction"].value_counts()

# Create a bar chart
# figsize=(8, 5) sets the width and height of the chart in inches
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the bars — the index gives the category names, the values give the bar heights
bars = ax.bar(satisfaction_counts.index, satisfaction_counts.values, color=["steelblue", "coral"])

# Add a title and axis labels to make the chart self-explanatory
ax.set_title("Satisfaction Distribution of Airline Passengers", fontsize=14)
ax.set_xlabel("Satisfaction Category", fontsize=12)
ax.set_ylabel("Number of Passengers", fontsize=12)

# Display the exact count above each bar so the viewer can read the precise numbers
# bar.get_height() gives the height (value) of each bar
# ax.text() places the number centred above the bar
for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,  # horizontal centre of the bar
        bar.get_height() + 200,              # just above the top of the bar
        f"{int(bar.get_height()):,}",        # count formatted with comma separator
        ha="center", va="bottom", fontsize=11
    )

# tight_layout() automatically adjusts spacing so nothing gets cut off
plt.tight_layout()

# show() renders and displays the chart
plt.show()

# Step 20: Summary of Key Findings
# After completing all the individual analysis steps above, this final step
# brings the most important results together in one place
# It acts like a "report card" — a quick reference for anyone who wants
# to understand the key patterns in the dataset without reading every step

print("\n" + "=" * 60)
print("     Step 20: Summary of Key Findings")
print("=" * 60)

# ── Total passengers ──────────────────────────────────────────
print(f"\nTotal passengers analysed: {len(df_clean):,}")

# ── Overall satisfaction counts and percentages ───────────────
print("\n[ Overall Satisfaction ]")
sat_counts = df_clean["satisfaction"].value_counts()
sat_pct    = df_clean["satisfaction"].value_counts(normalize=True) * 100
for category in sat_counts.index:
    print(f"  {category}: {sat_counts[category]:,} passengers ({sat_pct[category].round(2)} %)")

# ── Satisfaction % by Customer Type ───────────────────────────
print("\n[ Satisfaction % by Customer Type ]")
ct_pct = pd.crosstab(df_clean["Customer Type"], df_clean["satisfaction"], normalize="index") * 100
ct_pct = ct_pct.round(2)
print(ct_pct.to_string())

# ── Satisfaction % by Travel Class ────────────────────────────
print("\n[ Satisfaction % by Travel Class ]")
cl_pct = pd.crosstab(df_clean["Class"], df_clean["satisfaction"], normalize="index") * 100
cl_pct = cl_pct.round(2)
print(cl_pct.to_string())

# ── Satisfaction % by Type of Travel ──────────────────────────
print("\n[ Satisfaction % by Type of Travel ]")
tt_pct = pd.crosstab(df_clean["Type of Travel"], df_clean["satisfaction"], normalize="index") * 100
tt_pct = tt_pct.round(2)
print(tt_pct.to_string())

# ── Satisfaction % by Gender ──────────────────────────────────
print("\n[ Satisfaction % by Gender ]")
gd_pct = pd.crosstab(df_clean["Gender"], df_clean["satisfaction"], normalize="index") * 100
gd_pct = gd_pct.round(2)
print(gd_pct.to_string())

# ── Satisfaction % by Inflight Wifi Service Rating ────────────
print("\n[ Satisfaction % by Inflight Wifi Service Rating ]")
wf_pct = pd.crosstab(df_clean["Inflight wifi service"], df_clean["satisfaction"], normalize="index") * 100
wf_pct = wf_pct.round(2)
print(wf_pct.to_string())

# ── Satisfaction % by Departure Delay Status ──────────────────
# We check whether this column exists first, because it is created in Step 18
# If Step 18 was skipped for any reason, this block will not raise an error
if "Departure Delay Status" in df_clean.columns:
    print("\n[ Satisfaction % by Departure Delay Status ]")
    dd_pct = pd.crosstab(df_clean["Departure Delay Status"], df_clean["satisfaction"], normalize="index") * 100
    dd_pct = dd_pct.round(2)
    print(dd_pct.to_string())

print("\n" + "=" * 60)
print("     End of Analysis")
print("=" * 60)
