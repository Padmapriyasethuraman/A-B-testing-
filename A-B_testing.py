import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu

import warnings
warnings.filterwarnings('ignore')

# Assuming the correct file path
df = pd.read_csv('/content/marketing_AB.csv')
df.head()

# Check for duplicate user ids
print("Number of duplicate user ids:", df.duplicated(subset='user id').sum())

# Drop unnecessary columns
df.drop(['Unnamed: 0', 'user id'], axis=1, inplace=True)

# Check the columns of the DataFrame
print(df.columns)

# Select specific columns
df_cat = df[['test group', 'converted', 'most ads day', 'most ads hour']]

# Check the number of unique values in each selected column
print(df_cat.nunique())

for i in df_cat.columns:
    print(i.upper(), ":", df_cat[i].unique())

# Univariate Analysis
variable = 'test group'

plt.figure(figsize=(6,4))
# Count plot
plt.subplot(1,2,1)
sns.countplot(x=variable, data=df_cat)
plt.title(f'Count Plot - {variable}')

# Pie chart
plt.subplot(1,2,2)
counts = df_cat[variable].value_counts()
plt.pie(counts, labels=counts.index, autopct='%0.2f%%')
plt.title(f'Pie Chart - {variable}')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

variable = 'converted'

plt.figure(figsize=(6,4))
# Count plot
plt.subplot(1,2,1)
sns.countplot(x=variable, data=df_cat)
plt.title(f'Count Plot - {variable}')

# Pie chart
plt.subplot(1,2,2)
counts = df_cat[variable].value_counts()
plt.pie(counts, labels=counts.index, autopct='%0.2f%%')
plt.title(f'Pie Chart - {variable}')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

variable = 'most ads day'

plt.figure(figsize=(6,4))
# Count plot
plt.subplot(1,2,1)
sns.countplot(x=variable, data=df_cat, order=df_cat['most ads day'].value_counts().index)
plt.title(f'Count Plot - {variable}')
plt.xticks(rotation=90)

# Pie chart
plt.subplot(1,2,2)
counts = df_cat[variable].value_counts()
plt.pie(counts, labels=counts.index, autopct='%0.2f%%')
plt.title(f'Pie Chart - {variable}')

plt.tight_layout()

plt.show()

variable = 'most ads hour'
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
sns.countplot(x=variable, data=df_cat, order=df_cat['most ads hour'].value_counts().index)
plt.title(f'Count Plot - {variable}')
plt.xticks(rotation=90)

# Pie chart
plt.subplot(1,2,2)
counts = df_cat[variable].value_counts()
plt.pie(counts, labels=counts.index, autopct='%0.2f%%')
plt.title(f'Pie Chart - {variable}')

plt.tight_layout()

plt.show()

variable = 'total ads'

plt.figure(figsize=(6,4))
# Histogram
plt.subplot(1,2,1)
sns.histplot(data=df, x=variable)
plt.title(f'Histogram - {variable}')

# Box plot
plt.subplot(1,2,2)
sns.boxplot(data=df, y=variable)
plt.title(f'Box Plot - {variable}')

plt.tight_layout()

plt.show()

print(df['total ads'].describe())

variable = 'total ads'

plt.figure(figsize=(6,4))
# Histogram for filtered data
plt.subplot(1,2,1)
sns.histplot(data=df[df['total ads'] < 50], x=variable)
plt.title(f'Histogram - {variable}')

# Box plot for filtered data
plt.subplot(1,2,2)
sns.boxplot(data=df[df['total ads'] < 50], y=variable)
plt.title(f'Box Plot - {variable}')

plt.tight_layout()

plt.show()

# Drop NaN values for statistical tests
df = df.dropna(subset=['total ads', 'converted'])

# Check unique values in 'converted' column
print("Unique values in 'converted' column:", df['converted'].unique())

# Bivariate Analysis

ct_conversion_test_group = pd.crosstab(df['test group'], df['converted'], normalize='index')
ct_conversion_test_group.plot.bar(stacked=True)
plt.title('Conversion Rate by Test Group')
plt.show()

ct_conversion_day = pd.crosstab(df['most ads day'], df['converted'], normalize='index')
print(ct_conversion_day.sort_values(by=True, ascending=False))
ct_conversion_day.plot.bar(stacked=True)
plt.title('Conversion Rate by Most Ads Day')
plt.show()

ct_conversion_hour = pd.crosstab(df['most ads hour'], df['converted'], normalize='index')
print(ct_conversion_hour.sort_values(by=True, ascending=False))
ct_conversion_hour.plot.bar(stacked=True)
plt.title('Conversion Rate by Most Ads Hour')
plt.show()

sns.boxplot(x='converted', y='total ads', data=df)
plt.title('Total Ads by Conversion Status')
plt.show()

sns.boxplot(x='converted', y='total ads', data=df[df['total ads'] < 50])
plt.title('Total Ads (Filtered) by Conversion Status')
plt.show()

# Statistical Tests

# Chi-square tests for categorical variables
def chi_square_test(var1, var2):
    contingency_table = pd.crosstab(df[var1], df[var2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f'Chi-square test between {var1} and {var2}:')
    print(f'Chi2: {chi2}, p-value: {p}\n')

chi_square_test('test group', 'converted')
chi_square_test('most ads day', 'converted')
chi_square_test('most ads hour', 'converted')

# T-test for numerical variables
def t_test(var):
    group1 = df[df['converted'] == 'Yes'][var]
    group2 = df[df['converted'] == 'No'][var]
    if len(group1) > 0 and len(group2) > 0:
        t_stat, p_value = ttest_ind(group1, group2)
        print(f'T-test for {var}:')
        print(f'T-statistic: {t_stat}, p-value: {p_value}\n')
    else:
        print(f'Not enough data to perform T-test for {var}.\n')

t_test('total ads')

# Mann-Whitney U test for numerical variables (if data is not normally distributed)
def mann_whitney_test(var):
    group1 = df[df['converted'] == 'Yes'][var]
    group2 = df[df['converted'] == 'No'][var]
    if len(group1) > 0 and len(group2) > 0:
        u_stat, p_value = mannwhitneyu(group1, group2)
        print(f'Mann-Whitney U test for {var}:')
        print(f'U-statistic: {u_stat}, p-value: {p_value}\n')
    else:
        print(f'Not enough data to perform Mann-Whitney U test for {var}.\n')

mann_whitney_test('total ads')

# Check normality of the data
sns.histplot(df['total ads'])
plt.title('Histogram of Total Ads')
plt.show()

sns.histplot(df[df['converted'] == 'Yes']['total ads'])
plt.title('Histogram of Total Ads (Converted)')
plt.show()

sns.histplot(df[df['converted'] == 'No']['total ads'])
plt.title('Histogram of Total Ads (Not Converted)')
plt.show()
