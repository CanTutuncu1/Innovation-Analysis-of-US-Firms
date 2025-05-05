# CAN TUTUNCU Data Assignment 2 in Python

############
### Part 1: Data merging and cleaning

# import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import the patent data 
pat = pd.read_stata("/Users/tokyo/Documents/Bocconi University/ESS/20971 Innovation and Growth/Data Task/pat76_06_ipc.dta")

# we drop the NaN values in the column 'pdpass'.
pat = pat.dropna(subset='pdpass')

# count how many times pdpass is repeated for every year in a new column
pat['tot_pat_year'] = pat.groupby(['pdpass', 'appyear'])['pdpass'].transform('count')

# drop the duplicate rows
pat = pat.drop_duplicates(subset='pdpass')

# reset the index now (not necessary to do)
pat = pat.reset_index(drop=True)

# select the necessary columns
columns_to_keep = ['appyear', 'pdpass','tot_pat_year']
pat = pat[columns_to_keep]

# import the dynass data
dynass = pd.read_stata("/Users/tokyo/Documents/Bocconi University/ESS/20971 Innovation and Growth/Data Task/dynass.dta")

# merge the patent data and dynass data using pdpass
merged_pat = pd.merge(pat, dynass, on='pdpass')

# row-wise conditional check across multiple column groups using 'appyear'. Store the matching gvkey{i} for all i=1,2,3,4,5 into a new column 'gvkey'.
# matching function to apply on each row
def get_all_matches(row):
    matches = []
    for i in range(1, 6):
        beg = row.get(f'begyr{i}')
        end = row.get(f'endyr{i}')
        gvkey = row.get(f'gvkey{i}')
        app = row['appyear']

        if pd.notnull(beg) and pd.notnull(end) and pd.notnull(app): # Ignore the null entries
            if beg <= app <= end: #selection criteria
                matches.append(gvkey)  # gvkey{i} that matches
    return matches if matches else np.nan

# apply to every row and store in a new column 'gvkey'.
merged_pat['gvkey'] = merged_pat.apply(get_all_matches, axis=1)

# in case there are multiple matches to appyear in one row, explode them to new rows.
merged_pat = merged_pat.explode('gvkey')

# sort the dataframe with 'gvkey' in ascending order
merged_pat = merged_pat.sort_values(by='gvkey', ascending=True).reset_index(drop=True)
merged_pat = merged_pat.dropna(subset='gvkey')

# move the gvkey column to the front
cols = merged_pat.columns.tolist()
cols.insert(0, cols.pop(cols.index('gvkey')))
merged_pat = merged_pat[cols]

# (if not already) transform 'gvkey' into a string
merged_pat['gvkey'] = merged_pat['gvkey'].astype(str)

# group the data by every unique pair of 'gvkey' and 'appyear' and sum 'tot_pat_year for each group
merged_pat = merged_pat.groupby(['gvkey', 'appyear'], as_index=False)['tot_pat_year'].sum()

# import the pdpcohdr data
pdpcohdr = pd.read_stata("/Users/tokyo/Documents/Bocconi University/ESS/20971 Innovation and Growth/Data Task/pdpcohdr.dta")
pdpcohdr = pdpcohdr.dropna(subset='gvkey')

#transform 'gvkey' into a string
pdpcohdr['gvkey'] = pdpcohdr['gvkey'].astype(str)

# merge the refined patent data with the pdpcohdr data using gvkey
merged_gvkey = pd.merge(merged_pat, pdpcohdr, on='gvkey')

# select the necessary columns
columns_to_keep2 = ['gvkey', 'appyear', 'tot_pat_year','match']
merged_gvkey = merged_gvkey[columns_to_keep2]

# import the compustat data
compustat = pd.read_csv("/Users/tokyo/Documents/Bocconi University/ESS/20971 Innovation and Growth/Data Task/WRDS_Company.csv")
compustat['gvkey'] = compustat['gvkey'].astype(str)

# scale up 'emp' and 'sale' because pandas view dot as a decimal separator and not thousand separator
compustat['emp'] = compustat['emp'] * 1000
compustat['sale'] = compustat['sale'] * 1000

# merge compustat data with the refined pattern data using 'gvkey' and 'appyear'/'fyear'
final_data = pd.merge(merged_gvkey, compustat, left_on=['gvkey', 'appyear'],
    right_on=['gvkey', 'fyear'],
    how='right')

final_data = final_data.drop(columns='appyear')

# remove the rows where both 'tot_pat_year' and 'fyear' are NaN.
final_data = final_data.dropna(subset=['tot_pat_year', 'fyear'], how='all')

# for the rows where the matches are non-null, assign zero to 'tot_pat_year' (there were none it seems anyway)
mask = final_data['match'].notna() & final_data['tot_pat_year'].isna()
final_data.loc[mask, 'tot_pat_year'] = 0
final_data = final_data.drop(columns='match')

# select the row data that is betweeen 1980 and 2000
final_data = final_data[(final_data['fyear'] >= 1980) & (final_data['fyear'] <= 2000)]

# select firms only incorporated in the US.
final_data = final_data[final_data['fic'] == 'USA']

# furthermore from these firms select the ones that report in US dollars.
final_data = final_data[final_data['curcd'] == 'USD']

# replace the NaN entries in 'match' column with 0.0
final_data['tot_pat_year'] = final_data['tot_pat_year'].fillna(0.0)

############
### Part 2: Innovative Firms

# select the maximum value of 'tot_pat_year' for every group of 'gvkey' and store it in 'max_pat'
final_data['max_pat'] = final_data.groupby('gvkey')['tot_pat_year'].transform('max')

# in a new column variable 'innovative_firm', write for firms with at least one patent 'yes'
final_data['innovative'] = final_data['max_pat'].apply(lambda x: 'Yes' if x > 0 else 'No')

# check that each 'gvkey' has only one unique corresponding 'innovative' value
final_data.groupby('gvkey')['innovative'].nunique().value_counts()

# check the share of firms which have at least one pattern.
final_data['innovative'].value_counts('Yes')

# aggregate the employment size of firms by taking average of employment for every 'gvkey' group
firm_emp = final_data.groupby('gvkey').agg({
    'emp': 'mean',  # or median, max, etc.
    'innovative': 'first'
}).reset_index()

# compare the average employment size of innovative vs non-innovative firms
firm_emp.groupby('innovative')['emp'].mean()

# compare the standard deviation of employment size
firm_emp.groupby('innovative')['emp'].std()

firm_emp.groupby('innovative')['emp'].agg(['mean', 'std'])

# aggregate the sale size of firms by taking average of sale for every 'gvkey' group
firm_sale = final_data.groupby('gvkey').agg({
    'sale': 'mean',  # or median, max, etc.
    'innovative': 'first'
}).reset_index()

# compare the average sale size of innovative and non-innovative firms
firm_sale.groupby('innovative')['sale'].mean()

# compare the standard deviation of sale size
firm_sale.groupby('innovative')['sale'].std()

firm_sale.groupby('innovative')['sale'].agg(['mean', 'std'])

##########
### Part 3: Firm Growth by Firm Size

# select only the innovative firms
innovative_firms = final_data[final_data['innovative'] == 'Yes']

# calculate employment growth (forward) percentage wise
innovative_firms['emp_growth_rate'] = innovative_firms.groupby('gvkey')['emp'].pct_change(periods=1, fill_method=None).shift(-1) * 100

# apply winsorization by imposing 1000% growth cap
innovative_firms['emp_growth_rate'] = innovative_firms['emp_growth_rate'].clip(upper=1000)

# row-level mean. we assign equal weight to row year data of all firms
innovative_firms['emp_growth_rate'].mean()

# firm-level mean. we assign equal weight to every single firm after taking the average of their growth rate
innovative_firms.groupby('gvkey')['emp_growth_rate'].mean().mean()

# replicate figure 1
# create 20 roughly equal‐sized bins based on employee count
innovative_firms['size_bin'] = pd.qcut(innovative_firms['emp'], q=20)

# Now, group by these bins and calculate:
# - avg_emp: average employee count per bin
# - avg_growth: average forward employment growth per bin
binned = innovative_firms.groupby('size_bin').agg(
    avg_emp=('emp', 'mean'),
    avg_growth=('emp_growth_rate', 'mean')
).reset_index()

# create equal spacing in the x-axis corresponding to the bins
binned['bin_order'] = range(1, len(binned) + 1)

plt.figure(figsize=(10, 6))
plt.plot(binned['bin_order'], binned['avg_growth'], marker='o', linestyle='-', color='blue')
plt.xlabel('Average employee count in size bin')
plt.ylabel('Forward employment growth (%)')
plt.title('Figure 1: Firm Growth by Firm Size')
plt.xticks(ticks=binned['bin_order'], labels=[f"{x:,.0f}" for x in binned['avg_emp']], rotation=45)

plt.grid(False)
plt.tight_layout()
plt.show()

# create data frame for the regression
reg_data = innovative_firms[['gvkey', 'fyear', 'emp_growth_rate', 'emp']].copy()
reg_data = reg_data.replace([np.inf, -np.inf], np.nan).dropna()

# define log_emp
reg_data = reg_data[reg_data['emp'].div(1000) > 0]
reg_data['log_emp'] = np.log(reg_data['emp'].div(1000))

# set the panel index
reg_data = reg_data.set_index(['gvkey', 'fyear'])

# run the regression
from linearmodels.panel import PanelOLS # type: ignore
mod = PanelOLS.from_formula('emp_growth_rate ~ log_emp + EntityEffects + TimeEffects', data=reg_data)
results = mod.fit()
print(results.summary)

# replicate the graph for the alternative measure of firm growth rate (can't use .pct_change this time)
# calculate lagged employment per firm
innovative_firms2 = innovative_firms.copy()
innovative_firms2['emp_lag'] = innovative_firms2.groupby('gvkey')['emp'].shift(1)
innovative_firms2['alt_emp_growth_rate'] = ((innovative_firms2['emp'] - innovative_firms2['emp_lag']) /
                                      (0.5 * (innovative_firms2['emp'] + innovative_firms2['emp_lag']))).shift(-1) * 100

innovative_firms2.head()

# replicate figure 1
# create 20 roughly equal‐sized bins based on employee count
innovative_firms2['size_bin'] = pd.qcut(innovative_firms2['emp'], q=20)

# Now, group by these bins and calculate:
# - avg_emp: average employee count per bin
# - avg_growth: average forward employment growth per bin
binned2 = innovative_firms2.groupby('size_bin').agg(
    avg_emp=('emp', 'mean'),
    avg_growth=('alt_emp_growth_rate', 'mean')
).reset_index()

# create equal spacing in the x-axis corresponding to the bins
binned2['bin_order'] = range(1, len(binned2) + 1)

plt.figure(figsize=(10, 6))
plt.plot(binned2['bin_order'], binned2['avg_growth'], marker='o', linestyle='-', color='blue')
plt.xlabel('Average employee count in size bin')
plt.ylabel('Forward employment growth (%) (alternative)')
plt.title('Figure 1.2: Firm Growth by Firm Size (Alternative)')
plt.xticks(ticks=binned2['bin_order'], labels=[f"{x:,.0f}" for x in binned2['avg_emp']], rotation=45)

plt.grid(False)
plt.tight_layout()
plt.show()

### Part 4: Innovation Intensity by Firm Size

# compute the number of patents per employment
innovative_firms3 = innovative_firms.copy()
innovative_firms3['patents_per_emp'] = innovative_firms3['tot_pat_year'] / innovative_firms3['emp']

innovative_firms.count()

# count the number of zeroes in the total number of patents ever year
zero_count = (innovative_firms3['tot_pat_year'] == 0).sum()
print(zero_count)

# replicate figure 2
# create 20 roughly equal‐sized bins based on employee count
innovative_firms3['size_bin'] = pd.qcut(innovative_firms3['emp'], q=20)

# Now, group by these bins and calculate:
# - avg_emp: average employee count per bin
# - patens_per_emp: average patents per employee
binned3 = innovative_firms3.groupby('size_bin').agg(
    avg_emp=('emp', 'mean'),
    patens_per_emp=('patents_per_emp', 'mean')
).reset_index()

# create equal spacing in the x-axis corresponding to the bins
binned3['bin_order'] = range(1, len(binned3) + 1)

plt.figure(figsize=(10, 6))
plt.plot(binned3['bin_order'], binned3['patens_per_emp'], marker='o', linestyle='-', color='blue')
plt.xlabel('Average employee count in size bin')
plt.ylabel('Patents per employee')
plt.title('Figure 2: Innovation Intensity by Firm Size')
plt.xticks(ticks=binned3['bin_order'], labels=[f"{x:,.0f}" for x in binned3['avg_emp']], rotation=45)

plt.grid(False)
plt.tight_layout()
plt.show()

# create data frame for the regression
reg_data3 = innovative_firms3[['gvkey', 'fyear', 'patents_per_emp', 'emp']].copy()
reg_data3 = reg_data3.replace([np.inf, -np.inf], np.nan).dropna()

# define log_emp
reg_data3 = reg_data3[reg_data3['emp'].div(1000) > 0]
reg_data3['log_emp'] = np.log(reg_data3['emp'].div(1000))

# set the panel index
reg_data3 = reg_data3.set_index(['gvkey', 'fyear'])

# run the regression
mod = PanelOLS.from_formula('patents_per_emp ~ log_emp + EntityEffects + TimeEffects', data=reg_data3)
results = mod.fit()
print(results.summary)

### Part 5: R&D Intensity by Firm Size

# define log_sale
innovative_firms4 = innovative_firms.copy()
innovative_firms4 = innovative_firms4[innovative_firms4['sale'] > 0]
innovative_firms4['log_sale'] = np.log(innovative_firms4['sale'])

# create the lagged log_sales variable
innovative_firms4['lag_log_sale'] = (innovative_firms4 .groupby('gvkey')['log_sale'] .shift(1))

# define log_xrd
innovative_firms4 = innovative_firms4[innovative_firms4['xrd'] > 0]
innovative_firms4['log_xrd'] = np.log(innovative_firms4['xrd'])

# create data frame for the regression
reg_data4 = innovative_firms4[['gvkey', 'fyear', 'lag_log_sale', 'log_xrd','sic']].copy()
reg_data4 = reg_data4.replace([np.inf, -np.inf], np.nan).dropna()

# set the panel index
reg_data4 = reg_data4.set_index(['gvkey', 'sic', 'fyear'])

import statsmodels.formula.api as smf

# Ensure your lagged sales is correctly set (this assumes it's already lagged)
data = innovative_firms4[['gvkey', 'fyear', 'sic', 'lag_log_sale', 'log_xrd']].dropna()

# Store results
results = []

# Loop through each year and run regression
for year in sorted(data['fyear'].unique()):
    df_year = data[data['fyear'] == year]
    
    # Use sector fixed effects via C(sic)
    model = smf.ols('log_xrd ~ lag_log_sale + C(sic)', data=df_year).fit()
    
    # Save beta1 (log_sale) and standard error
    beta1 = model.params['lag_log_sale']
    stderr = model.bse['lag_log_sale']
    
    results.append({'year': year, 'beta1': beta1, 'stderr': stderr})

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Plot
plt.figure(figsize=(10, 6))
plt.errorbar(results_df['year'], results_df['beta1'], yerr=results_df['stderr'], fmt='-o', capsize=4)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Coefficient on lag_log_sales')
plt.title('β₁ Estimates Over Time with Industry Fixed Effects')
plt.grid(True)
plt.tight_layout()
plt.show()

