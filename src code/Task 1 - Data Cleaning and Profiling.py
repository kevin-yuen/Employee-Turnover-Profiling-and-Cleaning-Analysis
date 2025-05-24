#!/usr/bin/env python
# coding: utf-8

# In[23]:


# import libraries

import pandas as pd
import missingno as msno
import numpy as np
import recordlinkage
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


# In[24]:


# enable to display all columns

pd.set_option('display.max_columns', None)


# # Data Profiling

# ##### A.  Profile data by doing the following:
# 
# ##### 1.  Review the data dictionary in the attached "Employee Turnover Considerations and Dictionary" document and do the following:
# ##### a.  Describe the general characteristics of the initial dataset (e.g., rows, columns).
# ##### -->> rows, columns, descriptive stats, value count for categorical variables, non-null count for each variable
# ##### b.  Indicate the data type and data subtype for each variable.
# ##### c.  Provide a sample of observable values for each variable.
# ##### -->> 20 random sample values for each variable

# In[25]:


# define new column names for renaming columns

col_names = [
    'employee_number',
    'age',
    'tenure',
    'turnover',
    'hourly_rate',
    'hours_weekly',
    'compensation_type',
    'annual_salary',
    'driving_commuter_distance',
    'job_role_area',
    'gender',
    'marital_status',
    'num_companies_previously_worked',
    'annual_professonal_dev_hrs',
    'pay_check_method',
    'text_message_optin'
]


# In[26]:


# read and load source csv data file as a pandas dataframe

emp_to_df = pd.read_csv(
    '/Users/tulipz123/Documents/WGU-MDSA/D599/Task 1/source/Employee Turnover Dataset.csv', 
    sep=',',
    skiprows=1,
    names=col_names)


# In[27]:


# display dataset

emp_to_df


# In[28]:


# check number of rows loaded

display(len(emp_to_df))


# In[29]:


# check number of columns loaded, data type for each variable, and non-null count for each variable

display(emp_to_df.info())


# In[30]:


# cast data type of 'hourly_rate' to float before further data profiling

emp_to_df['hourly_rate'] = emp_to_df['hourly_rate'].str.slice(start=1).astype('float64')

emp_to_df


# In[31]:


# again, verify data type for each variable after casting

emp_to_df.dtypes


# In[32]:


# get descriptive statistics

emp_to_df.describe()


# In[33]:


# get value count for categorical variables

for col in emp_to_df.select_dtypes(include='object').columns:
    print(f'Column Name: {col}\n{emp_to_df[col].value_counts(dropna=False)}\n')  # some are 'duplicate' values (e.g. 'Information Technology' vs 'InformationTechnology')


# In[34]:


# generate a sample of 20 observable values for each variable

rand_sample_val_df = emp_to_df.sample(n=20).reset_index(drop=True)

rand_sample_val_df


# # Data Cleaning and Plan

# ##### B.  Inspect the dataset through data cleaning techniques for all duplicate entries, missing values, inconsistent entries, formatting errors, and outliers and do the following:
# 
# ##### 1.  Explain how you inspected the dataset for each of the quality issues listed in part B.
# ##### 2.  List your findings for each quality issue listed in part B.

# ### Inspect each categorical variable for inconsistent entries using .set() / .difference()

# In[35]:


# define a function to find inconsistent entries
def find_inconsistent_values(col_name, expected_values):
    # get unique values for 'job_role_area' from the dataset
    actual_values = set(emp_to_df[col_name])
    
    # find inconsistent entries (i.e. unexpected values)
    inconsistent_values = actual_values.difference(expected_values)
    
    return inconsistent_values

# find inconsistent entries for 'job_role_area'
expected_values_jra = ['Healthcare', 'Human Resources', 'Information Technology', 'Laboratory', 'Manufacturing', 'Marketing', 'Research', 'Sales']
inconsistent_values_jra = find_inconsistent_values('job_role_area', expected_values_jra)

print(f'job_role_area: {inconsistent_values_jra}')

# find inconsistent entries for 'pay_check_method'
expected_values_pcm = ['Mailed Check', 'Direct Deposit']
inconsistent_values_pcm = find_inconsistent_values('pay_check_method', expected_values_pcm)

print(f'pay_check_method: {inconsistent_values_pcm}')


# ### Inspect each categorical variable for formatting errors using Record Linkage

# In[36]:


# define a function to categorize job_role_area and pay_check_method
# the defined category will be used to create candidate pairs
# further comparison will be implemented to ensure they are the same value

# for job_role_area, the category will be defined with the following assumptions:
# 1. value contains 'Human' belongs to 'Human Resource'
# 2. value contains 'Information' belongs to 'Information Technology'

# for pay_check_method, the category will be defined with the following assumptions:
# 1. value contains 'Mail' belongs to 'Mailed Check'
# 2. value contains 'Direct' belongs to 'Direct Deposit'

def categorize(row, col_name, categories):
    value = str(row[col_name])
    
    if categories[0].split()[0] in value:
        return categories[0]
    elif categories[1].split()[0] in value:
        return categories[1]
    else:
        return value

job_role_area_categories = ['Human Resources', 'Information Technology']
pay_check_method_categories = ['Mail Check', 'Direct Deposit']

emp_to_df['job_role_area_category'] = emp_to_df.apply(lambda row: categorize(row, 'job_role_area', job_role_area_categories), axis=1)
emp_to_df['pay_check_method_category'] = emp_to_df.apply(lambda row: categorize(row, 'pay_check_method', pay_check_method_categories), axis=1)

emp_to_df


# ##### data similarities comparison logic starts here...

# In[37]:


# create a deep copy of emp_to_df
emp_to_df_copy = emp_to_df.copy()

# initialize Compare() object
job_role_comp = recordlinkage.Compare()
pay_check_comp = recordlinkage.Compare()


# In[38]:


# define a function to create candidate pairs for comparison
def create_candidate_pairs(col_name):
    indexer = recordlinkage.Index()
    indexer.block(col_name)
    candidate_links = indexer.index(emp_to_df, emp_to_df_copy)
    
    return candidate_links


# In[17]:


# create candidate pairs for 'job_role_area'
cand_links_job_role_area = create_candidate_pairs('job_role_area_category')

# initialize similarity measurement algorithms for 'job_role_area'
job_role_comp.string('job_role_area', 'job_role_area', threshold=0.85, label='job_role_similar')
job_role_comp.exact('job_role_area', 'job_role_area', label='job_role_exact')

features_job_role = job_role_comp.compute(cand_links_job_role_area, emp_to_df, emp_to_df_copy)


# In[39]:


# get data dissimilarities for 'job_role_area'

features_job_role[features_job_role['job_role_similar'] < 1]


# In[40]:


# get exact mismatches for 'job_role_area'

features_job_role[features_job_role['job_role_exact'] < 1]  


# ##### example of data not exact match

# In[41]:


emp_to_df.loc[2:2]


# In[42]:


emp_to_df.loc[388:388]


# In[22]:


# create candidate pairs for 'pay_check_method'
cand_links_pay_check_method = create_candidate_pairs('pay_check_method_category')

# initialize similarity measurement algorithms for 'pay_check_method'
pay_check_comp.string('pay_check_method', 'pay_check_method', threshold=0.85, label='pay_check_similar')
pay_check_comp.exact('pay_check_method', 'pay_check_method', label='pay_check_exact')

features_pay_check = pay_check_comp.compute(cand_links_pay_check_method, emp_to_df, emp_to_df_copy)


# In[43]:


# get data dissimilarities for 'pay_check_method'

features_pay_check[features_pay_check['pay_check_similar'] < 1]


# ##### example of data dissimilarities

# In[44]:


emp_to_df.loc[0:0]


# In[45]:


emp_to_df.loc[15:15]


# ### Inspect each variable for missing values
# 
# ##### per emp_to_df.info()...
# ##### total entries = 10,199
# ##### below columns have less than 10,199 non-null count:
# #####  num_companies_previously_worked - 9534  non-null
# #####  annual_professonal_dev_hrs - 8230  non-null
# #####  text_message_optin - 7933  non-null 

# In[46]:


emp_to_df.info()


# In[47]:


# inspect using np.isnan() / pd.isnull()

null_columns = ['num_companies_previously_worked', 'annual_professonal_dev_hrs', 'text_message_optin']

for column in null_columns:
    is_null = np.isnan(emp_to_df[column]) if emp_to_df[column].dtype != 'object' else pd.isnull(emp_to_df[column])
    
    cnt_of_nulls = is_null[is_null == True].count()
    cnt_of_non_nulls = is_null[is_null == False].count()
    
    print(f'Column: {column}')
    print(f'Count of nulls: {cnt_of_nulls}')
    print(f'Count of non-nulls: {cnt_of_non_nulls}')
    print(f'Total count (cnt of nulls + non-nulls): {cnt_of_nulls + cnt_of_non_nulls}\n')


# In[48]:


# visualize missing values

# revert back to the original dataframe by dropping the columns
emp_to_df.drop(labels=['job_role_area_category', 'pay_check_method_category'], inplace=True, axis=1)
# emp_to_df

msno.matrix(emp_to_df)


# In[49]:


msno.heatmap(emp_to_df)  # missing values appear independently and randomly (columns have no correlation)


# In[50]:


msno.dendrogram(emp_to_df)

# Summary of the Dendrogram:
# the missingness pattern of marital_status and pay_check_method is very similar
# marital_status, pay_check_method, and num_companies_previously_worked form one cluster
# annual_professional_dev_hrs + (marital_status, pay_check_method, num_companies_previously_worked) form another cluster
# text_message_optin + annual_professional_dev_hrs + (marital_status, pay_check_method, num_companies_previously_worked) form another cluster
# the missingness pattern between marital_status, pay_check_method, and num_companies_previously_worked are the most similar, comparing to the other clusters
# the missingness pattern between text_message_optin + annual_professional_dev_hrs + (marital_status, pay_check_method, num_companies_previously_worked) are the least similar, comparing to the other clusters


# ### Clean each variable before inspecting duplicate entries.

# In[51]:


# replace bad formatted values for consistency

columns_to_clean = ['job_role_area', 'pay_check_method', 'gender']

replacement_map = {
    'job_role_area': {
        'Information Technology': ['InformationTechnology', 'Information_Technology'],
        'Human Resources': ['HumanResources', 'Human_Resources']
    },
    'pay_check_method': {
        'Mailed Check': ['Mail Check', 'Mail_Check', 'MailedCheck'],
        'Direct Deposit': ['DirectDeposit', 'Direct_Deposit']
    }
}

for column in columns_to_clean:
    if column != 'gender':
        replacements = replacement_map[column]

        for new_val, old_vals in replacements.items():
            emp_to_df[column].replace(old_vals, value=new_val, inplace=True)
    else:
        emp_to_df[column].replace('Prefer Not to Answer', value=pd.NA, inplace=True)


# In[52]:


# identify duplicate rows

emp_to_df['is_duplicated'] = emp_to_df.duplicated(keep='first')

emp_to_df[['employee_number', 'is_duplicated']]


# In[53]:


emp_to_df[emp_to_df['employee_number'] == 99]


# #### Inspect for outliers

# In[54]:


# define columns which might have outliers

columns = [
    'tenure',
    'hourly_rate', 
    'hours_weekly',
    'annual_salary', 
    'driving_commuter_distance',
    'num_companies_previously_worked',
    'annual_professonal_dev_hrs'
]


# In[55]:


# identify outliers using boxplot

for column in columns:
    if column == 'num_companies_previously_worked' or column == 'annual_professonal_dev_hrs':
        # num_companies_previously_worked and annual_professonal_dev_hrs have nulls
        # nulls need to be dropped before finding outliers
        drop_nan_df = pd.DataFrame()

        drop_nan_df[column] = emp_to_df[column]
        drop_nan_df.dropna(axis=0, subset=column, inplace=True)
        
        plt.boxplot(drop_nan_df[column])
    else:   
        # non-null columns...|
        plt.boxplot(emp_to_df[column])
        
    print(f'Boxplot for {column}')
    plt.show()


# In[56]:


# identify outliers using z-scores

zscore_columns = []  # for finding outliers

# find z-score for each column that might have outliers
for column in columns:
    new_column = f'{column}_zscore'
    zscore_columns.append(new_column)
    
    if emp_to_df[column].std() == 0:
        emp_to_df[new_column] = 0
    else:
        emp_to_df[new_column] = zscore(emp_to_df[column], nan_policy='omit')

for zscore_column in zscore_columns:
    orig_col_name = zscore_column.rsplit('_', 1)[0]
    outlier_column = f'is_{orig_col_name}_outlier'
    
    # identify outliers based on the z-score
    is_outlier = (emp_to_df[zscore_column] < -3) | (emp_to_df[zscore_column] > 3)
    emp_to_df[outlier_column] = is_outlier

    # return outliers per the outlier indicator
    outliers = emp_to_df[outlier_column] == True
    
    if len(emp_to_df.loc[outliers]) > 0:
        print(emp_to_df.loc[outliers, zscore_column])


# ##### The distribution in the histogram for annual_salary is right-skewed; hence, use IQR method to find outliers

# In[57]:


# histogram distribution for annual_salary...right-skewed...

sns.histplot(emp_to_df['annual_salary'], kde=True)


# In[58]:


# histogram distribution for driving_commuter_distance...right-skewed...

sns.histplot(emp_to_df['driving_commuter_distance'], kde=True)


# In[59]:


# use IQR method to find outliers

outlier_series_list = []

def find_outliers(column):
    Q1 = emp_to_df[column].quantile(0.25)
    Q3 = emp_to_df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    outliers = emp_to_df[(emp_to_df[column] < Q1 - 1.5 * IQR) | (emp_to_df[column] > Q3 + 1.5 * IQR)]
    
    output = outliers[[f'{column}', f'{column}_zscore']]
    
    if len(output) > 0:
        print(f'{column}:\n{output}')
        
        outlier_series_list.append(output)  # store outliers to a variable, which will later be used to fix data quality

for column in columns:
    find_outliers(column)


# ### Fix data quality issues

# #### Handle missing data using imputation techniques

# In[60]:


def verify_counts(col_name):
    return emp_to_df[col_name].value_counts(dropna=False)
    
def impute_missingness(col_name, fill_in_value):
    emp_to_df[col_name].fillna(value=fill_in_value, inplace=True)
    
def compute_median(col_name):
    return np.nanmedian(emp_to_df[col_name])
    
def determine_distribution(col_name):
    return sns.histplot(emp_to_df[col_name], kde=True)


# ##### text_message_optin

# In[61]:


# verify counts before imputation
verify_counts('text_message_optin')


# In[62]:


# since count of 'Yes' > 'No', handle missingness using mode imputation
impute_missingness('text_message_optin', 'Yes')


# In[63]:


# verify counts after imputation
verify_counts('text_message_optin')


# ##### gender

# In[64]:


# verify counts before imputation
verify_counts('gender')


# In[65]:


# since count of 'Female' > 'Male', handle missingness using mode imputation
impute_missingness('gender', 'Female')


# In[66]:


# verify counts after imputation
verify_counts('gender')


# ###### num_companies_previously_worked

# In[67]:


# verify counts before imputation
verify_counts('num_companies_previously_worked')


# In[68]:


# determine distribution (e.g. normal, skewed)
determine_distribution('num_companies_previously_worked')


# In[69]:


# right-skewed, use median to impute missingness

# compute for median
median = compute_median('num_companies_previously_worked')

# impute missingness with the median
impute_missingness('num_companies_previously_worked', median)

# verify after imputation
verify_counts('num_companies_previously_worked')


# ##### annual_professonal_dev_hrs

# In[70]:


# verify counts before imputation
verify_counts('annual_professonal_dev_hrs')


# In[71]:


# determine distribution (e.g. normal, skewed)
determine_distribution('annual_professonal_dev_hrs')


# In[72]:


# left-skewed, use median to impute missingness

# compute for median
median = compute_median('annual_professonal_dev_hrs')

# impute missingness with the median
impute_missingness('annual_professonal_dev_hrs', median)

# verify after imputation
verify_counts('annual_professonal_dev_hrs')


# #### Cast the data type of each column to an appropriate type

# In[73]:


emp_to_df.dtypes


# In[74]:


emp_to_df['num_companies_previously_worked'] = emp_to_df['num_companies_previously_worked'].astype('int64')

emp_to_df['num_companies_previously_worked'].dtype


# In[75]:


# verify after casting
emp_to_df.dtypes


# #### Drop duplicates

# In[76]:


# verify the count of duplicates
emp_to_df[emp_to_df['is_duplicated'] == True]


# In[77]:


# drop duplicates
emp_to_df.drop_duplicates(subset=[col for col in emp_to_df.columns if col != 'is_duplicated'], keep='first', inplace=True)


# In[78]:


# verify the count of duplicates after dropping duplicates
emp_to_df


# #### Standardize value format / Fix inconsistencies
# 
# ##### logic already implemented for 'job_role_area' and 'pay_check_method'. see above.

# #### Fix outliers

# ##### Assuming the company needs to perform regression to offer higher salaries based on other factors like work experience and educational level in order to retain its employees, use capping to fix the outliers for annual_salary.
# #####  Per above, annual_salary = right-skewed.

# In[79]:


# outliers for 'annual_salary'
annual_salary_outliers_series = outlier_series_list[0]


# In[80]:


# verify the max salary before capping
annual_salary_outliers_series['annual_salary'].max()


# In[81]:


# determine upper cap limit
upper_threshold = annual_salary_outliers_series['annual_salary'].quantile(0.95)

# replace any value above the upper cap limit
annual_salary_outliers_series['annual_salary'] = annual_salary_outliers_series['annual_salary'].clip(upper=upper_threshold)


# In[82]:


# verify the max annual_salary after capping
annual_salary_outliers_series['annual_salary'].max()


# In[83]:


# update the original dataframe with the capping result
emp_to_df.loc[annual_salary_outliers_series.index, 'annual_salary'] = annual_salary_outliers_series['annual_salary']

# verify the max annual_salary of the original dataframe after capping
emp_to_df['annual_salary'].max()


# ##### Assuming the company needs to understand the relationship between commute distance and other variables such as salary and turnover rate, use binning for driving_commuter_distance to support dashboard creation.

# In[84]:


# outliers for 'driving_commuter_distance'
commute_distance_outliers_series = outlier_series_list[1]


# In[85]:


commute_distance_outliers_series[commute_distance_outliers_series['driving_commuter_distance'] <= 0]  # distance less than 0, which doesn't make any sense


# ##### Per above, distance values equal to or less than 0 needs to be fixed before binning

# In[86]:


# find rows with commute distance that is 0 or less
emp_to_df[emp_to_df['driving_commuter_distance'] <= 0]


# In[87]:


# assuming every employee needs to commute at least 0.1 mile for work,
# convert distance values that are equal to or less than 0 to 0.1
emp_to_df.loc[emp_to_df['driving_commuter_distance'] <= 0, 'driving_commuter_distance'] = 0.1


# In[88]:


# verify updates
emp_to_df[emp_to_df['driving_commuter_distance'] == 0.1]


# In[89]:


# start binning...
emp_to_df['commute_distance_category'] = pd.cut(emp_to_df['driving_commuter_distance'], bins=10)

emp_to_df


# ### Save final output as .csv

# In[90]:


# drop unnecessary columns

emp_to_df = emp_to_df.drop(columns=[col for col in emp_to_df.columns if col.endswith('_zscore')
                                         or col.endswith('_outlier')
                                         or col == 'is_duplicated'])


# In[91]:


emp_to_df.to_csv('../output/employer_turnover.csv', index=False)


# In[ ]:




