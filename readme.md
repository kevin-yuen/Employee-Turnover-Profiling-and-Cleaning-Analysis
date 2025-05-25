### Objective
A data profiling and cleaning project focused on employee turnover. This repository includes the analysis of raw HR data, addressing and fixing data quality issues such as missing values, duplicates, formatting errors, inconsistent entries, and outliers. The cleaned dataset is prepared for further analysis and predictive modeling to support employee retention strategies.

### Data Profiling Techniques
##### Missing values
- np.isnan() / pd.isnull()
- missingno library - purpose: visualize missingness patterns using .matrix(), .dendrogram(), and .heatmap()

##### Duplicates
- .duplicated()

##### Formatting errors
- recordlinkage library - purpose: find data similarities and dissimilarities

##### Inconsistency entries
- .set() / .difference

##### Outliers
- matplotlib library - purpose: identify outliers using .boxplot()
- scipy package - purpose: calculate z-scores using .zscore() to find outliers
- seaborn library - purpose: visualize data distribution
- IQR method

### Data Cleaning Techniques
##### Missing values
- imputation techniques (mean, median, mode)

##### Outliers
- capping (Winsorization) technique
- binning technique