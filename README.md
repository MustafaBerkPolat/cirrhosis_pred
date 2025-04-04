![Correlation Heatmap](https://github.com/user-attachments/assets/579416da-76df-4335-8ac1-15200c27c0b6)# Cirrhosis Stage Prediction with Machine Learning

## Overview

Cirrhosis is an advanced scarring on of liver tissue that can be caused by numerous conditions, most notably hepatitis and alcohol use. This project aims to predict the stage of a patient's cirrhosis via machine learning, utilizing indicators like age, gender, chemical imbalances in blood and urine, and presence of conditions like edema. 

## Data Used

A single dataset was used for both training and testing using cross-validation:
 - [Cirrhosis Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/cirrhosis-prediction-dataset)

This dataset contains the information collected from the Mayo Clinic trial in primary biliary cirrhosis (PBC) of the liver conducted between 1974 and 1984. A description of the clinical background for the trial and the covariates recorded here is in Chapter 0, especially Section 0.2 of Fleming and Harrington, Counting
Processes and Survival Analysis, Wiley, 1991. A more extended discussion can be found in Dickson, et al., Hepatology 10:1-7 (1989) and in Markus, et al., N Eng J of Med 320:1709-13 (1989).

A total of 424 PBC patients, referred to Mayo Clinic during that ten-year interval, met eligibility criteria for the randomized placebo-controlled trial of the drug D-penicillamine. The first 312 cases in the dataset participated in the randomized trial and contain largely complete data. The additional 112 cases did not participate in the clinical trial but consented to have basic measurements recorded and to be followed for survival. Six of those cases were lost to follow-up shortly after diagnosis, so the data here are on an additional 106 cases as well as the 312 randomized participants.

The exact information in the dataset are as follows:
1) ID: unique identifier
2) N_Days: number of days between registration and the earlier of death, transplantation, or study analysis time in July 1986
3) Status: status of the patient C (censored), CL (censored due to liver tx), or D (death)
4) Drug: type of drug D-penicillamine or placebo
5) Age: age in [days]
6) Sex: M (male) or F (female)
7) Ascites: presence of ascites N (No) or Y (Yes)
8) Hepatomegaly: presence of hepatomegaly N (No) or Y (Yes)
9) Spiders: presence of spiders N (No) or Y (Yes)
10) Edema: presence of edema N (no edema and no diuretic therapy for edema), S (edema present without diuretics, or edema resolved by diuretics), or Y (edema despite diuretic therapy)
11) Bilirubin: serum bilirubin in [mg/dl]
12) Cholesterol: serum cholesterol in [mg/dl]
13) Albumin: albumin in [gm/dl]
14) Copper: urine copper in [ug/day]
15) Alk_Phos: alkaline phosphatase in [U/liter]
16) SGOT: SGOT in [U/ml]
17) Triglycerides: triglicerides in [mg/dl]
18) Platelets: platelets per cubic [ml/1000]
19) Prothrombin: prothrombin time in seconds [s]
20) Stage: histologic stage of disease (1, 2, 3, or 4)

## Exploratory Data Analysis
As a first step, the individual columns' skewness is calculated to get an idea of how many outliers there are and how much they deviate from the norm. Most of the numerical measurements are heavily skewed, and out of 424 participants there are columns with anywhere between 10 and 150 NaN entries, so filling in the missing data with mean values would be unhealthy

|Numerical Columns|Skewness|
|-----------------|--------|
|ID            | 0.0|
|N_Days        | 0.47090441693552926|
|Age           | 0.08653818165415915|
|Bilirubin     | 2.7078487798071555|
|Cholesterol   | 3.3904966005028445|
|Albumin       | -0.4658471122696187|
|Copper        | 2.292478334674082|
|Alk_Phos      | 2.9784264300300034|
|SGOT          | 1.44222030514545|
|Tryglicerides | 2.510457567946973|
|Platelets     | 0.6247842191233116|
|Prothrombin   | 2.2152514903565708|
|Stage         | -0.49446476689189967|

---Dataframe Info---
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 20 columns):
| #   |Column         |Non-Null Count | Dtype  |
|---|----------------|----------------|--------|
| 0 |  ID            | 418 non-null |   int64  |
| 1 |  N_Days        | 418 non-null |   int64  |
| 2 |  Status        | 418 non-null |   object |
| 3 |  Drug          | 312 non-null |   object |
| 4 |  Age           | 418 non-null |   int64  |
| 5 |  Sex           | 418 non-null |   object |
| 6 |  Ascites       | 312 non-null |   object |
| 7 |  Hepatomegaly  | 312 non-null |   object |
| 8 |  Spiders       | 312 non-null |   object |
| 9 |  Edema         | 418 non-null |   object |
| 10 | Bilirubin     | 418 non-null |   float64|
| 11 | Cholesterol   | 284 non-null |   float64|
| 12 | Albumin       | 418 non-null |   float64|
| 13 | Copper        | 310 non-null |   float64|
| 14 | Alk_Phos      | 312 non-null |   float64|
| 15 | SGOT          | 312 non-null |   float64|
| 16 | Tryglicerides | 282 non-null |   float64|
| 17 | Platelets     | 407 non-null |   float64|
| 18 | Prothrombin   | 416 non-null |   float64|
| 19 | Stage         | 412 non-null |   float64|

The max values for Cholesterol, Copper, Alk_Phos, Bilirubin and Tryglicerides columns are anywhere between roughly 4 to 8 times higher than the 75% values, so there are plenty of outliers here. With the exception of Tryglicerides, these columns also have very high standard deviation values compared to their means, so the data seems to be very irregular in general.


---Dataframe Description---
|     | ID          | N_Days      | Age          | Bilirubin  | Cholesterol | Albumin  | Copper     |   Alk_Phos | SGOT       | Tryglicerides | Platelets | Prothrombin | Stage     |
|-----|-------------|-------------|--------------|------------|-------------|----------|------------|------------|------------|---------------|-----------|-------------|-----------|
|count|   418.000000|   418.000000|    418.000000|  418.000000|   284.000000|418.000000|  310.000000|  312.000000|  312.000000|     282.000000| 407.000000|   416.000000| 412.000000|  
|mean |   209.500000|  1917.782297|  18533.351675|    3.220813|   369.510563|  3.497440|   97.648387| 1982.655769|  122.556346|     124.702128| 257.024570|    10.731731|   3.024272|  
|std  |   120.810458|  1104.672992|   3815.845055|    4.407506|   231.944545|  0.424972|   85.613920| 2140.388824|   56.699525|      65.148639|  98.325585|     1.022000|   0.882042|  
|min  |     1.000000|    41.000000|   9598.000000|    0.300000|   120.000000|  1.960000|    4.000000|  289.000000|   26.350000|      33.000000|  62.000000|     9.000000|   1.000000|  
|25%  |   105.250000|  1092.750000|  15644.500000|    0.800000|   249.500000|  3.242500|   41.250000|  871.500000|   80.600000|      84.250000|  62.000000|     9.000000|   1.000000|  
|50%  |   209.500000|  1730.000000|  18628.000000|    1.400000|   309.500000|  3.530000|   73.000000| 1259.000000|  114.700000|     108.000000| 188.500000|    10.000000|   2.000000|  
|75%  |   313.750000|  2613.500000|  21272.500000|    3.400000|   400.000000|  3.770000|  123.000000| 1980.000000|  151.900000|     151.000000| 251.000000|    10.600000|   3.000000|  
|max  |   418.000000|  4795.000000|  28650.000000|   28.000000|  1775.000000|  4.640000|  588.000000|13862.400000|  457.250000|     598.000000| 721.000000|    18.000000|   4.000000|  

At a first glance at the pair plots of our numerical data, there aren't any strong correlations or clustering to take into immediate consideration. It may seem like the inclusion of the ID column is an oversight, but looking closely, we can see that there is a very clear cut-off line for the N-Days column where its upper bound somewhat linearly increases with ID until roughly the 300 ID mark, but then it resets and starts linearly increasing again, and past this 300-ID line we have no data for some of the columns like Cholesterol, Copper and Alk_Phos. This is in line with the dataset description mentioning that the first 312 participants contain more complete data whereas the remaining 106 did not participate in the clinical trial of the drug D-penicillamine, only giving consent to basic measurements. This can cause our predictions to be biased or be performed with incomplete information, but given the size of the dataset, I chose not to exclude the last 106 participants.
![Numerical Data Pair Plots](https://github.com/user-attachments/assets/29c7a8d3-b30c-459a-88a4-a5be7c90799c)


This dataset has plenty of outliers and some non-normal distributed data, so when testing for correlation I elected to check the Spearman correlation coefficient. The N_Days feature has a somewhat positive correlation with Albumin, and negative correlations with Bilirubin, Copper and cirrhosis stage, but these correlations are not strong enough to consider dimensionality reduction solely based off of them.
![Correlation Heatmap](https://github.com/user-attachments/assets/00832fab-1c4b-4118-bc9b-43b5362a42c4)

|Stage |Count |
|------|------|
|3.0   |   155|
|4.0   |   144|
|2.0   |    92|
|1.0   |    21|

Looking at the distribution of the cirrhosis stages, stages 3 and 4 are over-represented in the dataset while stage 1 is very low. Using log loss to evaluate model performance is sensible here, but oversampling is not necessarily a good choice here as the more prevalent group is the higher stages of cirrhosis, and it is better to have high false positives if it means cirrhosis cases are not missed by the model. The "cost" of misdiagnosis where a patient is thought to have a more advanced case of cirrhosis than they do is relatively minor compared to the "cost" of a more advanced case being skipped. One adjustment we can do here is using a balanced sampling method for the random forest classifier.

## Data Preprocessing

The details for these steps will be superficial as the code includes more details and the exact steps taken.


There are both numerical and categorical data in this dataset, with a lot of blank entries and outliers. In order to avoid problems with the prediction, the first step is filling in the missing numerical data with the median values (as explained before) and the missing categorical data with the mode values. Then, a function is defined to calculate the z-scores of every numerical column, and clamp the outlier entries to within 3 standard deviations of their means. Finally, the Stage column is mapped to integers 0 through 3 to avoid any potential issues with the classification algorithms inconsistently mapping them.
