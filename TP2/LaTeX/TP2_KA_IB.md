<h1><center> TP2: Pandas, data analysis library </center></h1>

The deadline for report submission is Tuesday, December 8th 2020.

Note: the goal of this TP is to become familiar with 'pandas' class in Python. This library is often used for data analysis and is convenient for manipulation tool.
We consider a case study based on a dataset that contains information about bookings of two hotels. Hotel 1 is a resort hotel and Hotel 2 is a city hotel in Portugal. The dataset was released by https://www.sciencedirect.com/science/article/pii/S2352340918315191.

We first list the basic function in pandas. PART 1 aims at using pandas as a visualization tools to a better understanding of data. PART 2 shows how easy it is to combine "pandas" dataframes and "sklearn" models to build additional features and predict. 

As a homework, we propose you a very concret problem which is open and for which we are waiting for your creativity (as usual)!


```python
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib notebook
#---------------------
#For JupyterLab
%matplotlib inline

import random
random.seed(1) #to fix random and have the same results for both of us 
```

# Short intro into pandas


```python
data = pd.read_csv("data/bookings.csv") 
```


```python
data.dtypes
```




    hotel                              object
    is_canceled                         int64
    lead_time                           int64
    arrival_date_year                   int64
    arrival_date_month                 object
    arrival_date_week_number            int64
    arrival_date_day_of_month           int64
    stays_in_weekend_nights             int64
    stays_in_week_nights                int64
    adults                              int64
    children                          float64
    babies                              int64
    meal                               object
    country                            object
    market_segment                     object
    distribution_channel               object
    is_repeated_guest                   int64
    previous_cancellations              int64
    previous_bookings_not_canceled      int64
    reserved_room_type                 object
    assigned_room_type                 object
    booking_changes                     int64
    deposit_type                       object
    agent                             float64
    company                           float64
    days_in_waiting_list                int64
    customer_type                      object
    adr                               float64
    required_car_parking_spaces         int64
    total_of_special_requests           int64
    reservation_status                 object
    reservation_status_date            object
    dtype: object




```python
data.head(5) # print first 5 entries of the dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>arrival_date_week_number</th>
      <th>arrival_date_day_of_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>...</th>
      <th>deposit_type</th>
      <th>agent</th>
      <th>company</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>reservation_status</th>
      <th>reservation_status_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>304.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>240.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>Check-Out</td>
      <td>2015-07-03</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
data.columns # print column names
```




    Index(['hotel', 'is_canceled', 'lead_time', 'arrival_date_year',
           'arrival_date_month', 'arrival_date_week_number',
           'arrival_date_day_of_month', 'stays_in_weekend_nights',
           'stays_in_week_nights', 'adults', 'children', 'babies', 'meal',
           'country', 'market_segment', 'distribution_channel',
           'is_repeated_guest', 'previous_cancellations',
           'previous_bookings_not_canceled', 'reserved_room_type',
           'assigned_room_type', 'booking_changes', 'deposit_type', 'agent',
           'company', 'days_in_waiting_list', 'customer_type', 'adr',
           'required_car_parking_spaces', 'total_of_special_requests',
           'reservation_status', 'reservation_status_date'],
          dtype='object')



This dataset contains information about bookings of two hotels. Hotel 1 is a resort hotel and Hotel 2 is a city hotel in Portugal. The dataset was released by https://www.sciencedirect.com/science/article/pii/S2352340918315191.

There are 32 columns in this dataset:

1. **hotel** -- one of the two hotels
2. **is_canceled** -- Value indicating if the booking was canceled (1) or not (0)
3. **lead_time** -- Number of days that elapsed between the entering date of the booking into the PMS and the arrival date
4. ....

For the full description of each column please see: https://www.kaggle.com/jessemostipak/hotel-booking-demand


```python
data['country'] # we can also print each column of the dataset
```




    0         PRT
    1         PRT
    2         GBR
    3         GBR
    4         GBR
             ... 
    119385    BEL
    119386    FRA
    119387    DEU
    119388    GBR
    119389    DEU
    Name: country, Length: 119390, dtype: object




```python
data['country'].unique() # list all unique values in the column
```




    array(['PRT', 'GBR', 'USA', 'ESP', 'IRL', 'FRA', nan, 'ROU', 'NOR', 'OMN',
           'ARG', 'POL', 'DEU', 'BEL', 'CHE', 'CN', 'GRC', 'ITA', 'NLD',
           'DNK', 'RUS', 'SWE', 'AUS', 'EST', 'CZE', 'BRA', 'FIN', 'MOZ',
           'BWA', 'LUX', 'SVN', 'ALB', 'IND', 'CHN', 'MEX', 'MAR', 'UKR',
           'SMR', 'LVA', 'PRI', 'SRB', 'CHL', 'AUT', 'BLR', 'LTU', 'TUR',
           'ZAF', 'AGO', 'ISR', 'CYM', 'ZMB', 'CPV', 'ZWE', 'DZA', 'KOR',
           'CRI', 'HUN', 'ARE', 'TUN', 'JAM', 'HRV', 'HKG', 'IRN', 'GEO',
           'AND', 'GIB', 'URY', 'JEY', 'CAF', 'CYP', 'COL', 'GGY', 'KWT',
           'NGA', 'MDV', 'VEN', 'SVK', 'FJI', 'KAZ', 'PAK', 'IDN', 'LBN',
           'PHL', 'SEN', 'SYC', 'AZE', 'BHR', 'NZL', 'THA', 'DOM', 'MKD',
           'MYS', 'ARM', 'JPN', 'LKA', 'CUB', 'CMR', 'BIH', 'MUS', 'COM',
           'SUR', 'UGA', 'BGR', 'CIV', 'JOR', 'SYR', 'SGP', 'BDI', 'SAU',
           'VNM', 'PLW', 'QAT', 'EGY', 'PER', 'MLT', 'MWI', 'ECU', 'MDG',
           'ISL', 'UZB', 'NPL', 'BHS', 'MAC', 'TGO', 'TWN', 'DJI', 'STP',
           'KNA', 'ETH', 'IRQ', 'HND', 'RWA', 'KHM', 'MCO', 'BGD', 'IMN',
           'TJK', 'NIC', 'BEN', 'VGB', 'TZA', 'GAB', 'GHA', 'TMP', 'GLP',
           'KEN', 'LIE', 'GNB', 'MNE', 'UMI', 'MYT', 'FRO', 'MMR', 'PAN',
           'BFA', 'LBY', 'MLI', 'NAM', 'BOL', 'PRY', 'BRB', 'ABW', 'AIA',
           'SLV', 'DMA', 'PYF', 'GUY', 'LCA', 'ATA', 'GTM', 'ASM', 'MRT',
           'NCL', 'KIR', 'SDN', 'ATF', 'SLE', 'LAO'], dtype=object)




```python
data.count()
```




    hotel                             119390
    is_canceled                       119390
    lead_time                         119390
    arrival_date_year                 119390
    arrival_date_month                119390
    arrival_date_week_number          119390
    arrival_date_day_of_month         119390
    stays_in_weekend_nights           119390
    stays_in_week_nights              119390
    adults                            119390
    children                          119386
    babies                            119390
    meal                              119390
    country                           118902
    market_segment                    119390
    distribution_channel              119390
    is_repeated_guest                 119390
    previous_cancellations            119390
    previous_bookings_not_canceled    119390
    reserved_room_type                119390
    assigned_room_type                119390
    booking_changes                   119390
    deposit_type                      119390
    agent                             103050
    company                             6797
    days_in_waiting_list              119390
    customer_type                     119390
    adr                               119390
    required_car_parking_spaces       119390
    total_of_special_requests         119390
    reservation_status                119390
    reservation_status_date           119390
    dtype: int64



This dataset contains 119390 different reservations. Some of the reservations have missing values.



```python
data.values # A data frame can be converted into a numpy array by calling the values attribute:
```




    array([['Resort Hotel', 0, 342, ..., 0, 'Check-Out', '2015-07-01'],
           ['Resort Hotel', 0, 737, ..., 0, 'Check-Out', '2015-07-01'],
           ['Resort Hotel', 0, 7, ..., 0, 'Check-Out', '2015-07-02'],
           ...,
           ['City Hotel', 0, 34, ..., 4, 'Check-Out', '2017-09-07'],
           ['City Hotel', 0, 109, ..., 0, 'Check-Out', '2017-09-07'],
           ['City Hotel', 0, 205, ..., 2, 'Check-Out', '2017-09-07']],
          dtype=object)



However this array cannot be directly fed to a scikit-learn model.

1. the values are heterogeneous (strings for categories, integers, and floating point numbers)
2. some attribute values are missing

# Predicting cancellation: Part I -- visualization

**Our goals** The goal of this part is to provide few examples of visualization combining ```pandas```, ```matplotlib```, ```seaborn```. 

We will look at a very natural and practical task -- predicting cancellation of a given reservation. Of course, the first instinct of the modern 'ML practitioner' is to throw all the data to some neural net and perform bunch of fine-tuning. There are really a lot of problems with such an approach. As a general rule of thumbs: explore your data before building ML pipelines! It is alway more interesting to investigate the data and find human-interpretable patterns.

Pandas allows to manipulate the dataset in a very convenient manner. Those familiar with SQL will certainly appreciate it! 

As the first visualization task let us understand which monthes have the most amount of cancellations.
We start by creating a new dataset that contains the information that we would like to plot.


```python

'''
 We create two datasets for each hotel .groupby("arrival_date_month") will group observations by the month
 and .count() function will simply count the amount of reservations for each month
'''
n_reserv_H1 = data.loc[(data["hotel"] == "Resort Hotel")].groupby("arrival_date_month")["hotel"].count()
n_cancel_H1 = data.loc[(data["hotel"] == "Resort Hotel")].groupby("arrival_date_month")["is_canceled"].sum()

# same thing we do for the second hotel
n_reserv_H2 = data.loc[(data["hotel"] == "City Hotel")].groupby("arrival_date_month")["hotel"].count()
n_cancel_H2 = data.loc[(data["hotel"] == "City Hotel")].groupby("arrival_date_month")["is_canceled"].sum()
```


```python
n_reserv_H1.head() # again we can have a look at the top 5 entries
```




    arrival_date_month
    April       3609
    August      4894
    December    2648
    February    3103
    January     2193
    Name: hotel, dtype: int64




```python
n_cancel_H1.head()
```




    arrival_date_month
    April       1059
    August      1637
    December     631
    February     795
    January      325
    Name: is_canceled, dtype: int64




```python
n_reserv_H1['April'] # number of reservation for Resort Hotel for April
```




    3609




```python
n_cancel_H1['April'] # number of cancelled reservations for Resort Hotel
```




    1059




```python
# finally, we gather everything together


data_visualH1 = pd.DataFrame({"hotel": "Resort Hotel",
                                "month": list(n_reserv_H1.index),
                                "n_booking": list(n_reserv_H1.values),
                                "n_cancel": list(n_cancel_H1.values)})
data_visualH2 = pd.DataFrame({"hotel": "City Hotel",
                                "month": list(n_reserv_H2.index),
                                "n_booking": list(n_reserv_H2.values),
                                "n_cancel": list(n_cancel_H2.values)})
data_visual = pd.concat([data_visualH1, data_visualH2], ignore_index=True)

# notice how easy it is to add a new column. We simply write the following
data_visual["percent_cancel"] = data_visual["n_cancel"] / data_visual["n_booking"] * 100 # percent of cancelations
```


```python
data_visual.head() # our final dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hotel</th>
      <th>month</th>
      <th>n_booking</th>
      <th>n_cancel</th>
      <th>percent_cancel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>April</td>
      <td>3609</td>
      <td>1059</td>
      <td>29.343308</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>August</td>
      <td>4894</td>
      <td>1637</td>
      <td>33.449121</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel</td>
      <td>December</td>
      <td>2648</td>
      <td>631</td>
      <td>23.829305</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel</td>
      <td>February</td>
      <td>3103</td>
      <td>795</td>
      <td>25.620367</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel</td>
      <td>January</td>
      <td>2193</td>
      <td>325</td>
      <td>14.819881</td>
    </tr>
  </tbody>
</table>
</div>




```python
# plotting is simple once we have good dataset.
plt.figure(figsize=(6, 6))
sns.barplot(x = "month", y = "percent_cancel" , hue="hotel",
            hue_order = ["Resort Hotel", "City Hotel"], data=data_visual)
plt.title("Cancelations per month")
plt.xticks(rotation=45)
plt.ylabel("Cancelations [%]")
plt.legend()
plt.show()
```


    
![png](TP2_KA_IB_files/TP2_KA_IB_25_0.png)
    


Previous plot is nice, but the ordering of the month is very annoying!

**Question 1.** Propose a solution that will re-order the barplot above using standard month ordering. Hint: use ```pd.Categorical()``` function of pandas.

**Answer:** we have to consider the `month` variable as a categorical and ordered variable (as a factor). The following lines allow to do so. And then we just plot the new `month_categorical` variable using the previous code.


```python
# create a vector of ordered months
months_ordered_from_july = data['arrival_date_month'].unique()
months_ordered_from_january = [months_ordered_from_july[i] for i in [*range(6, 12), *range(0, 6)]]
print(months_ordered_from_january)

#create a new 'month' variable which is categorical
data_visual["month_categorical"] = pd.Categorical(data_visual["month"], ordered=True,
                   categories=months_ordered_from_january)
```

    ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    


```python
plt.figure(figsize=(6, 6))
sns.barplot(x = "month_categorical", y = "percent_cancel" , hue="hotel",
            hue_order = ["Resort Hotel", "City Hotel"], data=data_visual)
plt.title("Cancelations per month")
plt.xticks(rotation=45)
plt.ylabel("Cancelations [%]")
plt.legend()
plt.show()
```


    
![png](TP2_KA_IB_files/TP2_KA_IB_29_0.png)
    


**Question 2.** Provide interpretation of the above plot.

**Answer:** The cancelations at the City hotel are, in proportion, higher than in the Resort Hotel. For the Resort hotel, the cancelations are in proportion higher in Summer and Spring than in Winter and Fall. On the contrary, this seasonality is less clear for the City hotel. We only observe pics in April-May-June, then in September-October, and finally in December.

**Question 3.** What is the most and the second most common country of origin for reservations of each hotel?

**Answer:** 

- **Resort Hotel** : the first most common country of origin for reservations is Portugal (PRT) and the second one is Great Britain (GBR)
- **City Hotel** : the first most common country of origin for reservations is Portugal (PRT) and the second one is France (FRA)

See the plot below. 


```python
# Lets create a database following the same model than before but using countries instead of months
#First Hotel
n2_reserv_H1 = data.loc[(data["hotel"] == "Resort Hotel")].groupby("country")["hotel"].count()
data2_visualH1 = pd.DataFrame({"hotel": "Resort Hotel",
                                "country": list(n2_reserv_H1.index),
                                "n_booking": list(n2_reserv_H1.values)})
data2_visualH1.sort_values(by=['n_booking'], ascending=False)
#print(data2_visualH1.sort_values(by=['n_booking'], ascending=False).head(5).to_markdown())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hotel</th>
      <th>country</th>
      <th>n_booking</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>95</th>
      <td>Resort Hotel</td>
      <td>PRT</td>
      <td>17630</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Resort Hotel</td>
      <td>GBR</td>
      <td>6814</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Resort Hotel</td>
      <td>ESP</td>
      <td>3957</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Resort Hotel</td>
      <td>IRL</td>
      <td>2166</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Resort Hotel</td>
      <td>FRA</td>
      <td>1611</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>77</th>
      <td>Resort Hotel</td>
      <td>MKD</td>
      <td>1</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Resort Hotel</td>
      <td>PLW</td>
      <td>1</td>
    </tr>
    <tr>
      <th>90</th>
      <td>Resort Hotel</td>
      <td>PER</td>
      <td>1</td>
    </tr>
    <tr>
      <th>80</th>
      <td>Resort Hotel</td>
      <td>MUS</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Resort Hotel</td>
      <td>CYM</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>125 rows × 3 columns</p>
</div>




```python
#Second Hotel
n2_reserv_H2 = data.loc[(data["hotel"] == "City Hotel")].groupby("country")["hotel"].count()
data2_visualH2 = pd.DataFrame({"hotel": "City Hotel",
                                "country": list(n2_reserv_H2.index),
                                "n_booking": list(n2_reserv_H2.values)})
data2_visualH2.sort_values(by=['n_booking'], ascending=False)
data2_visualH2.sort_values(by=['n_booking'], ascending=False)
#print(data2_visualH2.sort_values(by=['n_booking'], ascending=False).head(5).to_markdown())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hotel</th>
      <th>country</th>
      <th>n_booking</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>125</th>
      <td>City Hotel</td>
      <td>PRT</td>
      <td>30960</td>
    </tr>
    <tr>
      <th>50</th>
      <td>City Hotel</td>
      <td>FRA</td>
      <td>8804</td>
    </tr>
    <tr>
      <th>39</th>
      <td>City Hotel</td>
      <td>DEU</td>
      <td>6084</td>
    </tr>
    <tr>
      <th>53</th>
      <td>City Hotel</td>
      <td>GBR</td>
      <td>5315</td>
    </tr>
    <tr>
      <th>46</th>
      <td>City Hotel</td>
      <td>ESP</td>
      <td>4611</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>107</th>
      <td>City Hotel</td>
      <td>MRT</td>
      <td>1</td>
    </tr>
    <tr>
      <th>133</th>
      <td>City Hotel</td>
      <td>SDN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>111</th>
      <td>City Hotel</td>
      <td>NAM</td>
      <td>1</td>
    </tr>
    <tr>
      <th>112</th>
      <td>City Hotel</td>
      <td>NCL</td>
      <td>1</td>
    </tr>
    <tr>
      <th>83</th>
      <td>City Hotel</td>
      <td>KIR</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>166 rows × 3 columns</p>
</div>




```python
data2_visual = pd.concat([data2_visualH1, data2_visualH2], ignore_index=True)
data2_visual = data2_visual.sort_values(by=['country'], ascending=True)

plt.figure(figsize=(6, 6))
sns.catplot(x = "country", y = "n_booking" , hue="hotel", hue_order = ["Resort Hotel", "City Hotel"],
            data=data2_visual[data2_visual["n_booking"]>(0.1*np.mean(data2_visual["n_booking"]))],
            legend_out=False, kind="bar", height = 4, aspect=3)
plt.title("Reservations per country (more than 10% of the mean of booking)")
plt.xticks(rotation=45)
plt.ylabel("Number of reservations")
plt.legend()
plt.show()
```


    <Figure size 432x432 with 0 Axes>



    
![png](TP2_KA_IB_files/TP2_KA_IB_36_1.png)
    


**Question 4.** Plot the number of cancelations for repeated and not repeated guests for both hotels.

**Answer** : See the 2 plots below the teacher's proposition


```python
# For both hotels at the same time (teacher's proposition)
plt.figure(figsize=(6, 6))
sns.countplot(x="is_canceled", hue='is_repeated_guest', data=data)
plt.title("Cancelations vs repeated guest", fontsize=16)
plt.plot()
```




    []




    
![png](TP2_KA_IB_files/TP2_KA_IB_39_1.png)
    


Most guests in these two hotels are not repeated, while the repreated guests are less likely to cancel.


```python
# The same only with City Hotel (H1)
plt.figure(figsize=(6, 6))
sns.countplot(x="is_canceled", hue='is_repeated_guest', data=data[(data['hotel'] == 'City Hotel')])
plt.title("Cancelations vs repeated guest in City Hotel (H1)", fontsize=16)
plt.plot()
```




    []




    
![png](TP2_KA_IB_files/TP2_KA_IB_41_1.png)
    



```python
# The same only with Resort Hotel (H2)
plt.figure(figsize=(6, 6))
sns.countplot(x="is_canceled", hue='is_repeated_guest', data=data[(data['hotel'] == 'Resort Hotel')])
plt.title("Cancelations vs repeated guest in Resort Hotel (H2)", fontsize=16)
plt.plot()
```




    []




    
![png](TP2_KA_IB_files/TP2_KA_IB_42_1.png)
    


End of question 4.


```python
data_req = data[(data['hotel'] == 'City Hotel')].groupby(['total_of_special_requests', 'is_canceled']).size().unstack(level=1)
data_req.plot(kind='bar', stacked=True, figsize=(6,6))
plt.title('Special Request vs Cancellation in H1 (City Hotel)')
plt.xlabel('Number of Special Request', fontsize=10)
plt.xticks(rotation=300)
plt.ylabel('Count', fontsize=10)
```




    Text(0, 0.5, 'Count')




    
![png](TP2_KA_IB_files/TP2_KA_IB_44_1.png)
    


Most of the reservations in the city hotel have no special requests and the cancelation in this case is almost 50/50. However, when special requests are made, the cancelation rate is significantly lower.



```python
# From raw value to percentage
total = [i+j for i,j in zip(data_req[0], data_req[1])]
data_req['percent_0'] = [i / j * 100 for i,j in zip(data_req[0], total)]
data_req['percent_1'] = [i / j * 100 for i,j in zip(data_req[1], total)]
data_req.iloc[:, 2:4]
data_req
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>is_canceled</th>
      <th>0</th>
      <th>1</th>
      <th>percent_0</th>
      <th>percent_1</th>
    </tr>
    <tr>
      <th>total_of_special_requests</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21617</td>
      <td>26340</td>
      <td>45.075797</td>
      <td>54.924203</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16699</td>
      <td>4721</td>
      <td>77.959851</td>
      <td>22.040149</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6403</td>
      <td>1739</td>
      <td>78.641611</td>
      <td>21.358389</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1307</td>
      <td>280</td>
      <td>82.356648</td>
      <td>17.643352</td>
    </tr>
    <tr>
      <th>4</th>
      <td>177</td>
      <td>21</td>
      <td>89.393939</td>
      <td>10.606061</td>
    </tr>
    <tr>
      <th>5</th>
      <td>25</td>
      <td>1</td>
      <td>96.153846</td>
      <td>3.846154</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_req.iloc[:, 2:4].plot(kind='bar', stacked=True, figsize=(6,6))
plt.title('Special Request vs Cancellation in H1 (City Hotel)')
plt.xlabel('Number of Special Request', fontsize=10)
plt.xticks(rotation=300)
plt.ylabel('Count', fontsize=10)
```




    Text(0, 0.5, 'Count')




    
![png](TP2_KA_IB_files/TP2_KA_IB_47_1.png)
    



**Question 5.** Make the same plot for Resort Hotel. Make your conclusions.

**Answer**

Most of the reservations in the Resort Hotel have no special request and the cancelation in this case is almost of 32 %. However, when special requests are made, the cancelation rate is lower (22 % when 1 special request is made, 23 % when 2 special requests are made, 18 % when 3 special request is made, 11 % when 4 special request is made, 7 % when 5 special request is made), but this decrease with the number of special requests is lower than with the City Hotel. 



```python
data_req2 = data[(data['hotel'] == 'Resort Hotel')].groupby(['total_of_special_requests',
                                                             'is_canceled']).size().unstack(level=1)
data_req2.plot(kind='bar', stacked=True, figsize=(6,6))
plt.title('Special Request vs Cancellation in H2 (Resort Hotel)')
plt.xlabel('Number of Special Request', fontsize=10)
plt.xticks(rotation=300)
plt.ylabel('Percentage', fontsize=10)
```




    Text(0, 0.5, 'Percentage')




    
![png](TP2_KA_IB_files/TP2_KA_IB_49_1.png)
    



```python
# From raw value to percentage
total = [i+j for i,j in zip(data_req2[0], data_req2[1])]
data_req2['percent_0'] = [i / j * 100 for i,j in zip(data_req2[0], total)]
data_req2['percent_1'] = [i / j * 100 for i,j in zip(data_req2[1], total)]
data_req2.iloc[:, 2:4]
#print(data_req2.to_markdown())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>is_canceled</th>
      <th>percent_0</th>
      <th>percent_1</th>
    </tr>
    <tr>
      <th>total_of_special_requests</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>67.729529</td>
      <td>32.270471</td>
    </tr>
    <tr>
      <th>1</th>
      <td>78.002710</td>
      <td>21.997290</td>
    </tr>
    <tr>
      <th>2</th>
      <td>76.652165</td>
      <td>23.347835</td>
    </tr>
    <tr>
      <th>3</th>
      <td>81.758242</td>
      <td>18.241758</td>
    </tr>
    <tr>
      <th>4</th>
      <td>89.436620</td>
      <td>10.563380</td>
    </tr>
    <tr>
      <th>5</th>
      <td>92.857143</td>
      <td>7.142857</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_req2.iloc[:, 2:4].plot(kind='bar', stacked=True, figsize=(6,6))
plt.title('Special Request vs Cancellation in H2 (Resort Hotel)')
plt.xlabel('Number of Special Request', fontsize=10)
plt.xticks(rotation=300)
plt.ylabel('Percentage', fontsize=10)
```




    Text(0, 0.5, 'Percentage')




    
![png](TP2_KA_IB_files/TP2_KA_IB_51_1.png)
    


# Predicting cancellations: Part II -- ML

**Our goals** The main message here is -- do not re-invent the wheel. The following few lines of code highlight the simplicity with which we can combine ```pandas``` dataframes and ```sklearn``` models. By learning few simple tools (i.e. ```pipeline```, ```gridsearchcv```) our code becomes readable, compact, and can be used to build extra features on top of it.


```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
```


```python
numeric_features = ["lead_time", "arrival_date_week_number", "total_of_special_requests"]
categorical_features = ["hotel", "market_segment","deposit_type","customer_type"]
features = numeric_features + categorical_features
X = data.drop(["is_canceled"], axis=1)[features]
y = data["is_canceled"]
```

Before using any ML algorithm from sklearn we need to handle missing values. There is no unique answer on how to deal with missing values in your dataset. We will use possibly the simplest approach. First of all if the feature is numerical and is misisng, we are going to replace it with zero. Secondly, if the feature is categorical and is missing, then we are going to define a new category and call it ```Not defined```.


```python
numeric_transformer = SimpleImputer(strategy="constant", fill_value=0) # to deal with missing numeric data
categorical_transformer = Pipeline(steps=[
                                    ("imputer", SimpleImputer(strategy="constant",
                                                              fill_value="Not defined")),
                                    ("onehot", OneHotEncoder(handle_unknown='ignore'))]) # to deal with missing categorical data 
preproc = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features),
                                          ("cat", categorical_transformer, categorical_features)])
```

**Question 1:** What is ```OneHotEncoder()```? Why do we use it in our case?

**Answer:**

One-Hot-Encoding is a way (among others such as label encoding) to convert categorical values into numerical values because most of the ML algorithms work better with numerical inputs.  In this strategy, each category value is converted into a new column and assigned a 1 or 0 (notation for true/false) value to the column. Let’s consider the previous example of the names of the hotels (first column) with one-hot encoding (2 last columns). See just below.

We use it in our case because **many variables are categorical** : hotel, arrival_date_month, customer_type, company, deposit_type, ...

*Source : see [here](https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd).*


```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder 
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
# passing hotel-types-cat column (label encoded values of bridge_types)
enc_df = pd.DataFrame(enc.fit_transform(data[['hotel']]).toarray())
# merge with main dataframe on key values
hotel_df = data.iloc[:, 0:1].join(enc_df)
hotel_df.head()
#print(hotel_df.head().to_markdown())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hotel</th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



Now imagine a situation when you want to try a lot of different models and for each model you want to make some cross-validation to select the best hyperparameters. On top of that you want to preprocess the data so that you feed something meaningfull into each method.

The next chunk of code shows how to do it.


```python
models = [("logreg", LogisticRegression(max_iter=500))]
grids = {"logreg" : {'logreg__C': np.logspace(-2, 2, 5, base=2)}}
for name, model in models:
    pipe = Pipeline(steps=[('preprocessor', preproc), (name, model)])
    clf = GridSearchCV(pipe, grids[name], cv=3)
    clf.fit(X, y)
    print('Results for {}'.format(name))
    print(clf.cv_results_)
```

    C:\Users\Kim Antunez\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\Kim Antunez\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-45-c19b5045d6f5> in <module>
          4     pipe = Pipeline(steps=[('preprocessor', preproc), (name, model)])
          5     clf = GridSearchCV(pipe, grids[name], cv=3)
    ----> 6     clf.fit(X, y)
          7     print('Results for {}'.format(name))
          8     print(clf.cv_results_)
    

    ~\Anaconda3\lib\site-packages\sklearn\utils\validation.py in inner_f(*args, **kwargs)
         70                           FutureWarning)
         71         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
    ---> 72         return f(**kwargs)
         73     return inner_f
         74 
    

    ~\Anaconda3\lib\site-packages\sklearn\model_selection\_search.py in fit(self, X, y, groups, **fit_params)
        734                 return results
        735 
    --> 736             self._run_search(evaluate_candidates)
        737 
        738         # For multi-metric evaluation, store the best_index_, best_params_ and
    

    ~\Anaconda3\lib\site-packages\sklearn\model_selection\_search.py in _run_search(self, evaluate_candidates)
       1186     def _run_search(self, evaluate_candidates):
       1187         """Search all candidates in param_grid"""
    -> 1188         evaluate_candidates(ParameterGrid(self.param_grid))
       1189 
       1190 
    

    ~\Anaconda3\lib\site-packages\sklearn\model_selection\_search.py in evaluate_candidates(candidate_params)
        713                                for parameters, (train, test)
        714                                in product(candidate_params,
    --> 715                                           cv.split(X, y, groups)))
        716 
        717                 if len(out) < 1:
    

    ~\Anaconda3\lib\site-packages\joblib\parallel.py in __call__(self, iterable)
       1049                 self._iterating = self._original_iterator is not None
       1050 
    -> 1051             while self.dispatch_one_batch(iterator):
       1052                 pass
       1053 
    

    ~\Anaconda3\lib\site-packages\joblib\parallel.py in dispatch_one_batch(self, iterator)
        864                 return False
        865             else:
    --> 866                 self._dispatch(tasks)
        867                 return True
        868 
    

    ~\Anaconda3\lib\site-packages\joblib\parallel.py in _dispatch(self, batch)
        782         with self._lock:
        783             job_idx = len(self._jobs)
    --> 784             job = self._backend.apply_async(batch, callback=cb)
        785             # A job can complete so quickly than its callback is
        786             # called before we get here, causing self._jobs to
    

    ~\Anaconda3\lib\site-packages\joblib\_parallel_backends.py in apply_async(self, func, callback)
        206     def apply_async(self, func, callback=None):
        207         """Schedule a func to be run"""
    --> 208         result = ImmediateResult(func)
        209         if callback:
        210             callback(result)
    

    ~\Anaconda3\lib\site-packages\joblib\_parallel_backends.py in __init__(self, batch)
        570         # Don't delay the application, to avoid keeping the input
        571         # arguments in memory
    --> 572         self.results = batch()
        573 
        574     def get(self):
    

    ~\Anaconda3\lib\site-packages\joblib\parallel.py in __call__(self)
        261         with parallel_backend(self._backend, n_jobs=self._n_jobs):
        262             return [func(*args, **kwargs)
    --> 263                     for func, args, kwargs in self.items]
        264 
        265     def __reduce__(self):
    

    ~\Anaconda3\lib\site-packages\joblib\parallel.py in <listcomp>(.0)
        261         with parallel_backend(self._backend, n_jobs=self._n_jobs):
        262             return [func(*args, **kwargs)
    --> 263                     for func, args, kwargs in self.items]
        264 
        265     def __reduce__(self):
    

    ~\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py in _fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters, fit_params, return_train_score, return_parameters, return_n_test_samples, return_times, return_estimator, error_score)
        529             estimator.fit(X_train, **fit_params)
        530         else:
    --> 531             estimator.fit(X_train, y_train, **fit_params)
        532 
        533     except Exception as e:
    

    ~\Anaconda3\lib\site-packages\sklearn\pipeline.py in fit(self, X, y, **fit_params)
        333             if self._final_estimator != 'passthrough':
        334                 fit_params_last_step = fit_params_steps[self.steps[-1][0]]
    --> 335                 self._final_estimator.fit(Xt, y, **fit_params_last_step)
        336 
        337         return self
    

    ~\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py in fit(self, X, y, sample_weight)
       1415                       penalty=penalty, max_squared_sum=max_squared_sum,
       1416                       sample_weight=sample_weight)
    -> 1417             for class_, warm_start_coef_ in zip(classes_, warm_start_coef))
       1418 
       1419         fold_coefs_, _, n_iter_ = zip(*fold_coefs_)
    

    ~\Anaconda3\lib\site-packages\joblib\parallel.py in __call__(self, iterable)
       1046             # remaining jobs.
       1047             self._iterating = False
    -> 1048             if self.dispatch_one_batch(iterator):
       1049                 self._iterating = self._original_iterator is not None
       1050 
    

    ~\Anaconda3\lib\site-packages\joblib\parallel.py in dispatch_one_batch(self, iterator)
        864                 return False
        865             else:
    --> 866                 self._dispatch(tasks)
        867                 return True
        868 
    

    ~\Anaconda3\lib\site-packages\joblib\parallel.py in _dispatch(self, batch)
        782         with self._lock:
        783             job_idx = len(self._jobs)
    --> 784             job = self._backend.apply_async(batch, callback=cb)
        785             # A job can complete so quickly than its callback is
        786             # called before we get here, causing self._jobs to
    

    ~\Anaconda3\lib\site-packages\joblib\_parallel_backends.py in apply_async(self, func, callback)
        206     def apply_async(self, func, callback=None):
        207         """Schedule a func to be run"""
    --> 208         result = ImmediateResult(func)
        209         if callback:
        210             callback(result)
    

    ~\Anaconda3\lib\site-packages\joblib\_parallel_backends.py in __init__(self, batch)
        570         # Don't delay the application, to avoid keeping the input
        571         # arguments in memory
    --> 572         self.results = batch()
        573 
        574     def get(self):
    

    ~\Anaconda3\lib\site-packages\joblib\parallel.py in __call__(self)
        261         with parallel_backend(self._backend, n_jobs=self._n_jobs):
        262             return [func(*args, **kwargs)
    --> 263                     for func, args, kwargs in self.items]
        264 
        265     def __reduce__(self):
    

    ~\Anaconda3\lib\site-packages\joblib\parallel.py in <listcomp>(.0)
        261         with parallel_backend(self._backend, n_jobs=self._n_jobs):
        262             return [func(*args, **kwargs)
    --> 263                     for func, args, kwargs in self.items]
        264 
        265     def __reduce__(self):
    

    ~\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py in _logistic_regression_path(X, y, pos_class, Cs, fit_intercept, max_iter, tol, verbose, solver, coef, class_weight, dual, penalty, intercept_scaling, multi_class, random_state, check_input, max_squared_sum, sample_weight, l1_ratio)
        758                 func, w0, method="L-BFGS-B", jac=True,
        759                 args=(X, target, 1. / C, sample_weight),
    --> 760                 options={"iprint": iprint, "gtol": tol, "maxiter": max_iter}
        761             )
        762             n_iter_i = _check_optimize_result(
    

    ~\Anaconda3\lib\site-packages\scipy\optimize\_minimize.py in minimize(fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options)
        616     elif meth == 'l-bfgs-b':
        617         return _minimize_lbfgsb(fun, x0, args, jac, bounds,
    --> 618                                 callback=callback, **options)
        619     elif meth == 'tnc':
        620         return _minimize_tnc(fun, x0, args, jac, bounds, callback=callback,
    

    ~\Anaconda3\lib\site-packages\scipy\optimize\lbfgsb.py in _minimize_lbfgsb(fun, x0, args, jac, bounds, disp, maxcor, ftol, gtol, eps, maxfun, maxiter, iprint, callback, maxls, finite_diff_rel_step, **unknown_options)
        358             # until the completion of the current minimization iteration.
        359             # Overwrite f and g:
    --> 360             f, g = func_and_grad(x)
        361         elif task_str.startswith(b'NEW_X'):
        362             # new iteration
    

    ~\Anaconda3\lib\site-packages\scipy\optimize\_differentiable_functions.py in fun_and_grad(self, x)
        198         if not np.array_equal(x, self.x):
        199             self._update_x_impl(x)
    --> 200         self._update_fun()
        201         self._update_grad()
        202         return self.f, self.g
    

    ~\Anaconda3\lib\site-packages\scipy\optimize\_differentiable_functions.py in _update_fun(self)
        164     def _update_fun(self):
        165         if not self.f_updated:
    --> 166             self._update_fun_impl()
        167             self.f_updated = True
        168 
    

    ~\Anaconda3\lib\site-packages\scipy\optimize\_differentiable_functions.py in update_fun()
         71 
         72         def update_fun():
    ---> 73             self.f = fun_wrapped(self.x)
         74 
         75         self._update_fun_impl = update_fun
    

    ~\Anaconda3\lib\site-packages\scipy\optimize\_differentiable_functions.py in fun_wrapped(x)
         68         def fun_wrapped(x):
         69             self.nfev += 1
    ---> 70             return fun(x, *args)
         71 
         72         def update_fun():
    

    ~\Anaconda3\lib\site-packages\scipy\optimize\optimize.py in __call__(self, x, *args)
         72     def __call__(self, x, *args):
         73         """ returns the the function value """
    ---> 74         self._compute_if_needed(x, *args)
         75         return self._value
         76 
    

    ~\Anaconda3\lib\site-packages\scipy\optimize\optimize.py in _compute_if_needed(self, x, *args)
         66         if not np.all(x == self.x) or self._value is None or self.jac is None:
         67             self.x = np.asarray(x).copy()
    ---> 68             fg = self.fun(x, *args)
         69             self.jac = fg[1]
         70             self._value = fg[0]
    

    ~\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py in _logistic_loss_and_grad(w, X, y, alpha, sample_weight)
        120 
        121     # Logistic loss is the negative of the log of the logistic function.
    --> 122     out = -np.sum(sample_weight * log_logistic(yz)) + .5 * alpha * np.dot(w, w)
        123 
        124     z = expit(yz)
    

    <__array_function__ internals> in dot(*args, **kwargs)
    

    KeyboardInterrupt: 


**Question 2:** In the previous example we again encounter the convergence problem. Of course we can set higher number of iterations, but it is time consuming. As you have seen, proper normalization can resolve the issue. Insert a normalization step in the pipeline. Note that we do not want to normalize the categorical data, it simply does not make sense. Be careful to normalize only the numerical data. Did it resolve the warning?




**Answer :**
To normalize the numeric features of the data, we modify the `numeric_transformer`.
    
    #numeric_transformer = SimpleImputer(strategy="constant", fill_value=0) # to deal with missing numeric data
    
We change it into the following pipeline : 

    from sklearn.preprocessing import StandardScaler
    numeric_transformer = Pipeline(steps=[
                                        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                                        ("scaler", StandardScaler())]) #NEW : rescale the data !

And it solved the warning!



```python
from sklearn.preprocessing import StandardScaler
#numeric_transformer = SimpleImputer(strategy="constant", fill_value=0) # to deal with missing numeric data
numeric_transformer = Pipeline(steps=[
                                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                                    ("scaler", StandardScaler())]) #NEW : rescale the data !
preproc = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features),
                                          ("cat", categorical_transformer, categorical_features)])
```


```python
models = [("logreg", LogisticRegression(max_iter=500))]
grids = {"logreg" : {'logreg__C': np.logspace(-2, 2, 5, base=2)}}
for name, model in models:
    pipe = Pipeline(steps=[('preprocessor', preproc), (name, model)])
    clf = GridSearchCV(pipe, grids[name], cv=3)
    clf.fit(X, y)
    print('Results for {}'.format(name))
    print(clf.cv_results_)
```

    Results for logreg
    {'mean_fit_time': array([1.07707651, 1.06985847, 1.26302139, 0.94357912, 1.00347384]), 'std_fit_time': array([0.22217166, 0.14027325, 0.44600286, 0.12203947, 0.05911086]), 'mean_score_time': array([0.0600067 , 0.0611678 , 0.05684272, 0.0552206 , 0.06061093]), 'std_score_time': array([0.00130701, 0.0016936 , 0.00354926, 0.00281719, 0.00184487]), 'param_logreg__C': masked_array(data=[0.25, 0.5, 1.0, 2.0, 4.0],
                 mask=[False, False, False, False, False],
           fill_value='?',
                dtype=object), 'params': [{'logreg__C': 0.25}, {'logreg__C': 0.5}, {'logreg__C': 1.0}, {'logreg__C': 2.0}, {'logreg__C': 4.0}], 'split0_test_score': array([0.70128402, 0.7000779 , 0.68552906, 0.67786517, 0.66600498]), 'split1_test_score': array([0.78262181, 0.78244591, 0.78232028, 0.78221977, 0.78219464]), 'split2_test_score': array([0.73585285, 0.73582772, 0.73582772, 0.736079  , 0.736079  ]), 'mean_test_score': array([0.73991956, 0.73945051, 0.73455902, 0.73205464, 0.72809287]), 'std_test_score': array([0.03333029, 0.03372404, 0.03952503, 0.04269752, 0.04776919]), 'rank_test_score': array([1, 2, 3, 4, 5])}
    


**Question 3:** As we can see, previous code uses only logistic regression. Modify the above code inserting your favorite ML method.


**Answer** : 

Below, you can see the same actions but using the **SVM** method instead of logistic regressions. The main differences are: 

- This time we use the following scaler : `MaxAbsScaler()`
- We use `LinearSVC()` instead of `LogisticRegression()`
- We test the parameter `svc__C` instead of `logreg__C`

**=> The results of SVM seem to be a bit better than the ones for the logistic regression : the mean test scores are globally higher.**


```python
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
numeric_transformer = Pipeline(steps=[
                                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                                    ("scaler", MinMaxScaler())]) #NEW : MinMaxScaler
preproc = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features),
                                          ("cat", categorical_transformer, categorical_features)])
models = [("logreg", LogisticRegression(max_iter=500)),
          ("svc", LinearSVC(max_iter=700))]
grids = {"logreg" : {'logreg__C': np.logspace(-2, 2, 5, base=2)},
         "svc" : {'svc__C': np.logspace(-2, 2, 5, base=2)}}
for name, model in models:
    pipe = Pipeline(steps=[('preprocessor', preproc), (name, model)])
    clf = GridSearchCV(pipe, grids[name], cv=3)
    clf.fit(X, y)
    print('Results for {}'.format(name))
    #print(clf.cv_results_)
    display(pd.DataFrame(clf.cv_results_)[['params','mean_test_score', 'std_test_score',
                                           'rank_test_score']])
    #print(pd.DataFrame(clf.cv_results_)[['params','mean_test_score', 'std_test_score',
    #                                       'rank_test_score']].to_markdown())
```

    Results for logreg
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>params</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'logreg__C': 0.25}</td>
      <td>0.739920</td>
      <td>0.033077</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{'logreg__C': 0.5}</td>
      <td>0.739283</td>
      <td>0.033688</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{'logreg__C': 1.0}</td>
      <td>0.734693</td>
      <td>0.039152</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{'logreg__C': 2.0}</td>
      <td>0.731133</td>
      <td>0.043546</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{'logreg__C': 4.0}</td>
      <td>0.727850</td>
      <td>0.047672</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>


    C:\Users\Kim Antunez\Anaconda3\lib\site-packages\sklearn\svm\_base.py:977: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    

    Results for svc
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>params</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'svc__C': 0.25}</td>
      <td>0.740933</td>
      <td>0.036266</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{'svc__C': 0.5}</td>
      <td>0.738647</td>
      <td>0.039248</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{'svc__C': 1.0}</td>
      <td>0.737424</td>
      <td>0.040830</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{'svc__C': 2.0}</td>
      <td>0.735489</td>
      <td>0.043405</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{'svc__C': 4.0}</td>
      <td>0.734057</td>
      <td>0.045323</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>


**Remark:** Note that in this part we picked only a small subset of features. We could have used other features as well.


# The homework: Part III

**The story!**
You are a data scientist working for the hotel, one day the manager comes and tells you.

In our hotels we have an option to offer a parking spot, which costs money. Apparently, not every customer is aware that we have such an option. I really want to offer parking spots for everyone who needs it, because the hotel can earn more money like that. Look, I can send an SMS notification to our customers. Of course SMS are not free, and, moreover, people get very much angry if they receive stupid notifications for no reason. For each new reservation, I would like to decide if I should or shouldn't send the notification to the customer.

**Problem** Explore your data to help the manager and construct a prediction algorithm, using the above template as an inspiration. 

**Warning!** Be aware, that some columns are not actually avaiable at the moment of reservation. For instance, the target column ```is_canceled``` from the previous part clearly cannot be observed at the moment when we need to decide to send the SMS.


```python
#import packages for part III
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
```

## FIRST STEP : choose the relevant features and build the dataset

### Remove features that cannot be observed

We first **delete** from the dataset **4 features** that cannot be observed at the moment when the SMS is sent : 
`is_canceled`, `assigned_room_type`, `reservation_status` and `reservation_status_date`. See below for more details about these variables. 

-**hotel** : Hotel (H1 = Resort Hotel or H2 = City Hotel)
 
-<s>**is_canceled** : Value indicating if the booking was canceled (1) or not (0)</s>

-**lead_time** : Number of days that elapsed between the entering date of the booking into the PMS and the arrival date

-**arrival_date_year** : Year of arrival date

-**arrival_date_month** : Month of arrival date

-**arrival_date_week_number** : Week number of year for arrival date

-**arrival_date_day_of_month** : Day of arrival date

-**stays_in_weekend_nights** : Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel

-**stays_in_week_nights** : Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel

-**adults** : Number of adults

-**children** : Number of children

-**babies** : Number of babies

-**meal** : Type of meal booked. Categories are presented in standard hospitality meal packages: Undefined/SC – no meal package; BB – Bed & Breakfast; HB – Half board (breakfast and one other meal – usually dinner); FB – Full board (breakfast, lunch and dinner)

-**country** : Country of origin. Categories are represented in the ISO 3155–3:2013 format

-**market_segment** : Market segment designation. In categories, the term “TA” means “Travel Agents” and “TO” means “Tour Operators”

-**distribution_channel** : Booking distribution channel. The term “TA” means “Travel Agents” and “TO” means “Tour Operators”

-**is_repeated_guest** : Value indicating if the booking name was from a repeated guest (1) or not (0)

-**previous_cancellations** : Number of previous bookings that were cancelled by the customer prior to the current booking

-**previous_bookings_not_canceled** : Number of previous bookings not cancelled by the customer prior to the current booking

-**reserved_room_type** : Code of room type reserved. Code is presented instead of designation for anonymity reasons.

<s>-**assigned_room_type** : Code for the type of room assigned to the booking. Sometimes the assigned room type differs from the reserved room type due to hotel operation reasons (e.g. overbooking) or by customer request. Code is presented instead of designation for anonymity reasons.</s>

-**booking_changes** : Number of changes/amendments made to the booking from the moment the booking was entered on the PMS 

-**deposit_type** : Indication on if the customer made a deposit to guarantee the booking. This variable can assume three categories: No Deposit – no deposit was made; Non Refund – a deposit was made in the value of the total stay cost; Refundable – a deposit was made with a value under the total cost of stay.

-**agent** : ID of the travel agency that made the booking

-**company** : ID of the company/entity that made the booking or responsible for paying the booking. ID is presented instead of designation for anonymity reasons

-**days_in_waiting_list** : Number of days the booking was in the waiting list before it was confirmed to the customer

-**customer_type** : Type of booking, assuming one of four categories: Contract - when the booking has an allotment or other type of contract associated to it; Group – when the booking is associated to a group; Transient – when the booking is not part of a group or contract, and is not associated to other transient booking; Transient-party – when the booking is transient, but is associated to at least other transient booking

-**adr** : Average Daily Rate as defined by dividing the sum of all lodging transactions by the total number of staying nights

-**required_car_parking_spaces** : Number of car parking spaces required by the customer

-**total_of_special_requests** :  Number of special requests made by the customer (e.g. twin bed or high floor)

<s>-**reservation_status** : Reservation last status, assuming one of three categories: Canceled – booking was canceled by the customer; Check-Out – customer has checked in but already departed; No-Show – customer did not check-in and did inform the hotel of the reason why</s>

<s>-**reservation_status_date** : Date at which the last status was set. This variable can be used in conjunction with the ReservationStatus</s>

### Remove features with too many missing data

Then, we look at the features containing lots of missing data. The feature "company" displays a lot of missing data. Therefore we don't use this feature in our predicting models. 


```python
#Check Missing Data
plt.figure(figsize=(10,4))
sns.heatmap(data.isna(),cbar=False)
```




    <AxesSubplot:>




    
![png](TP2_KA_IB_files/TP2_KA_IB_76_1.png)
    


### Creation of dataset

Finally, we create the data.frame we need by creating a binary variable called `bool_parking` whose value is 1 if `required_car_parking_spaces` > 0


```python
data2 = data.copy()

# New variable : boolean of value 1 if required_car_parking_spaces > 0
data2["bool_parking"] = data2["required_car_parking_spaces"] > 0 
data2["bool_parking"] = data2["bool_parking"].astype(int)

#We modify agent which is a strange variable (mix of numeric and categorical) by adding the prefix "A"
data2["agent"] = data2["agent"].map(lambda x:x if np.isnan(x) else ("A" + str(int(x))))

#We distinguish numerical and categorical features
numeric_features = ["lead_time", "arrival_date_year",
                    "adr",
                    "arrival_date_week_number",
                    "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights", 
                    "adults", "children",
                    "babies", "previous_cancellations", 
                    "days_in_waiting_list", "previous_bookings_not_canceled", "booking_changes", 
                    "total_of_special_requests"]
categorical_features = ["meal", "country", "market_segment", "distribution_channel",
                        "hotel", "reserved_room_type", "deposit_type","agent", 
                       "customer_type","is_repeated_guest","arrival_date_month"]                        
features = numeric_features + categorical_features
```

## SECOND STEP : Exploratory Data analysis 

Now, we do some plots to see what kind of variables seem to be important to predict the outcome `bool_parking`. 

These plots make us draw some first conclusions about the interesting predictors of the outcome : 

People more susceptible to use a parking seem to be people : 

- coming in groups and transient travellers : guests who are predominantly on-the-move and seek short hotel-stays (`customer_type = Group` or `Transient`)
- going to the Resort Hotel (`hotel = Resort Hotel`) => It may mean that we should do 2 different predictive models, one for each hotel
- who book directly at the hotel (`market_segment = direct`)
- who book a certain kind of room (`reserved_room_type = H`) but for anonymity reasons we don't know what it represents
- coming in the summer (`arrival_date_week_number` around `25`)
- who don't book in advance (`lead_time` close to `0`)
- rich (`adr` is `high`)
- who have children (children is `high`)


```python
#------ For executing the code Javascript IPython in JupyterLab -----------
%matplotlib inline
#--------------------------------------------------------
############# categorical variables
variables = categorical_features.copy()
variables.append("bool_parking")
variables.pop(7) #variable agent not very intersting to plot (to many of them)
variables.pop(1) #variable country not very intersting to plot (to many of them)
df = data2.filter(variables)
df = pd.melt(df, df.columns[-1], df.columns[:-1])
#df.head()
df = df.groupby(['variable','value'])['bool_parking'].value_counts(normalize=True)
df = df.mul(100)
df = df.rename('percent').reset_index()

for vExpl in list(variables):
    if vExpl != 'bool_parking':
        data_graph = df[(df['variable'] == vExpl)]
        g = sns.catplot(x="value", y="percent", hue="bool_parking", col="variable", data=data_graph,
               col_wrap=3, kind="bar", sharex=False, sharey=True,legend_out=False, height=4,aspect=1.6)
```


    
![png](TP2_KA_IB_files/TP2_KA_IB_80_0.png)
    



    
![png](TP2_KA_IB_files/TP2_KA_IB_80_1.png)
    



    
![png](TP2_KA_IB_files/TP2_KA_IB_80_2.png)
    



    
![png](TP2_KA_IB_files/TP2_KA_IB_80_3.png)
    



    
![png](TP2_KA_IB_files/TP2_KA_IB_80_4.png)
    



    
![png](TP2_KA_IB_files/TP2_KA_IB_80_5.png)
    



    
![png](TP2_KA_IB_files/TP2_KA_IB_80_6.png)
    



    
![png](TP2_KA_IB_files/TP2_KA_IB_80_7.png)
    



    
![png](TP2_KA_IB_files/TP2_KA_IB_80_8.png)
    



```python
#---------------------------------------------------------------------------------------------
#Agent : Percentage of reservations with parking for agent with more than 50% mean of bookings
#        (exclude percentage with few reservations)
#---------------------------------------------------------------------------------------------
n_reserv_byagent_H1 = data2.loc[
    (data["hotel"] == "Resort Hotel")].groupby("agent")["hotel"].count()
n_bparking_byagent_H1 = data2.loc[
    (data2["hotel"] == "Resort Hotel")].groupby("agent")["bool_parking"].sum()
data_agent_visualH1 = pd.DataFrame({"hotel": "Resort Hotel",
                                "agent": list(n_reserv_byagent_H1.index),
                                "n_booking": list(n_reserv_byagent_H1.values),
                                "n_bool_parking": list(n_bparking_byagent_H1.values)})
data_agent_visualH1.sort_values(by=['n_bool_parking'], ascending=False)
n_reserv_byagent_H2 = data2.loc[
    (data["hotel"] == "City Hotel")].groupby("agent")["hotel"].count()
n_bparking_byagent_H2 = data2.loc[
    (data2["hotel"] == "City Hotel")].groupby("agent")["bool_parking"].sum()
data_agent_visualH2 = pd.DataFrame({"hotel": "City Hotel",
                                "agent": list(n_reserv_byagent_H2.index),
                                "n_booking": list(n_reserv_byagent_H2.values),
                                "n_bool_parking": list(n_bparking_byagent_H2.values)})
data_agent_visualH2.sort_values(by=['n_bool_parking'], ascending=False)
data_agent_visual = pd.concat([data_agent_visualH1, data_agent_visualH2], ignore_index=True)
data_agent_visual["percent_bParking"] = data_agent_visual[
    "n_bool_parking"] / data_agent_visual["n_booking"] * 100 # percent of parking
data_agent_visual = data_agent_visual.sort_values(by=['percent_bParking'], ascending=False)
plt.figure(figsize=(15, 5))
sns.barplot(x = "agent", y = "percent_bParking" , hue="hotel",
            hue_order = ["Resort Hotel", "City Hotel"], 
            data=data_agent_visual[
                data_agent_visual["n_bool_parking"]>0.5*np.mean(data_agent_visual["n_bool_parking"])])
plt.title("Reservations parking per agent (more than 50% of the mean of booking)")
plt.xticks(rotation=45)
plt.ylabel("Percentage of reservations with parking")
plt.legend()
plt.show()
```


    
![png](TP2_KA_IB_files/TP2_KA_IB_81_0.png)
    



```python
############# numeric variables
sns.distributions._has_statsmodels = False
variables = numeric_features.copy()
variables.append("bool_parking")
df = data2.filter(variables)
df = pd.melt(df, df.columns[-1], df.columns[:-1])
g = sns.FacetGrid(df, col="variable", hue="bool_parking",
                  col_wrap=3,sharex=False, sharey=False,legend_out=False)
g.map(sns.kdeplot, "value") #, shade=True
g.add_legend()
```




    <seaborn.axisgrid.FacetGrid at 0x29ecd337348>




    
![png](TP2_KA_IB_files/TP2_KA_IB_82_1.png)
    


## THIRD STEP : Modelisation

Now that the data-processing is done, we apply some ML models to predict the variable `bool_parking`.

###  Models including all the variables

#### Preparing the test and training sets


```python
#--------------------------------------------------------------------------------
# Feature Engineering : Process of converting raw data into a structured format 
# Making the data ready to use for model training (transformers)
# Creating the test and training sets
#--------------------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
                                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                                    ("scaler", MinMaxScaler())]) 
categorical_transformer = Pipeline(steps=[
                                    ("imputer", SimpleImputer(strategy="constant",
                                                              fill_value="Not defined")),
                                    ("onehot", OneHotEncoder(handle_unknown='ignore'))]) #
preproc = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features),
                                          ("cat", categorical_transformer, categorical_features)])

X = data2.drop(["is_canceled","assigned_room_type","reservation_status",
                "reservation_status_date", "company"],
              axis=1)[features]
y = data2["bool_parking"]

X, y = shuffle(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y)

X_preproc = preproc.fit_transform(X)
X_train_preproc, X_test_preproc, y_train_preproc, y_test_preproc = train_test_split(X_preproc, y)

```

#### Running the models using Cross-validation


```python
models = [("logreg", LogisticRegression(max_iter=700)),
          ("svc", LinearSVC(max_iter=800))]
grids = {"logreg" : {'logreg__C': np.logspace(-2, 2, 5, base=2)},
         "svc" : {'svc__C': np.logspace(-2, 2, 5, base=2)}}
X_train, X_test, y_train, y_test = train_test_split(X, y)

for name, model in models:
    pipe = Pipeline(steps=[('preprocessor', preproc), (name, model)])
    clf = GridSearchCV(pipe, grids[name], cv=3)
    clf.fit(X_train, y_train)
    print('Results for {}'.format(name))
    print('Returned hyperparameter: {}'.format(clf.best_params_))
    print('Best classification accuracy in train is: {}'.format(clf.best_score_))
    print('Classification accuracy on test is: {}'.format(clf.score(X_test, y_test)))    
    display(pd.DataFrame(clf.cv_results_)[['params','mean_test_score', 'std_test_score',
                                           'rank_test_score']])
    #print(pd.DataFrame(clf.cv_results_)[['params','mean_test_score', 'std_test_score',
    #                                       'rank_test_score']].to_markdown())
```

    Results for logreg
    Returned hyperparameter: {'logreg__C': 0.25}
    Best classification accuracy in train is: 0.9377833861775132
    Classification accuracy on test is: 0.9381533101045296
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>params</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'logreg__C': 0.25}</td>
      <td>0.937783</td>
      <td>0.000028</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{'logreg__C': 0.5}</td>
      <td>0.937772</td>
      <td>0.000016</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{'logreg__C': 1.0}</td>
      <td>0.937783</td>
      <td>0.000028</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{'logreg__C': 2.0}</td>
      <td>0.937772</td>
      <td>0.000043</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{'logreg__C': 4.0}</td>
      <td>0.937772</td>
      <td>0.000043</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


    Results for svc
    Returned hyperparameter: {'svc__C': 0.25}
    Best classification accuracy in train is: 0.9377945546199372
    Classification accuracy on test is: 0.9381533101045296
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>params</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'svc__C': 0.25}</td>
      <td>0.937795</td>
      <td>0.000015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{'svc__C': 0.5}</td>
      <td>0.937795</td>
      <td>0.000015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{'svc__C': 1.0}</td>
      <td>0.937795</td>
      <td>0.000015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{'svc__C': 2.0}</td>
      <td>0.937795</td>
      <td>0.000015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{'svc__C': 4.0}</td>
      <td>0.937795</td>
      <td>0.000015</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


#### Running the models with the best parameters and evaluating with different scores


```python
logreg = LogisticRegression(max_iter=700, C=0.25) #best parameter logreg : C=0.25
logreg.fit(X_train_preproc,y_train_preproc)
y_pred = logreg.predict(X_test_preproc)
print("MSE logreg: ", mean_squared_error(y_test_preproc, y_pred, squared=False))
print("R2 logreg: ", r2_score(y_test, y_pred))
print("accuracy logreg : ", accuracy_score(y_test_preproc, y_pred)) #OK because Y_pred is binary

print("\n")

svc = LinearSVC(max_iter=800, C=0.25) #best parameter svc : C=0.25
svc.fit(X_train_preproc,y_train_preproc)
y_pred = svc.predict(X_test_preproc)
print("MSE svc: ", mean_squared_error(y_test_preproc, y_pred, squared=False))
print("R2 svc: ", r2_score(y_test_preproc, y_pred))
print("accuracy svc : ", accuracy_score(y_test_preproc, y_pred)) #OK because Y_pred is binary
```

    MSE logreg:  0.24720360451504717
    R2 logreg:  -0.06592386258124439
    accuracy logreg :  0.9388903779147681
    
    
    MSE svc:  0.24720360451504717
    R2 svc:  -0.06508706822723398
    accuracy svc :  0.9388903779147681
    

**=> Both SVM and LogisticRegression WITHOUT selection of features present really performent results. The accuracy is nearly 94%. Indeed, the number of individuals is +100 000 (n) and the number of features (p) is around 30. Therefore, we have n >> p which prevents from overfitting**

###  Models with best predictive features chosen MANUALLY

Now we try to pick up **MANUALLY** features with high predicting powers to see if it can reduce over-fitting and show better results. We filter columns with the 8 pre-selected variables (4 numeric features : "arrival_date_week_number","lead_time", "adr","children" and 4 categorical features "customer_type","hotel", "market_segment", "reserved_room_type")


#### Preparing the test and training sets


```python
dataHW = data2.copy()

#--------------------------------------------------------------------------------
# Feature Selection : Picking up the most predictive features 
#--------------------------------------------------------------------------------
numeric_features = ["arrival_date_week_number","lead_time", "adr","children"]
categorical_features = ["customer_type","hotel", "market_segment", "reserved_room_type"]                       
features = numeric_features + categorical_features

#--------------------------------------------------------------------------------
# Feature Engineering : Process of converting raw data into a structured format 
# Making the data ready to use for model training (transformers)
# Creating the test and training sets
#--------------------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
                                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                                    ("scaler", MinMaxScaler())]) 
categorical_transformer = Pipeline(steps=[
                                    ("imputer", SimpleImputer(strategy="constant",
                                                              fill_value="Not defined")),
                                    ("onehot", OneHotEncoder(handle_unknown='ignore'))])
preproc = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features),
                                          ("cat", categorical_transformer, categorical_features)])

X = dataHW[features]
y = dataHW["bool_parking"]

X, y = shuffle(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y)

X_preproc = preproc.fit_transform(X)
X_train_preproc, X_test_preproc, y_train_preproc, y_test_preproc = train_test_split(X_preproc, y)

```

#### Running the models using Cross-validation


```python
models = [("logreg", LogisticRegression(max_iter=700)),
          ("svc", LinearSVC(max_iter=800))]
grids = {"logreg" : {'logreg__C': np.logspace(-2, 2, 5, base=2)},
         "svc" : {'svc__C': np.logspace(-2, 2, 5, base=2)}}
for name, model in models:
    pipe = Pipeline(steps=[('preprocessor', preproc), (name, model)])
    clf = GridSearchCV(pipe, grids[name], cv=3)
    clf.fit(X_train, y_train)
    print('Results for {}'.format(name))
    print('Returned hyperparameter: {}'.format(clf.best_params_))
    print('Best classification accuracy in train is: {}'.format(clf.best_score_))
    print('Classification accuracy on test is: {}'.format(clf.score(X_test, y_test)))    
    display(pd.DataFrame(clf.cv_results_)[['params','mean_test_score', 'std_test_score',
                                           'rank_test_score']])
    #print(pd.DataFrame(clf.cv_results_)[['params','mean_test_score', 'std_test_score',
    #                                       'rank_test_score']].to_markdown())
```

    Results for logreg
    Returned hyperparameter: {'logreg__C': 0.25}
    Best classification accuracy in train is: 0.9377833861775132
    Classification accuracy on test is: 0.9381533101045296
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>params</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'logreg__C': 0.25}</td>
      <td>0.937783</td>
      <td>0.000028</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{'logreg__C': 0.5}</td>
      <td>0.937772</td>
      <td>0.000016</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{'logreg__C': 1.0}</td>
      <td>0.937783</td>
      <td>0.000028</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{'logreg__C': 2.0}</td>
      <td>0.937772</td>
      <td>0.000043</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{'logreg__C': 4.0}</td>
      <td>0.937772</td>
      <td>0.000043</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


    Results for svc
    Returned hyperparameter: {'svc__C': 0.25}
    Best classification accuracy in train is: 0.9377945546199372
    Classification accuracy on test is: 0.9381533101045296
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>params</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'svc__C': 0.25}</td>
      <td>0.937795</td>
      <td>0.000015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{'svc__C': 0.5}</td>
      <td>0.937795</td>
      <td>0.000015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{'svc__C': 1.0}</td>
      <td>0.937795</td>
      <td>0.000015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{'svc__C': 2.0}</td>
      <td>0.937795</td>
      <td>0.000015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{'svc__C': 4.0}</td>
      <td>0.937795</td>
      <td>0.000015</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


#### Running the models with the best parameters and evaluating with different scores


```python
logreg = LogisticRegression(max_iter=700, C=0.25) #best parameter logreg : C=0.25
logreg.fit(X_train_preproc,y_train_preproc)
y_pred = logreg.predict(X_test_preproc)
print("MSE logreg: ", mean_squared_error(y_test_preproc, y_pred, squared=False))
print("R2 logreg: ", r2_score(y_test, y_pred))
print("accuracy logreg : ", accuracy_score(y_test_preproc, y_pred)) #OK because Y_pred is binary

print("\n")

svc = LinearSVC(max_iter=800, C=0.25) #best parameter svc : C=0.25
svc.fit(X_train_preproc,y_train_preproc)
y_pred = svc.predict(X_test_preproc)
print("MSE svc: ", mean_squared_error(y_test_preproc, y_pred, squared=False))
print("R2 svc: ", r2_score(y_test_preproc, y_pred))
print("accuracy svc : ", accuracy_score(y_test_preproc, y_pred)) #OK because Y_pred is binary
```

    MSE logreg:  0.246864551270153
    R2 logreg:  -0.06660949113779324
    accuracy logreg :  0.939057893326186
    
    
    MSE svc:  0.246864551270153
    R2 svc:  -0.06489707089086316
    accuracy svc :  0.939057893326186
    

**=> Both SVM and LogisticRegression WITH MANUAL selection of features still present really good results. Even if the different scores are a bit less promising than the ones of the previous method, they are still very good. For example, the accuracy is here still nearly 94%.**

###  Models with best predictive features chosen with REGULARIZED REGRESSIONS


We try to improve the models using a polynomial regression with a regularization called LASSO. This method only selects the features that best predict the outcome. Indeed, as we explained earlier, subsetting the features may prevent from overfitting. LASSO regression is such as:

$$\hat f = \underset{f \in \mathcal F_n^\text{poly}}{argmin} \{ \frac{1}{2m} \sum_{i=1}^m (Y_i - f(X_i))^2 + \alpha \sum_{k=1}^n | a_k | \} $$

where $\alpha$ > 0 is a parameter to chose. 

- If $\alpha$ is too large, all the coefficients of the regression equal to zero
- If $\alpha$ is too small and close to 0, we get the coefficients of the usual linear regression.

Thus, we must find a compromise, by choosing alpha by cross-validation. 

The “lasso path” can also help us. This is a graph that relates the value of $\alpha$ to the estimated coefficients.

#### Preparing the test and training sets


```python
#--------------------------------------------------------------------------------
# Feature Selection : Picking up the most predictive features 
#--------------------------------------------------------------------------------
#We distinguish numerical and categorical features
numeric_features = ["lead_time", "arrival_date_year",
                    "adr",
                    "arrival_date_week_number",
                    "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights", 
                    "adults", "children",
                    "babies", "previous_cancellations", 
                    "days_in_waiting_list", "previous_bookings_not_canceled", "booking_changes", 
                    "total_of_special_requests"]
categorical_features = ["meal", "country", "market_segment", "distribution_channel",
                        "hotel", "reserved_room_type", "deposit_type","agent", 
                       "customer_type","is_repeated_guest","arrival_date_month"]                        
features = numeric_features + categorical_features

#--------------------------------------------------------------------------------
# Feature Engineering : Process of converting raw data into a structured format 
# Making the data ready to use for model training (transformers)
# Creating the test and training sets
#--------------------------------------------------------------------------------

numeric_transformer = Pipeline(steps=[
                                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                                    ("scaler", MinMaxScaler())]) 
categorical_transformer = Pipeline(steps=[
                                    ("imputer", SimpleImputer(strategy="constant",
                                                              fill_value="Not defined")),
                                    ("onehot", OneHotEncoder(handle_unknown='ignore'))]) #
preproc = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features),
                                          ("cat", categorical_transformer, categorical_features)])

X = data2.drop(["is_canceled","assigned_room_type","reservation_status","reservation_status_date",
                "company"],
              axis=1)[features]
y = data2["bool_parking"]

X, y = shuffle(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y)

X_preproc = preproc.fit_transform(X)
X_train_preproc, X_test_preproc, y_train_preproc, y_test_preproc = train_test_split(X_preproc, y)

```

#### Running the model using Cross-validation


```python
#lasso path 
from sklearn.linear_model import lasso_path
my_alphas = np.array([0.001,0.01,0.02,0.025,0.05,0.1,0.25,0.5,0.8,1.0])
alpha_for_path, coefs_lasso, _ = lasso_path(X_train_preproc,y,alphas=my_alphas)
nb_coeff = coefs_lasso.shape[0] #578
nb_alpha = coefs_lasso.shape[1]
import matplotlib.cm as cm
couleurs = cm.rainbow(np.linspace(0,1,nb_coeff))

#lasso path plot(one curve per variable)
plt.figure(figsize=(3, 3))
fig, ax1 = plt.subplots()
for i in range(nb_coeff):
    ax1.plot(alpha_for_path,coefs_lasso[i,:],c=couleurs[i])
plt.xlabel('Alpha')
plt.xlim(0,0.05)
plt.ylabel('Coefficients')
plt.title('Lasso path')
plt.show()
```


    <Figure size 216x216 with 0 Axes>



    
![png](TP2_KA_IB_files/TP2_KA_IB_101_1.png)
    



```python
#number of non-zero coefficient(s) for each alpha
nbNonZero = np.apply_along_axis(func1d=np.count_nonzero,arr=coefs_lasso,axis=0)
plt.figure(figsize=(4, 2))
plt.plot(alpha_for_path,nbNonZero)
plt.xlabel('Alpha')
plt.ylabel('Nb. de variables')
plt.title('Nb. variables vs. Alpha')
plt.show()
```


    
![png](TP2_KA_IB_files/TP2_KA_IB_102_0.png)
    



```python
#function to create labels for one-hot vectors
# inspiration : https://stackoverflow.com/questions/41987743/merge-two-multiindex-levels-into-one-in-pandas
def labels_one_hot(data, categorical, numeric):
    enc_cols=[]
    for var in categorical:
        uniq_val = X[var].unique().astype(str).tolist()
        indice_nan = [i for i,x in enumerate(uniq_val) if x == 'nan']
        for index in indice_nan:
            uniq_val[index] = "undefined"
        uniq_val = [var + "_" + sub for sub in uniq_val] 
        enc_cols = enc_cols + uniq_val
    cols = numeric + enc_cols
    return(cols)
```


```python
nom_var = labels_one_hot(X,categorical_features,numeric_features)
#print the non-zero coefficients for alpha = 0.001
coeff001 = pd.DataFrame({'Variables':nom_var,'Coefficients':coefs_lasso[:,9]}) #alpha = 0.001
coeff001[coeff001['Coefficients']>0]
#print(coeff001[coeff001['Coefficients']>0].to_markdown())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variables</th>
      <th>Coefficients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>arrival_date_week_number</td>
      <td>0.000007</td>
    </tr>
    <tr>
      <th>15</th>
      <td>meal_BB</td>
      <td>0.001885</td>
    </tr>
    <tr>
      <th>156</th>
      <td>country_SLV</td>
      <td>0.002526</td>
    </tr>
    <tr>
      <th>209</th>
      <td>distribution_channel_GDS</td>
      <td>0.004029</td>
    </tr>
    <tr>
      <th>212</th>
      <td>hotel_Resort Hotel</td>
      <td>0.000188</td>
    </tr>
    <tr>
      <th>213</th>
      <td>reserved_room_type_A</td>
      <td>0.004701</td>
    </tr>
    <tr>
      <th>223</th>
      <td>deposit_type_Non Refund</td>
      <td>0.017430</td>
    </tr>
    <tr>
      <th>562</th>
      <td>customer_type_Contract</td>
      <td>0.006019</td>
    </tr>
    <tr>
      <th>564</th>
      <td>is_repeated_guest_0</td>
      <td>0.032911</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Choose the best alpha using cros-validation
from sklearn.linear_model import LassoCV
lcv = LassoCV(alphas=my_alphas,normalize=False,fit_intercept=False,random_state=0,cv=5)
lcv.fit(X_train_preproc,y_train_preproc) 
#mean of MSE for each alpha
avg_mse = np.mean(lcv.mse_path_,axis=1)
#alphas and MSE 
print(pd.DataFrame({'alpha':lcv.alphas_,'MSE':avg_mse}))
#best alpha
print("best alpha:", lcv.alpha_) #0.001
```

       alpha       MSE
    0  1.000  0.061803
    1  0.800  0.061803
    2  0.500  0.061803
    3  0.250  0.061803
    4  0.100  0.061803
    5  0.050  0.060326
    6  0.025  0.056838
    7  0.020  0.056129
    8  0.010  0.055185
    9  0.001  0.052854
    best alpha: 0.001
    

#### Running the models with the best alpha and evaluating with different scores


```python
from sklearn.linear_model import Lasso
lasso = Lasso(fit_intercept=False,normalize=False, alpha=0.001) #best parameter lasso : alpha=0.001
lasso.fit(X_train_preproc,y_train_preproc)
#print(lasso.coef_)
y_pred = lasso.predict(X_test_preproc)
print("MSE lasso: ", mean_squared_error(y_test_preproc, y_pred, squared=False))
print("R2 lasso: ", r2_score(y_test, y_pred))
#print("accuracy lasso : ", accuracy_score(y_test_preproc, y_pred)) #does not exist because Y_pred not binary
```

    MSE lasso:  0.23160623528346608
    R2 lasso:  -0.06815575882526304
    

**=> The best alpha is the lowest (0.001). It is logical because, as we said earlier, the number of features is low (p=30) compared to the number of individuals (n=+100 000). Therefore, regularized regressions are not necessary in our case. However, we can see that the MSE obtained on the test set (0.23) is lower (and thus better) than the one of the previous models (0.25), which means that the regularization may have improved a little bit the model !**
