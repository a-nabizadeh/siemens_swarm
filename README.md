# [Swarm behavior on the Grid](https://ecosystem.siemens.com/techforsustainability/swarm-behaviour-on-the-grid/overview)

### Siemens sustainability challenge


## Section 1: data  

### **PJM Hourly Energy Consumption Data**
Dataset: [kaggle link](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)

PJM Interconnection LLC (PJM) is a regional transmission organization (RTO) in the United States. It is part of the Eastern Interconnection grid operating an electric transmission system serving all or parts of Delaware, Illinois, Indiana, Kentucky, Maryland, Michigan, New Jersey, North Carolina, Ohio, Pennsylvania, Tennessee, Virginia, West Virginia, and the District of Columbia.

The hourly power consumption data comes from PJM's website and are in megawatts (MW).

Raw data have a range of [issues](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption/discussion/175795?select=pjm_hourly_est.csv&sort=votes). A Kaggle user prepared this [gist](https://gist.github.com/ioanpier/e231b22bb9f705ef6280c8b73e40b4a1) for data cleaning which we use here (`process_missing_and_duplicate_timestamps` in  `utils.py`).
