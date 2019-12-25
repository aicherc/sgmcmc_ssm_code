These python scripts demo fitting the Stochastic Volatility Model (SVM) on the exchange rate data `EURUS_data.csv` between the EURO and USD from November 2017 to October 2018.

The scripts demonstrate how to fit SVM using SGMCMC and LD using both the naive and PaRIS particle smoothers.

* `process_exchange_data.py` converts the raw exchange rate csv into log-returns and saves it in a compressed numpy format.
* `exchange_rate_single_demo.py` fit the SVM to a single segment of the exchange rate data. (e.g. a week)
* `exchange_rate_subset_demo.py` fit the SVM to the first five segments of the exchange rate data (e.g. a month).
* `exchange_rate_full_demo.py` fit the SVM to the all the exchange rate data (e.g. the whole year). This script takes much longer to run - a few hours.


Additional documentation can be found within each script.
