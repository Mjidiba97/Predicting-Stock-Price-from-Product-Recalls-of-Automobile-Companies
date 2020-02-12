# Predicting-Stock-Price-from-Product-Recalls-of-Automobile-Companies
This a data challenge created by me and my collegues that aims to predict the impact of product recalls on the stock price of the automobile companies.

The objective of this challenge is to predict the impact of a product recall on a company's stock price, more precisely we study said impact on the biggest automobile groups. A product recall is a procedure launched by a company to get back a product sold to consumers because it is suscpected to have a defect. 

The correlation between a product recall and a company's share value has been demonstrated by empirical studies (see the article <a href ='https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6288.1987.tb01265.x'>Automotive Recalls and Informational Efficiency</a>). Thus, the impact will be measured through the company's stock price depreciation.

We will use data coming from 3 sources : 

* <b>UK Vehicle Safety Branch Recalls Database</b>: for automobile recalls data (<a href ='https://data.gov.uk/dataset/18c00cf3-3bb2-4930-b30d-78113113aaa7/vehicle-safety-branch-recalls-database'>link</a>)

* <b>Yahoo Finance</b>: for historical stock price data (<a href='https://fr.finance.yahoo.com'>link</a>)

* <b>ADEME</b> (French environment agency): for vehicule models data (<a href='https://www.data.gouv.fr/fr/datasets/emissions-de-co2-et-de-polluants-des-vehicules-commercialises-en-france/#_'>link</a>)

Within the project, you will find a starting kit based on the 'Data Camp' course that we took on our <a href = 'https://datascience-x-master-paris-saclay.fr/'>Master's Degree in Data Science<a>. the starting kit is divided into 4 parts:

* <b>Data folder</b>: contains the RAW data that we gathered for this challenge.
* <b>Jupyter Notebook</b>: contains a full description of the challenge, some exploratory data analysis and a basic prediction model for the reader.
* <b>Submissisons folder</b>: contains two python files for the final submission. The first one (<i>feature_extractor.py</i>) is for feature engineering, and the second one (<i>regressor.py</i>) is for the regression model.
