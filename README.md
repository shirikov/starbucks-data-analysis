# Analysis of Starbucks Data

This repo contains the scripts for an analysis of marketing campaign effectiveness. The analysis examines how several promotional offers were received by customers in different segments. The data were provided by Starbucks as part of a Udacity Data Science course; Starbucks simulated the data based on the patterns in the actual data from the app.

## Required Software

Python 3 (e.g., an Anaconda installation) with the following libraries installed: pandas, numpy, sklearn.

## Analysis Goals

The main question explored in this project is how the probability of completing an offer and receiving a reward changes depending on the customer segment. To answer that question, we need to understand what variables variables predict the completion of promotional offers. Completion in this case means spending the amount specified in the promotion as a requirement for redeeming the offered bonus. As an example, there may be a BOGO offer that requires one to spend 5 dollars in order to get a 5 dollar reward, presumably in the form of a free product.

The analysis sets up a model to predict offer completion using available data on customers and offers, and then it considers what variables may increase or decrease the probability of completion. In addition, the project uses the same setup to examine the variables that may affect the probability of viewing the offer in the app.

## Original Data

The original data are in three files:

- portfolio.json - offer ids and meta data about each offer (duration, type, etc.);
- profile.json - demographic data for each customer;
- transcript.json - records for transactions, offers received, offers viewed, and offers completed.

Schema/explanation of each variable in the files:

**portfolio.json**
- id (string) - offer id
- offer_type (string) - type of offer: BOGO, discount, informational
- difficulty (int) - minimum required spend to complete an offer
- reward (int) - reward given for completing an offer
- duration (int) - time for offer to be open, in days
- channels (list of strings)

**profile.json**
- age (int) - age of the customer 
- became_member_on (int) - date when customer created an app account
- gender (str) - gender of the customer (some entries contain 'O' for other rather than M or F)
- id (str) - customer id
- income (float) - customer's income

**transcript.json**
- event (str) - record description (transaction, offer received, offer viewed, offer completed)
- person (str) - customer id
- time (int) - time in hours since the start of test
- value - (dict of strings) - either an offer id or transaction amount depending on the record

## How to Run the Analysis

All the code is contained in three Jupyter notebooks:

(1) **starbucks_data_exploration.ipynb**: an exploratory analysis of the data, including the distribution of key variables of interest and some code to understand how the offers were sent to customers.

(2) **starbucks_data_cleanup.ipynb**: code to clean up and reshape the data, combining variables from three original JSON files into one data set, saved as '*data/cleaned_offer_data.csv*'. This file is already saved in the repository, so there is no need to run this notebook if one simply wants to check out the analysis.

(3) **starbucks_offer_completion_analysis.ipynb**: code to run the analysis, which includes model tuning, code to obtain the results with respect to the questions of interest, a discussion of these results, and takeaways from the analysis.

To run the code, you can simply open each notebook and execute it.

## Analysis Results

Two different machine learning algorithms were used to predict offer completion: logistic regression and gradient boosting. These two models produced almost identical results in terms of prediction accuracy, so the main analysis was conducted using logistic regression given that it allowed for straightforward interpretation of regression coefficients.

Customers have completed more than half of offers they received. Offer completion probability differed somewhat based on gender, age, income, how long one has been a member, and on the specific offer characteristics. However, many customers completed the offers possibly without being aware that they received these offers (they did not view the corresponding promotions in the app); in other words, they may have completed these promotions just because they were spending money as usual. In many customer segments, there was not much difference in the probability of completing the offer depending on whether one viewed the offer in the app or not.

The notebook **starbucks_offer_completion_analysis.ipynb** discusses these results and a number of potential practical takeaways. The project, data exploration, and the results of the analysis are also described in a [blog post](https://medium.com/@antonshirikov/who-gets-the-bonuses-predicting-the-completion-of-promotional-offers-in-the-starbucks-app-80a83f8eaeb3).

## Acknowledgments

The data used in the analysis were made available to Udacity students by Starbucks solely for the purpose of this analysis and cannot be used elsewhere. The code provided in this repo can be used and modified as needed.