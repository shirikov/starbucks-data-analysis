# Analysis of Starbucks data

This repo contains the scripts for an analysis of marketing campaign effectiveness. The analysis examines how several promotional offers were received by customers in different segments. The data were provided by Starbucks as part of a Udacity Data Science course; Starbucks simulated the data based on the patterns in the actual data.

## Required Software

Python 3 (e.g., an Anaconda installation) with the following libraries installed: pandas, numpy, sklearn, matplotlib, datetime.

## Questions Explored in the Project


# Original Data

The original data are in three files:

- portfolio.json - offer ids and meta data about each offer (duration, type, etc.);
- profile.json - demographic data for each customer;
- transcript.json - records for transactions, offers received, offers viewed, and offers completed.

Here is the schema and explanation of each variable in the files:

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

## Files


## Analysis Results


## Acknowledgments

The data used in the analysis were made available to Udacity students by Starbucks solely for the purpose of this analysis and cannot be used elsewhere. The code provided in this repo can be used and modified as needed.