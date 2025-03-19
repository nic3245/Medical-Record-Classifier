# Overview

This project is based on Track 2 of the 2018 n2c2 challenge, found here - https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

Note: data is not included in repository as it was not allowed in the data agreement. Please see the website for more details.

Please follow the instructions at the top of each notebook.  

You may go through the notebooks in any order.  

If you want to run the openAI API code, you will need your own key, and it will likely cost money.  

Everything else is runnable with Python 3.10.3 and the packages: tensorflow transformers numpy pandas nltk matplotlib openai

## Baseline

A Logistic Regression model is implemented as a baseline, located in LRBaseline.ipynb.

## LLMs

ClinicalBERT is evaluated in ClinicalBERTEvaluation.ipynb. ChatGPT is also used to summarize the clinical notes and then pass into ClinicalBERT in ChatGPTSummaryAndClinicalBERT.ipynb. ChatGPT is evaluated in ChatGPTEvaluation.ipynb.

## Results

Results can be found in the report - Report on Results.pdf
