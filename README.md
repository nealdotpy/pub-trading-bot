# PYTHON TRADING BOT
A python implementation of API trading. The bot will facilitate trades utilizing the Alpaca Platform (initially using the paper endpoint.)

Read the article describing the project here: https://medium.com/@nealdotpy/if-i-cant-make-money-my-python-trading-bot-will-368a592b3c6d

**This repository hosts code that generally outlines the first strategy I used. 
A more detailed repository exists; however, due to the nature of algorithmic trading, they are private.**

This repository exists as a proof-of-concept.

# FUNCTIONALITY
This program will include the following features:
* [X] Scrape a database of stock tickers (NYSE, NASDAQ)
* [X] Store all of these tickers in the GCP BigQuery Table
* [X] Scrape all of the historical stock data and store in a BQT
* [X] Creation of an algorithm that allocates a portfolio (strategy)
* [X] Connect the bot to the algorithm and the Alpaca Trading Account
* [X] Add daily email logging.
* [ ] Experiment with strategies
