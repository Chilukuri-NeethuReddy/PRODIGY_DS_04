# PRODIGY_DS_04

# ProdigyInfoTech_TASK4

## TASK 4: Analyzing and Visualizing Sentiment Patterns in Social Media Data

This project analyzes sentiment patterns in social media data to understand public opinion and attitudes towards specific topics or brands using the NLTK Vader sentiment analyzer.

## Dataset:

The dataset used in this project is the Tweets dataset, containing tweets related to US airline sentiment.
https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset

## Steps

1. *Data Loading*: Load the dataset and explore its structure.
  
2. *Data Preprocessing*: Check for missing values and analyze sentiment distribution.
   
3. *Sentiment Analysis*: Use NLTK's Vader SentimentIntensityAnalyzer to calculate sentiment scores for each tweet.
   
4. *Visualization*: Visualize sentiment distribution, sentiment scores distribution, and sentiment scores by airline.

## How to Run

1. Clone the repository.
   
2. Ensure you have the necessary libraries installed (pandas, matplotlib, seaborn, nltk).
 
3. Place the dataset (Tweets.csv) in the same directory as the script.
 
4. Run the script (sentiment_analysis.py) to perform sentiment analysis and generate visualizations.

## Code

```python

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from IPython.display import display

# Load the dataset

df = pd.read_csv('Tweets.csv')

# Explore the structure and first few rows

display(df.head())

# Check for missing values

print(df.isnull().sum())
```

![Screenshot 2024-07-07 110324](https://github.com/Chilukuri-NeethuReddy/PRODIGY_DS_04/assets/174725064/70e0310d-051f-438e-b149-eb0b793b6683)

![Screenshot 2024-07-07 110438](https://github.com/Chilukuri-NeethuReddy/PRODIGY_DS_04/assets/174725064/10348037-4ac0-4e62-ba6a-2ee939cd533c)

![Screenshot 2024-07-07 110452](https://github.com/Chilukuri-NeethuReddy/PRODIGY_DS_04/assets/174725064/1570b1dc-51e4-40ef-90dd-d631b6232041)

```python

# Preprocess the data: We will focus on 'airline_sentiment' and 'text' columns

df = df[['airline_sentiment', 'airline', 'text']]

# Define custom color palettes or colormaps

custom_palette = ["#FF0000", "#0000FF", "#FFA500"]  # Red, Blue, Orange

# Plot sentiment distribution with custom colors

plt.figure(figsize=(8, 6))

sns.countplot(x='airline_sentiment', data=df, palette=custom_palette)

plt.title('Sentiment Distribution in US Airline Tweets')

plt.xlabel('Sentiment')

plt.ylabel('Count')

plt.show()
```

![Screenshot 2024-07-07 110513](https://github.com/Chilukuri-NeethuReddy/PRODIGY_DS_04/assets/174725064/89522de6-a975-476e-aff3-5e5c58cae4f5)

```python
# Visualize sentiment scores by airline

plt.figure(figsize=(12, 8))

custom_palette = ["#00FF00", "#0000FF", "#FF0000"] 

sns.boxplot(x='airline', y='sentiment_scores', data=df, palette=custom_palette)

plt.title('Sentiment Scores by Airline')

plt.xlabel('Airline')

plt.ylabel('Sentiment Score')

plt.xticks(rotation=45)

plt.show()
```
###  sentiment scores by airline


![Screenshot 2024-07-07 110536](https://github.com/Chilukuri-NeethuReddy/PRODIGY_DS_04/assets/174725064/9e1d53b4-d87d-40d0-b0ac-0210cc0bf9ff)


```python
# Plot the distribution of sentiment scores

plt.figure(figsize=(10, 6))

sns.histplot(df, x='sentiment_scores', bins=30, kde=True, color="#90EE90")

plt.title('Distribution of Sentiment Scores')

plt.xlabel('Sentiment Score')

plt.ylabel('Count')

plt.show()
```

###  distribution of sentiment scores
![Screenshot 2024-07-07 110552](https://github.com/Chilukuri-NeethuReddy/PRODIGY_DS_04/assets/174725064/6f441463-39ff-4912-a82d-87369cadcf7d)






