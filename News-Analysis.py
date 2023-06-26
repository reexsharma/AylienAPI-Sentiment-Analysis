import matplotlib.pyplot as plt
import pandas as pd
import re
import jsonlines
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.dates as mdates
from scipy import stats
from scipy.signal import argrelextrema
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Filename of the JSONL file
JSON_FILE = "news_output3.jsonl"


def read_jsonl(file_path):
    """Reads a jsonl file and returns it as a list of dictionaries.
    
    Args:
        file_path (str): The path to the jsonl file.

    Returns:
        data (list): A list of dictionaries each containing a single line from the jsonl file.
    """
    with jsonlines.open(file_path) as file:
        data = [article for article in file]
    return data


def find_titles(data):
    """Finds and returns all the titles in the data.
    
    Args:
        data (list): The data from which to extract the titles.

    Returns:
        titles (list): A list of titles.
    """
    titles = []
    for entry in range(len(data)):
        titles.append(data[entry]['title'])
    return titles


def process_words(titles):
    """Processes the titles by removing newlines, non-alphanumeric characters and non-alpha words,
    and converting to lowercase.
    
    Args:
        titles (list): A list of titles.

    Returns:
        processed_titles (list): A list of processed titles.
    """
    processed_titles = []
    for title in titles:
        title = title.replace('\n', ' ')
        title = title.strip()
        title = re.sub('[^\w\s;]', '', title)
        title = title.lower()
        title = ' '.join(word for word in title.split() if word.isalpha())
        processed_titles.append(title)
    return processed_titles


def remove_stop_words(processed_titles, stop_words_file):
    """Removes stop words from the titles.
    
    Args:
        processed_titles (list): A list of processed titles.
        stop_words_file (str): The path to the file containing the stop words.

    Returns:
        filtered_titles (list): A list of titles with stop words removed.
    """
    with open(stop_words_file, 'r') as file:
        stop_words = set(file.read().lower().split())

    filtered_titles = []
    for title in processed_titles:
        title_words = title.split()  # Split the title into words
        filtered_title = ' '.join(word for word in title_words if word.lower() not in stop_words)  # Iterate over words, not characters
        filtered_titles.append(filtered_title)
    return filtered_titles


def find_datetimes(data):
    """Finds and returns all the publication dates in the data.
    
    Args:
        data (list): The data from which to extract the publication dates.

    Returns:
        date_times (list): A list of publication dates.
    """
    date_times = [data[entry]['published_at'] for entry in range(len(data))]
    return date_times


def find_country_state(data):
    """Finds and returns all the country, state and sentiment information in the data.
    
    Args:
        data (list): The data from which to extract the country, state and sentiment information.

    Returns:
        countries (list): A list of tuples each containing the country, state and sentiment information.
    """
    
    countries = []
    for entry in range(len(data)):
        locations = data[entry]['source']['locations']
        title = data[entry]['title']
        for location in range(len(locations)):
            country = data[entry]['source']['locations'][location]['country']
            state = data[entry]['source']['locations'][location]['state']
            countries.append((country, state, sentiment(title)))
    return countries


def country_df(countries):
    """
    Create a DataFrame from a list of countries, states and sentiment scores. 
    Removes entries without a state. 

    Args:
        countries (list): A list of tuples, each containing the country, state and sentiment information.

    Returns:
        df (DataFrame): Pandas DataFrame with 'Country', 'State', and 'Sent Score' as columns. 
    """
    df = pd.DataFrame(countries, columns=['Country', 'State', 'Sent Score'])
    df.dropna(subset=['State'], inplace=True)
    df = df[df['State'] != '']
    return df


def sentiment(title):
    """
    Compute the sentiment score of a given text.

    Args:
        title (str): Text to compute sentiment score for.

    Returns:
        sentiment (float): Sentiment score of the text.
    """

    # Calculate Sentiment Score
    blob = TextBlob(str(title))
    sentiment = blob.sentiment.polarity
    return sentiment


def weekly_sentiment(sentiment_scores, timestamps):
    """
    Calculate the average sentiment per week.

    Args:
        sentiment_scores (list): List of sentiment scores.
        timestamps (list): List of timestamps corresponding to the sentiment scores.

    Returns:
        weekly_sentiment (DataFrame): Pandas DataFrame with timestamps and the average sentiment for that week.
    """
    # Sentiment avg is being sorted by week to week change
    df = pd.DataFrame({'Sentiment': sentiment_scores, 'Timestamp': pd.to_datetime(timestamps)})
    weekly_sentiment = df.groupby(pd.Grouper(key='Timestamp', freq='W')).mean()
    return weekly_sentiment


def plot_sentiment_distribution(weekly_sentiment):
    """
    Plot the distribution of sentiment scores over time.

    Args:
        weekly_sentiment (DataFrame): DataFrame with timestamps and the average sentiment for that week.
    """
    _, ax = plt.subplots(figsize=(12, 6))
    ax.plot(weekly_sentiment.index, weekly_sentiment['Sentiment'], color='green')
    ax.set_xlabel('Week')
    ax.set_ylabel('Average Sentiment')
    ax.set_title('Weekly Average Sentiment of Polarity Over Time')

    ax.xaxis.set_tick_params(rotation=45)
    ax.xaxis.set_tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig('polarity_distribution.png')
    plt.show()


def count_timestamps_per_sentiment(sentiment_scores, timestamps):
    """
    Count the number of timestamps for each sentiment score.

    Args:
        sentiment_scores (list): List of sentiment scores.
        timestamps (list): List of timestamps corresponding to the sentiment scores.

    Returns:
        timestamp_counts (DataFrame): DataFrame with sentiment scores and the number of corresponding timestamps.
    """
    # Create a DataFrame with sentiment scores and timestamps
    df = pd.DataFrame({'Sentiment': sentiment_scores, 'Timestamp': pd.to_datetime(timestamps)})

    # Group by sentiment and count the number of timestamps
    timestamp_counts = df.groupby('Sentiment').count()

    # Rename the column for clarity
    timestamp_counts.rename(columns={'Timestamp': 'Count'}, inplace=True)

    print(timestamp_counts)

    return timestamp_counts


def moving_average(sentiment_scores, dates, window_size):
    """
    Calculate the moving average of sentiment scores.

    Args:
        sentiment_scores (list): List of sentiment scores.
        dates (list): List of dates corresponding to the sentiment scores.
        window_size (int): Size of the moving average window.

    Returns:
        Timestamps (Series), moving_avg (Series): Dates and corresponding moving average of sentiment scores.
    """

    df = pd.DataFrame({'Sentiment': sentiment_scores, 'Timestamp': pd.to_datetime(dates)})
    df = df.sort_values('Timestamp')  # Ensure data is sorted by date
    df = df[(np.abs(stats.zscore(df['Sentiment'])) < 3)]  # Remove outliers
    moving_avg = df['Sentiment'].rolling(window=window_size).mean()  # Calculate moving average
    return df['Timestamp'], moving_avg


def plot_sentiment_moving_average(moving_avg_dates, moving_avg):
    """
    Plot the moving average of sentiment scores over time.

    Args:
        moving_avg_dates (Series): Dates.
        moving_avg (Series): Corresponding moving average of sentiment scores.
    
    Returns:
        A matplotlib plot of the moving average of sentiment scores over time
    """

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot data
    ax.plot(moving_avg_dates, moving_avg, color='blue')

    # Annotate Graph
    ax.set_xlabel('Date', labelpad=20)
    ax.set_ylabel('Moving Average Sentiment', labelpad=20)
    ax.set_title('Moving Average Sentiment of Generative AI Over Time')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))

    # Plot the date of the release of ChatGPT
    chatgpt_release_date = pd.to_datetime('2022-11-30')  # Nov 30, 2022 is the ChatGPT release date
    ax.axvline(chatgpt_release_date, color='red', linestyle='--')  # Add vertical line
    plt.text(chatgpt_release_date, ax.get_ylim()[1], 'ChatGPT 3.5 Release', fontsize=10,
             verticalalignment='top')  # Add text

    ax.xaxis.set_tick_params(rotation=45, labelsize=8)
    
    # Save and Show file
    plt.tight_layout()
    plt.savefig('moving_average.png')
    plt.show()


def categorize_sentiment_scores(sentiment_scores):
    """
    Categorize sentiment scores into 'Very bad', 'Bad', 'Good' and 'Very good'.

    Args:
        sentiment_scores (list): List of sentiment scores.

    Returns:
        categories (list): List of categories corresponding to the sentiment scores.
    """
    # Instantiate the categories empty list
    categories = []
    for score in sentiment_scores:
        if -1.0 <= score < -0.5:
            categories.append('Very bad')
        elif -0.5 <= score < 0:
            categories.append('Bad')
        elif 0 < score <= 0.5:
            categories.append('Good')
        else:
            categories.append('Very good')
    return categories


def create_wordclouds(df):
    """
    Creates and displays word clouds for each category of sentiment.

    Args:
        df (DataFrame): A DataFrame containing the titles and their associated sentiment categories.

    Returns:
        Word clouds for each sentiment category.
    """
    for category in ['Very bad', 'Bad', 'Good', 'Very good']:
        # Filter summaries of a specific category
        summaries = df[df['Category'] == category]['Title']

        # Check if there are any summaries in this category
        if summaries.empty:
            print(f"No titles in category {category}")
            continue

        # Join all summaries into a single string
        text = ' '.join(summaries)

        # Generate word cloud
        wordcloud = WordCloud(width=1400, height=1000).generate(text)

        # Display the word cloud
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud for {category} Sentiment')
        plt.axis("off")
        plt.show()


def create_violin_plot(data, x_col, y_col, title):
    """
    Creates and displays a violin plot.

    Args:
        data (DataFrame): The DataFrame to use for the plot.
        x_col (str): The column in 'data' to use for the x-axis.
        y_col (str): The column in 'data' to use for the y-axis.
        title (str): The title of the plot.
    """
    country_counts = data['Country'].value_counts().head(6)
    data = data[data['Country'].isin(country_counts.index)]  # Filter to only include top 6 countries

    # Create a list of violin plots, one for each country
    violin_plots = []
    for country in country_counts.index:
        violin_plots.append(go.Violin(
            x=data[data['Country'] == country][x_col],
            y=data[data['Country'] == country][y_col],
            name=country,
            box_visible=True,
            meanline_visible=True))

    layout = go.Layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col)

    fig = go.Figure(data=violin_plots, layout=layout)
    fig.write_html("countries_plotly.html")
    fig.show()


def plot_volume_over_time(dates):
    """
    Plots the number of articles over time.

    Args:
        dates (list): A list of dates when the articles were published.
    
    Returns:
        A matplotlib line plot of the number of articles over time.
    """
    
    # Organize Datetime Data w/ article counts
    df = pd.DataFrame({'Timestamp': pd.to_datetime(dates)})
    df = df.sort_values('Timestamp')  # Ensure data is sorted by date
    df['Count'] = 1

    # Create Graph
    df.set_index('Timestamp', inplace=True)
    df.resample('W').sum().plot(kind='line')  # Resample by week
    plt.axvline(pd.to_datetime('2022-11-30'), color='red', linestyle='--')  # ChatGPT release date
    plt.title('Number of Articles Over Time')
    plt.ylabel('Count')
    plt.show()


def plot_sentiment_distribution_before_after(sentiment_scores, dates):
    """
    Plots the distribution of sentiment scores before and after the release of ChatGPT.

    Args:
        sentiment_scores (list): A list of sentiment scores.
        dates (list): A list of dates when the articles were published.
    """
    ...
    df = pd.DataFrame({'Sentiment': sentiment_scores, 'Timestamp': pd.to_datetime(dates)})
    df = df.sort_values('Timestamp')  # Ensure data is sorted by date
    mask = df['Timestamp'] < pd.to_datetime('2022-11-30')  # ChatGPT release date

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")  # Set style for the plot
    sns.histplot(df[mask]['Sentiment'], bins=50, color='blue', kde=True, label='Before Release', element='step')
    sns.histplot(df[~mask]['Sentiment'], bins=50, color='orange', kde=True, label='After Release', element='step')

    # Add labels and title
    plt.title('Sentiment Distribution Before and After ChatGPT Release', fontsize=16, pad=20)
    plt.xlabel('Sentiment', fontsize=13, labelpad=15)
    plt.ylabel('Density', fontsize=13, labelpad=15)

    # Add legend and grid
    plt.legend(title='Period', title_fontsize='13', loc='upper right', fontsize='12')

    # Remove top and right spines
    sns.despine()
    plt.show()


def monthly_sentiment(sentiment_scores, timestamps):
    """
    Calculates the monthly average sentiment scores.

    Args:
        sentiment_scores (list): List of sentiment scores.
        timestamps (list): List of corresponding timestamps.

    Returns:
        pandas.DataFrame: DataFrame with monthly average sentiment scores.
    """
    
    # Same thing as Weekly scores, but with freq of 'Month'
    df = pd.DataFrame({'Sentiment': sentiment_scores, 'Timestamp': pd.to_datetime(timestamps)})
    monthly_sentiment = df.groupby(pd.Grouper(key='Timestamp', freq='M')).mean()
    return monthly_sentiment


def find_peaks(data, comparator):
    """
    Find the peaks in a series of data.

    Args:
        data (array-like): The data to find peaks in.
        comparator (callable): A comparator function that takes two arguments and returns a boolean.

    Returns:
        peaks (array-like): The indices of the peaks in the data.
    """

    return argrelextrema(data.to_numpy(), comparator)


def annotate_extrema(ax, dates, sentiments, sorted_titles, peaks, valleys):
    """
    Annotates the extrema points in the sentiment over time plot.

    Args:
        ax (Axes): The axes to draw the annotations on.
        dates (list): The dates of the articles.
        sentiments (list): The sentiment scores of the articles.
        sorted_titles (list): The titles of the articles, sorted by date.
        peaks (list): The indices of the peak sentiment scores.
        valleys (list): The indices of the valley sentiment scores.
    """
    # Annotating peaks in graph
    for peak in peaks:
        ax.annotate(sorted_titles[peak], (mdates.date2num(dates[peak]), sentiments[peak]),
                    xytext=(15, 15), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', lw=1.5))
    # Annotating lows in graph
    for valley in valleys:
        ax.annotate(sorted_titles[valley], (mdates.date2num(dates[valley]), sentiments[valley]),
                    xytext=(-60, -30), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', lw=1.5))

def plot_sentiment_with_extrema(weekly_sentiment, sorted_titles):
    """
    Plots the sentiment over time, with markers for the peaks and valleys.

    Args:
        weekly_sentiment (DataFrame): A DataFrame with weekly sentiment scores.
        sorted_titles (list): The titles of the articles, sorted by date.
    """
    # Create a dataframe with sentiment scores, their corresponding dates and titles
    df = pd.DataFrame({"Date": weekly_sentiment.index, "Sentiment": weekly_sentiment["Sentiment"], "Title": sorted_titles})

    # Find peaks and valleys
    peaks = find_peaks(df["Sentiment"], np.greater)
    valleys = find_peaks(df["Sentiment"], np.less)

    # Add marker properties based on peaks and valleys
    df["marker_size"] = [10 if i in peaks[0] else (5 if i in valleys[0] else 0) for i in range(len(df))]
    df["marker_color"] = ["red" if i in peaks[0] else ("blue" if i in valleys[0] else "green") for i in range(len(df))]

    # Create a figure
    fig = go.Figure()

    # Add a scatter plot to the figure
    fig.add_trace(
        go.Scatter(
            x=df["Date"], 
            y=df["Sentiment"], 
            mode="lines+markers",
            marker=dict(
                size=df["marker_size"], 
                color=df["marker_color"],
            ),
            hovertemplate="<b>%{x}</b><br><br>" + "Sentiment: %{y}<br>Title: %{text}",
            text=df["Title"]
        )
    )

    # Update layout
    fig.update_layout(
        title_text="Average Sentiment Polarity Over Time",
        title_x=0.5,
        xaxis_title="Month",
        yaxis_title="Sentiment Score",
        hovermode="x",
    )

    # Show the figure
    fig.write_html("event_annotation.html")
    fig.show()


def main():
    # Read the latest JSON file with all news data
    data = read_jsonl(JSON_FILE)

    # Find and create a dataframe for all countries in selected data
    countries = find_country_state(data)
    countries_df = country_df(countries)

    # Grab list of all titles in data, and process them.
    titles = find_titles(data)
    processed_titles = process_words(titles)

    # Filter the titles by removing stopwords, so that we can make wordclouds from them
    filtered_titles = remove_stop_words(processed_titles, 'stop_words_english.txt')

    # Grab list of dates from available data
    dates = find_datetimes(data)

    # Get a list of polarities for every title in data
    polarities = [sentiment(title) for title in filtered_titles]

    # Sort polarities by month
    sent_scores_monthly = monthly_sentiment(polarities, dates)

    # Sort titles by their publish month and convert to list
    df_titles = pd.DataFrame({'Title': processed_titles, 'Timestamp': pd.to_datetime(dates)})
    df_titles_sorted = df_titles.groupby(pd.Grouper(key='Timestamp', freq='M')).first()
    sorted_titles = df_titles_sorted['Title'].tolist()

    # Plot monthly sentiment change, with titles showing as you hover over peaks/valleys
    plot_sentiment_with_extrema(sent_scores_monthly, sorted_titles)

    # Increase the window size for calculating the moving average to 10
    moving_avg_dates, moving_avg = moving_average(polarities, dates, 4)
    plot_sentiment_moving_average(moving_avg_dates, moving_avg)

    # Categorize sentiment scores
    categories = categorize_sentiment_scores(polarities)

    # Create a DataFrame with word cloud categories and filtered titles
    wc_df = pd.DataFrame({'Category': categories, 'Title': filtered_titles})

    # Create word clouds
    create_wordclouds(wc_df)

    # Plot a violin plot
    create_violin_plot(data=countries_df, x_col='Country', y_col='Sentiment Score', 
                       title='Distribution of Sentiment Scores For Top 6 Countries')

    # Plot number of articles published over time with ChatGPT launch as dashed line.
    plot_volume_over_time(dates)

    # Plot sentiment data before and after launch of ChatGPT
    plot_sentiment_distribution_before_after(polarities, dates)


if __name__ == '__main__':
    main()
