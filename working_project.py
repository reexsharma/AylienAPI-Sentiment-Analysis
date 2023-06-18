import matplotlib.pyplot as plt
import pandas as pd
import re
import jsonlines
from textblob import TextBlob
import wordcloud as WordCloud

JSON_FILE = "news_output.jsonl"
OTHER_JSON_FILE = "news_output2.jsonl"


def read_jsonl(file_path):
    with jsonlines.open(file_path) as file:
        data = [article for article in file]
    return data


def process_words(summaries):
    processed_summaries = []  # Create an empty list to store the processed summaries
    for summary in summaries:
        processed_sentences = []
        for sentence in summary:
            sentence = sentence.replace('\n', ' ')  # Remove the newline characters
            sentence = sentence.strip()  # Removing leading/trailing whitespace
            sentence = re.sub('[^\w\s;]', '', sentence)  # Remove punctuation
            sentence = sentence.lower()  # Make all characters lowercase
            sentence = ' '.join(word for word in sentence.split() if word.isalpha())  # Keep only alphabetic words
            processed_sentences.append(sentence)
        processed_summaries.append(processed_sentences)
    return processed_summaries


def remove_stop_words(processed_summaries, stop_words_file):
    '''
    Function: remove_stop_words
    Does: Removes words from the stop words file from the text
    Parameters:
        processed_summaries: a list of dictionaries of processed summaries of articles (str)
    Returns: filtered_sums: a new list of dictionaries, filtered for common stop words (str)
    '''

    # Read stop words from the file and store them in a set
    with open(stop_words_file, 'r') as file:
        stop_words = set(file.read().lower().split())

    # Append to filtered_sums, a new list of dicts only if word not in stop_words
    filtered_sums = []
    for summary in processed_summaries:
        filtered_sum = [word for word in summary if word.lower() not in stop_words]
        filtered_sums.append(filtered_sum)
    return filtered_sums


def find_summaries(data):
    summaries = []
    for entry in range(len(data)):
        summaries.append(data[entry]['summary']['sentences'])
    return summaries


def find_datetimes(data):
    date_times = []
    date_times.append([data[entry]['published_at'] for entry in range(len(data))])
    return date_times


def sentiment(filtered_sums):
    sentiment_polarity = []
    # Perform sentiment analysis on each filtered summary
    for summary in filtered_sums:
        blob = TextBlob(str(summary))
        sentiment = blob.sentiment.polarity
        sentiment_polarity.append(sentiment)
    return sentiment_polarity


def plot_sentiment_distribution(sentiment_scores, timestamps):
    # Create a DataFrame with sentiment scores and timestamps
    df = pd.DataFrame({'Sentiment': sentiment_scores, 'Timestamp': pd.to_datetime(timestamps)})

    # Group sentiment scores by month and calculate mean sentiment for each month
    weekly_sentiment = df.groupby(pd.Grouper(key='Timestamp', freq='W')).mean()

    # Plotting the sentiment distribution
    _, ax = plt.subplots()
    ax.plot(weekly_sentiment.index, weekly_sentiment['Sentiment'], color='green')
    ax.set_xlabel('Week')
    ax.set_ylabel('Average Sentiment')
    ax.set_title('Weekly Average Sentiment Polarity Over Time')

    plt.xticks(rotation=45)
    plt.savefig('polarity_distribution.png')
    plt.show()

def most_polar(sentiment_scores, timestamps):
    # Create a DataFrame with sentiment scores and timestamps
    df = pd.DataFrame({'Sentiment': sentiment_scores, 'Timestamp': pd.to_datetime(timestamps)})

    # Group sentiment scores by month and calculate mean sentiment for each month
    weekly_sentiment = df.groupby(pd.Grouper(key='Timestamp', freq='W')).mean()

    top5_values = weekly_sentiment['Sentiment'].nlargest(5)
    top5_rows = df.loc[df['Sentiment'].nlargest(5).index]
    return top5_rows

def create_wcloud(top5_rows, filtered_sums):
    wordcloud = []
    for i in range(filtered_sums):
        if i in top5_rows:
           wordcloud.append(filtered_sums[i])
        else:
            continue
    for sum in wordcloud:
        wc = WordCloud(width = 800, height = 400)
        wc.generate(sum)

        plt.figure()
        plt.imshow(wc, interpolation = 'bilinear')
        plt.axis("off")
        plt.show()

def main():
    # All data
    data = read_jsonl(JSON_FILE)

    # Pre-processed summaries
    summaries = find_summaries(data)

    # Processed summaries
    processed_summaries = process_words(summaries)

    # Filtered for stop words
    filtered_sums = remove_stop_words(processed_summaries, 'stop_words_english.txt')

    # Extract dates
    dates = find_datetimes(data)

    # Sentiment polarities
    polarities = sentiment(filtered_sums)

    # Plot sentiment distribution plot
    plot_sentiment_distribution(polarities, dates[0])

    # Find the 5 most polar artciles using visualization
    polar_5 = most_polar(polarities, dates[0])

    # Generate a wordcloud of the top 5 most polar articles
    create_wcloud(polar_5, filtered_sums)

if __name__ == '__main__':
    main()

    