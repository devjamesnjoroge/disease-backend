from flask import Flask, request, jsonify
import pandas as pd
import pickle
from flask_cors import CORS  # Import CORS
import json
import logging  # Import standard Python logging
import os  # Import os to use environment variables

app = Flask(__name__)

CORS(app)  # This enables CORS for all routes

# Enable logging at the debug level using the standard Python logging module
app.logger.setLevel(logging.DEBUG)

# Add a handler to print logs to the console
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)  # Set the handler's log level to DEBUG
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)

# Load the model and vectorizer from the pickle file
with open('LogReg.pkl', 'rb') as file:
    loaded_obj = pickle.load(file)
vectorizer = loaded_obj['vectorizer']
classifier = loaded_obj['classifier']

# Define symptom labels for clarity in the response
symptom_labels = ['respiratory', 'fever', 'fatigue', 'pain', 'gastrointestinal']

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Check if a file was provided in the request
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file provided'}), 400
        
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Check for required columns in the CSV file
        required_columns = ['tweetText', 'tweetURL', 'tweetAuthor', 'handle', 'geo', 'createdAt', 
                            'replyCount', 'quoteCount', 'retweetCount', 'likeCount', 'views', 'bookmarkCount']
        if not all(column in df.columns for column in required_columns):
            return jsonify({'error': f'CSV file must contain the following columns: {required_columns}'}), 400
        
        # Replace empty or NaN values with None (null) for JSON serialization
        df = df.where(pd.notnull(df), None)

        # Extract tweet texts, metadata, and engagement metrics
        tweet_texts = df['tweetText'].tolist()
        tweet_urls = df['tweetURL'].tolist()
        tweet_authors = df['tweetAuthor'].tolist()
        handles = df['handle'].tolist()
        geos = df['geo'].tolist()
        created_at_dates = df['createdAt'].tolist()
        reply_counts = df['replyCount'].tolist()
        quote_counts = df['quoteCount'].tolist()
        retweet_counts = df['retweetCount'].tolist()
        like_counts = df['likeCount'].tolist()
        views = df['views'].tolist()
        bookmarks = df['bookmarkCount'].tolist()

        # Vectorize the tweet texts for model input
        text_vector = vectorizer.transform(tweet_texts)
        
        # Use the classifier to predict symptoms for each tweet
        predictions = classifier.predict(text_vector)  # Predictions shape: (num_tweets, num_symptoms)

        # Structure the results for each tweet
        results = []
        for i, tweet in enumerate(tweet_texts):
            symptom_presence = predictions[i]  # Array of 0s and 1s for each symptom
            detected_symptoms = [symptom_labels[j] for j, present in enumerate(symptom_presence) if present == 1]
            
            # Determine if the tweet is related to TB based on symptom presence
            is_tb = int(any(detected_symptoms))

            # Calculate importance score based on engagement and TB presence
            importance_score = (
                0.4 * reply_counts[i] +
                0.3 * retweet_counts[i] +
                0.2 * like_counts[i] +
                0.1 * views[i] +
                (is_tb * 100)  # Boost score significantly if TB-related
            )

            # Combine tweet metadata, analysis results, and importance score
            results.append({
                'tweetText': tweet,
                'tweetURL': tweet_urls[i],
                'tweetAuthor': tweet_authors[i],
                'handle': handles[i],
                'geo': geos[i] if geos[i] else None,  # Ensure geo is null if empty
                'createdAt': created_at_dates[i],
                'is_tb': bool(is_tb),
                'detected_symptoms': detected_symptoms,
                'importance_score': importance_score,
                'replyCount': reply_counts[i],
                'quoteCount': quote_counts[i],
                'retweetCount': retweet_counts[i],
                'likeCount': like_counts[i],
                'views': views[i],
                'bookmarkCount': bookmarks[i]
            })

        # Sort results by importance score in descending order
        results.sort(key=lambda x: x['importance_score'], reverse=True)

        # Log the response data for debugging purposes
        app.logger.debug("Response Data: %s", json.dumps(results))

        # Return the response as JSON
        response = jsonify(results)
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response

    except Exception as e:
        # Log any error that occurs during processing
        app.logger.error(f"Error during analysis: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Use the PORT environment variable for Render compatibility
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
