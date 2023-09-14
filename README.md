# project1
Reddit sentiment analysis

This app allows to choose two subreddits and two topics (french or english language) and compare opinions.

To see the app working, please see demo video.
The actual text processing time is not shown, it takes a few minutes if the data was not previously processed. To save API and CPU usage, the app always saves data to a database with pymongo connection and check before downloading again.

To use the app, you need to add your own pymongo connection and Reddit API credentials. Then check requirements and run with streamlit run app.py.
