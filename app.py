import streamlit as st

import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

import praw

from get_data import is_in_db, get_threads, get_comments, get_thread_processing, get_thread_processing_en, get_dataframe
from analysis import get_dataset_info, get_comments_histo_perc, add_comm_class, get_comm_class, get_best_worst_comments, get_comments_histo_abs, get_global_sentiment

##Decor
st.set_page_config(page_title='Reddit Hivemind',page_icon='🐝',layout='wide')

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                    padding-left: 0.5rem;
                    padding-right: 0.5rem;
                }
        </style>
        """, unsafe_allow_html=True)

## Title and description
st.title("Reddit Hivemind")

st.subheader("What's going on here?")
st.text("""
        Choose two subreddits and two topics (french or english language) and compare opinions.
        """)

## Connecting to db
DB_NAME = None
DB_URL = None

myclient = MongoClient(DB_URL)
mydb = myclient[DB_NAME]

## Connecting to Reddit API
reddit = praw.Reddit(client_id=None, 
                     client_secret=None, 
                     user_agent= None, 
                     username=None, 
                     password=None)

## Setting Reddit API requests limits
MAX_THREADS_PER_TOPIC = 100
MAX_COMMENTS_PER_SUBJECT = 1000

## Downloading data from Reddit API or database
def loading_data(keyword : str, forum : str, lang : str) -> None:
        """
        Checking if request result is already in database. If not, downloading relevant threads and comments
        from reddit API and computing sentiment analysis.

                Parameters:
                                - keyword : topic
                                - forum : subreddit
                                - lang : en or fr
                Return:
                                - None, write result in mongodb if it wasn't already there.
        """
        threads = get_threads(mydb,reddit, forum, MAX_THREADS_PER_TOPIC, keyword, lang = lang)
        theme_col_name = forum + '_'+ keyword
        theme_col = mydb[theme_col_name]
        if len(threads) != 0: 
                # Write threads to db
                if len(threads) > 1:
                        theme_col.insert_many(threads)
                elif len(threads) == 1:
                        theme_col.insert_one(threads[0])

                # Get comments for threads found 
                threads_id = mydb[theme_col_name].distinct('_id')
                for id in threads_id:
                        comms = get_comments(mydb, reddit, forum, keyword, post_id=id, limit=MAX_COMMENTS_PER_SUBJECT)
                        comm_col_name = theme_col_name + '_comments_' + id
                        comm_col = mydb[comm_col_name]
                        if len(comms) > 1:
                                comm_col.insert_many(comms)
                        elif len(comms) == 1:
                                comm_col.insert_one(comms[0])
                        # Processing new comments
                        if len(comms)!= 0:
                                if lang == 'French':
                                        processed_comms = get_thread_processing(mydb, comm_col_name)
                                elif lang == 'English':
                                        processed_comms = get_thread_processing_en(mydb, comm_col_name)
                                proc_comm_col_name = comm_col_name + '_processed'
                                proc_comm_col = mydb[proc_comm_col_name]
                                if len(processed_comms) > 1:
                                        proc_comm_col.insert_many(processed_comms)
                                elif len(processed_comms) == 1:
                                        proc_comm_col.insert_one(processed_comms[0])

# Interface
## Get input parameters
with st.form('input_form'):
        col_a, col_b = st.columns([0.5,0.5])
        with col_a : 
                KEYWORD_1 = st.text_input("First search term")
                FORUM_1 = st.text_input("First subreddit")
                LANG_1 = st.radio('Choose first subreddit language.',["French","English"])

        with col_b : 
                KEYWORD_2 = st.text_input("Second search term")
                FORUM_2 = st.text_input("Second subreddit")
                LANG_2 = st.radio('Choose second subreddit language.',["French","English"])

        bt_download = st.form_submit_button('Go')

## Load and show data
if bt_download:
        with st.spinner('Loading data...'):
                loading_data(KEYWORD_1,FORUM_1,LANG_1)
                df_1 = get_dataframe(mydb, FORUM_1, KEYWORD_1)

                loading_data(KEYWORD_2,FORUM_2,LANG_2)
                df_2 = get_dataframe(mydb, FORUM_2, KEYWORD_2)
        st.success('Done loading data.')

        # Table with number of threads and comments analyzed
        with st.container():
                st.subheader('Data analyzed')
                col_1, col_2 = st.columns([0.5,0.5])
                with col_1:
                        st.table(get_dataset_info(FORUM_1,KEYWORD_1,df_1))
                with col_2:
                        st.table(get_dataset_info(FORUM_2,KEYWORD_2,df_2))

        # Overall sentiment on topic
        with st.container():
                st.subheader('Global sentiment index')
                st.caption('1 is the most positive, -1 the most negative.')
                col_1, col_2 = st.columns([0.5,0.5])
                with col_1:
                        get_global_sentiment(df_1)
                with col_2:
                        get_global_sentiment(df_2)

        # Percentage of positive, negative, neutral opinions by year
        with st.container():
                st.subheader('General sentiment over time')
                get_comments_histo_perc(df_1,df_2)

        # Number of opinions expressed by year
        with st.container():
                st.subheader('Interest on topic over time')
                get_comments_histo_abs(df_1,df_2)

        # Best and worst comments found, in which thread
        with st.container():
                st.subheader('Best and worst comments')
                col_1, col_2 = st.columns([0.5,0.5])
                with col_1:
                        get_best_worst_comments(df_1)
                with col_2:
                        get_best_worst_comments(df_2)