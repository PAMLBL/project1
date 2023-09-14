import pandas as pd
from datetime import date
import re
from nltk.corpus import stopwords

from textblob import Blobber
from textblob_fr import PatternTagger as PatternTaggerFR
from textblob_fr import PatternAnalyzer as PatternAnalyzerFR

import pymongo

import praw

def is_in_db(db, sub_name : str, keyword : str, data : str, thread_id : str = '') -> bool: 
    """
    Check if reddit data is already downloaded in db.

        Parameters : 
                    - db : mongodb pymongo client
                    - sub_name : subreddit of search
                    - keyword : what was searched
                    - data : threads, comments or processed depending on which type of data to look for in db.
                    - thread_id : optional, only for data = comments or processed, to identify which thread 
                    comments to look for.
        
        Returns:
                    - True or False
    """
    if data == 'threads':
        if (sub_name + '_' + keyword) in db.list_collection_names():
            return True
        else:
            return False
    elif data == 'comments':
        if (sub_name + '_' + keyword + '_comments_' + thread_id) in db.list_collection_names():
            return True
        else:
            return False
    elif data == 'processed':
        if (sub_name + '_' + keyword + '_comments_' + thread_id + '_processed') in db.list_collection_names():
            return True
        else:
            return False
        
# API Downloads
def get_threads(db, reddit, sub_name : str, num_search : int , keyword : str, lang : str) -> list:
    """
    Download threads matching subreddit and keyword search if not already in db.
    
        Parameters:
                    - db : mongodb pymongo client
                    - reddit : praw reddit client
                    - sub_name : subreddit to search
                    - num_search : maximum number of threads returned
                    - keyword : what to search
                    - lang : French or English, to process text of found threads.
        
        Returns:
                    - empty list if search result already in db
                    - list of dictionnaries w/ one entry per thread found :
                        {'_id','title','score','url','text','text_polarity','text_subjectivity',
                        'upvote_ratio','created','created_parsed','num_comments'}
    """
    if not is_in_db(db, sub_name, keyword, data='threads'):
        threads = []
        subreddit = reddit.subreddit(sub_name)
        for submission in subreddit.search(keyword,limit=num_search,sort='top',time_filter="all"):
            if lang == 'French':
                thread_text = submission.title + submission.selftext
                thread_text_clean = comm_clean(thread_text)
                thread_sentiment = get_sentiment(thread_text)
                threads.append({'_id' : str(submission.id) , 
                            'title': str(submission.title),
                                    'score': str(submission.score),
                                    'url': str(submission.url),
                                    'text': str(submission.selftext),
                                    'text_polarity' : thread_sentiment[0],
                                    'text_subjectivity' : thread_sentiment[1],
                                    'subreddit': str(submission.subreddit),
                                    'upvote_ratio': str(submission.upvote_ratio),
                                    'created': str(submission.created),
                                    'created_parsed': str(date.fromtimestamp(submission.created)),
                                    'num_comments': str(submission.num_comments)})
            elif lang == 'English':
                thread_text = submission.title + submission.selftext
                thread_text_clean = comm_clean_en(thread_text)
                thread_sentiment = get_sentiment_en(thread_text)
                threads.append({'_id' : str(submission.id) , 
                            'title': str(submission.title),
                                    'score': str(submission.score),
                                    'url': str(submission.url),
                                    'text': str(submission.selftext),
                                    'text_polarity' : thread_sentiment[0],
                                    'text_subjectivity' : thread_sentiment[1],
                                    'subreddit': str(submission.subreddit),
                                    'upvote_ratio': str(submission.upvote_ratio),
                                    'created': str(submission.created),
                                    'created_parsed': str(date.fromtimestamp(submission.created)),
                                    'num_comments': str(submission.num_comments)})
        return threads
    else:
        return []
    
def get_comments(db, reddit, sub_name : str, keyword : str, post_id : str, limit : int) -> list:
    """
    Download comments for threads matching result of subreddit and keyword search.
    
        Parameters:
                    - db : mongodb pymongo client
                    - reddit : praw reddit client
                    - sub_name : subreddit that was searched (needed for collection naming conventions)
                    - keyword : what was searched in the subreddit (needed for collection naming conventions)
                    - post_id : thread where to download comments
                    - limit : max number of comments to download for thread
        
        Returns:
                    - list of dictionnaries w/ one entry per thread found :
                        {'_id','author','text','created','is_submitter','score'}
    """
    all_comments = []
    submission = reddit.submission(post_id)
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        all_comments.append({'_id': str(comment.id),
                            'author': str(comment.author),
                            'text': str(comment.body),
                            'created': str(comment.created_utc),
                            'is_submitter': str(comment.is_submitter),
                            'score': str(comment.score)
                            })
        if len(all_comments) > limit :
            break
    return all_comments

# Text processing FR
def comm_clean(text : str) -> str:
    """
    Remove url, reddit user references, markdown signs, stop words and punctuation from comments.

        Parameters:
                    - text : text of comment to clean
        Returns:
                    - cleaned text
    """
    url_pattern = r"(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z0-9]{2,}(\.[a-zA-Z0-9]{2,})(\.[a-zA-Z0-9]{2,})?"
    user_pattern = r"\/u\/[A-Za-z0-9]*\s*"
    md_pattern = r"\n"
    punc_pattern = r"[^\w\s]+"
    fr_stopwords = stopwords.words('french')
    fr_stopwords.append("op")
    text1 = re.sub(md_pattern,' ',text)
    text2 = re.sub(url_pattern,'',text1)
    text3 = re.sub(user_pattern,'',text2)
    text4 = re.sub(punc_pattern,' ',text3)
    return ' '.join([word.strip() for word in text4.split(' ') if word.strip().lower() not in fr_stopwords])

def get_sentiment(text : str) -> list:
    """
    Use TextBlobFr to get sentiment on a text.

        Parameters:
                    - text : text of comment to analyze
        Returns:
                    - list with [polarity, subjectivity]
    """
    tb = Blobber(pos_tagger=PatternTaggerFR(), analyzer=PatternAnalyzerFR())
    analysed = tb(text)
    return [analysed.sentiment[0],analysed.sentiment[1]]

def get_thread_processing(db, col_name : str) -> list:
    """
    For all comments in a thread, compute clean version of text plus sentiment analysis.

        Parameters:
                    - db : mongodb pymongo client
                    - col_name : where to find data for this thread. Convention : subreddit_keyword_commments_threadid

        Returns:
                    - list of dict with processed comment data w/ one entry per comment
                    {_id, created, score, polarity, subjectivity, text, clean_text}
    """
    comdf = pd.DataFrame(list(db[col_name].find()))
    comdf["text_clean"] = comdf["text"].apply(comm_clean) 
    comdf["sentiment"] = comdf["text_clean"].apply(get_sentiment)
    comdf['polarity'] = comdf['sentiment'].apply(lambda x: x[0])
    comdf['subjectivity'] = comdf['sentiment'].apply(lambda x: x[1])
    comdf.fillna('',inplace=True)
    processed = []
    for i in range(len(comdf)):
        processed.append({"_id" : comdf.loc[i,'_id'],
                        "created" : comdf.loc[i,'created'],
                        "score" : comdf.loc[i,'score'],
                        "polarity": comdf.loc[i,'polarity'],
                        "subjectivity": comdf.loc[i,'subjectivity'],
                        "text" : comdf.loc[i,"text"],
                        "clean_text": comdf.loc[i,'text_clean']})
    return processed

# Text processing EN
def comm_clean_en(text : str) -> str:
    """
    Remove url, reddit user references, markdown signs, stop words and punctuation from comments.

        Parameters:
                    - text : text of comment to clean
        Returns:
                    - cleaned text
    """
    url_pattern = r"(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z0-9]{2,}(\.[a-zA-Z0-9]{2,})(\.[a-zA-Z0-9]{2,})?"
    user_pattern = r"\/u\/[A-Za-z0-9]*\s*"
    md_pattern = r"\n"
    punc_pattern = r"[^\w\s]+"
    en_stopwords = stopwords.words('english')
    en_stopwords.append("op")
    text1 = re.sub(md_pattern,' ',text)
    text2 = re.sub(url_pattern,'',text1)
    text3 = re.sub(user_pattern,'',text2)
    text4 = re.sub(punc_pattern,' ',text3)
    return ' '.join([word.strip() for word in text4.split(' ') if word.strip().lower() not in en_stopwords])

def get_sentiment_en(text : str) -> list:
    """
    Use TextBlob to get sentiment on a text.

        Parameters:
                    - text : text of comment to analyze
        Returns:
                    - list with [polarity, subjectivity]
    """
    tb = Blobber()
    analysed = tb(text)
    return [analysed.sentiment[0],analysed.sentiment[1]]

def get_thread_processing_en(db, col_name : str) -> list:
    """
    For all comments in a thread, compute clean version of text plus sentiment analysis.

        Parameters:
                    - db : mongodb pymongo client
                    - col_name : where to find data for this thread. Convention : subreddit_keyword_commments_threadid

        Returns:
                    - list of dict with processed comment data w/ one entry per comment
                    {_id, created, score, polarity, subjectivity, text, clean_text}
    """
    comdf = pd.DataFrame(list(db[col_name].find()))
    comdf["text_clean"] = comdf["text"].apply(comm_clean_en) 
    comdf["sentiment"] = comdf["text_clean"].apply(get_sentiment_en)
    comdf['polarity'] = comdf['sentiment'].apply(lambda x: x[0])
    comdf['subjectivity'] = comdf['sentiment'].apply(lambda x: x[1])
    comdf.fillna('',inplace=True)
    processed = []
    for i in range(len(comdf)):
        processed.append({"_id" : comdf.loc[i,'_id'],
                        "created" : comdf.loc[i,'created'],
                        "score" : comdf.loc[i,'score'],
                        "polarity": comdf.loc[i,'polarity'],
                        "subjectivity": comdf.loc[i,'subjectivity'],
                        "text" : comdf.loc[i,"text"],
                        "clean_text": comdf.loc[i,'text_clean']})
    return processed

# Query DB for data
def get_dataframe(db, subreddit : str, keyword : str) -> pd.DataFrame:
    """
    Group data on all comments for a given query (subreddit + search term) in a dataframe.

        Parameters:
                    - db : mongodb pymongo client
                    - subreddit : subreddit where the search was done
                    - keyword : what was searched in that subreddit

        Returns:
                    - Dataframe with columns : 
                    thread_id, thread_title, thread_date, thread_num_comments, thread_upvote_ratio, thread_score,
                    thread_text, thread_polarity, thread_subjectivity, comm_text,
                    comm_clean_text, comm_timestamp, comm_score, comm_id, comm_polarity, comm_subjectivity
                    thread_year, thread_month
    """
    thread_collec_name = subreddit + '_' + keyword
    threads_id = db[thread_collec_name].distinct('_id')
    data = []
    for id in threads_id:
        col = thread_collec_name + '_comments_' + id + '_processed'
        thread_doc = db[thread_collec_name].find_one({'_id': id})
        for doc in db[col].find():
            if len(doc['clean_text']) == 0 or doc['clean_text'] in ['effacé','deleted']:
                pass
            else:
                data.append({'thread_id' : id,
                        'thread_title' : thread_doc['title'],
                        'thread_date' : thread_doc['created_parsed'],
                        'thread_num_comments' : thread_doc['num_comments'],
                        'thread_upvote_ratio' : thread_doc['upvote_ratio'],
                        'thread_score' : thread_doc['score'],
                        'thread_text' : thread_doc['text'],
                        'thread_polarity' : thread_doc['text_polarity'],
                        'thread_subjectivity' : thread_doc['text_subjectivity'],
                        'comm_text' : doc['text'],
                        'comm_clean_text' : doc['clean_text'],
                        'comm_timestamp' : doc['created'],
                        'comm_score' : doc['score'],
                        'comm_id' : doc['_id'],
                        'comm_polarity' : doc['polarity'],
                        'comm_subjectivity' : doc['subjectivity']})
    data_df = pd.DataFrame(data)

    data_df.thread_num_comments = data_df.thread_num_comments.astype('int')
    data_df.thread_upvote_ratio = data_df.thread_upvote_ratio.astype('float')
    data_df.thread_score = data_df.thread_score.astype('int')
    data_df.comm_timestamp = data_df.comm_timestamp.astype('float')
    data_df.comm_score = data_df.comm_score.astype('int')
    data_df.comm_polarity = data_df.comm_polarity.astype('float')
    data_df.comm_subjectivity = data_df.comm_subjectivity.astype('float')
    data_df.thread_polarity = data_df.thread_polarity.astype('float')
    data_df.thread_subjectivity = data_df.thread_subjectivity.astype('float')
    data_df['thread_year'] = data_df['thread_date'].apply(lambda x: pd.to_datetime(x).year)
    data_df['thread_month'] = data_df['thread_date'].apply(lambda x: pd.to_datetime(x).month)
    return data_df