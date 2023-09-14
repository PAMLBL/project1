import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

def get_comm_class(polarity : float, threshold : float) -> str:
    """
    For a given polarity score and threshold, classify comments between positive, negative and neutral.

    Parameters:
                - polarity (comment) : Textblob score between -1 (neg) and +1 (pos)
                - threshold : if polarity between [-threshold, threshold], comment is neutral
    Return:
                - class : positive, negative and neutral
    """
    if polarity > threshold:
        return 'positive'
    if polarity < (- threshold):
        return 'negative'
    else:
        return 'neutral' 

def add_comm_class(df : pd.DataFrame, threshold : float) -> pd.DataFrame:
    """
    Add comment class (pos, neg or neutral) to dataframe. Both thread polarity and comm polarity are taken into account.

    Parameters:
                - df : dataframe with all results for a keyword and subreddit, see get_dataframe function in get_data.py
                - threshold : if polarity between [-threshold, threshold], comment is neutral.
    Return:
                - original dataframe + comm_polarity_class column
    """
    df['comm_polarity_class'] = (df['thread_polarity'].apply(lambda x: 1 if x>=0 else -1)*df['comm_polarity']).apply(lambda x: get_comm_class(x, threshold))
    return df


def get_dataset_info(subreddit : str, keyword : str, df : pd.DataFrame) -> pd.DataFrame:
    """
    Create a table showing where data was downloaded and how much was found.

    Parameters:
                - subreddit : forum where the search was done
                - keyword : what was searched
                - df :  dataframe with all results, see get_dataframe function in get_data.py
    Returns:
                - dataframe with [key, value] rows. keys are [subreddit, keyword, number of threads found, number of comments found]
    """
    num_threads_found = df['thread_id'].nunique()
    num_comments_analysed = len(df)
    return pd.DataFrame([['subreddit',subreddit], ['keyword',keyword],
            ['number of threads found', num_threads_found],
            ['number of comments analyzed' ,num_comments_analysed]],columns=['Info','Value']).set_index('Info')


def get_comments_histo_perc(df1 : pd.DataFrame, df2 : pd.DataFrame) -> None:
    """
    Graph percentage of positive, negative and neutral opinions per year for two topics.

    Parameters:
                - df1 : dataframe with all results for a keyword and subreddit, see get_dataframe function in get_data.py
                - df2 : same as df1 but different keyword/subreddit combo
    Return:
                - None, graph directly printed in streamlit
    """
    # Style
    sns.set_palette(sns.color_palette(["#FC0606","#FFFFFF","#43B861"],3))
    sns.set(rc={'axes.facecolor':(0,0,0,0), 'figure.facecolor':(0,0,0,0)})
    
    # Necessary computations
    df1 = add_comm_class(df1,threshold=0.1)
    chart1 = df1.groupby(by=['thread_year','comm_polarity_class']).agg({"comm_id" : "count", "comm_score" : "sum"}).reset_index()
    chart1['comm_score_perc'] = 100 * chart1['comm_score'].apply(lambda x: x if x>= 0 else 0) / chart1.groupby('thread_year')["comm_score"].transform("sum")
    chart1['origin'] = 1

    df2 = add_comm_class(df2,threshold=0.1)
    chart2 = df2.groupby(by=['thread_year','comm_polarity_class']).agg({"comm_id" : "count", "comm_score" : "sum"}).reset_index()
    chart2['comm_score_perc'] = 100 * chart2['comm_score'].apply(lambda x: x if x>= 0 else 0) / chart2.groupby('thread_year')["comm_score"].transform("sum")
    chart2['origin'] = 2

    chart = pd.concat([chart1,chart2],axis=0,ignore_index=True)

    # Actual plotting
    g = sns.catplot(data=chart,col='origin',x='thread_year',y="comm_score_perc",hue="comm_polarity_class",kind="bar",legend=False)
    g.add_legend(title="type of comment")
    g.set_ylabels("Percentage of total upvotes received")
    for ax in g.axes.ravel():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    g.set_xlabels("Year comments where posted")
    g.set_titles("Polarity of comments per year")
    st.pyplot(fig=g)


def get_best_worst_comments(df : pd.DataFrame) -> None:
    """
    Write comment with lowest and highest polarity * subjectivity. 
    If multiples max or min, chooses the one with the lowest index.

    Parameters:
                - df : dataframe with all results for a keyword and subreddit, see get_dataframe function in get_data.py
    Return:
                - None : write directly results with streamlit
    """
    st.write("Worst comment")
    # creating relevant metric
    df["opinion"] = df['comm_polarity'] * df['comm_subjectivity']
    st.caption(df.loc[df['opinion'].idxmin(),'comm_text'])
    st.caption(' in thread ' + df.loc[df['opinion'].idxmin(),'thread_title'])
    st.write("Best comment")
    st.caption(df.loc[df['opinion'].idxmax(),'comm_text'])
    st.caption(' in thread ' + df.loc[df['opinion'].idxmin(),'thread_title'])


def get_comments_histo_abs(df1 : pd.DataFrame, df2 : pd.DataFrame) -> None:
    """
    Graph interest on a topic in a given subreddit over time (year). 
    Interest is number of votes (up or down) distributed per year.

    Parameters:
                - df1 : dataframe with all results for a keyword and subreddit, see get_dataframe function in get_data.py
                - df2 : same as df1 but different keyword/subreddit combo
    Return:
                - None, graph directly in streamlit
    """
    # Style
    sns.set(rc={'axes.facecolor':(0,0,0,0), 'figure.facecolor':(0,0,0,0)})
    
    # Computations
    chart1 = df1.groupby(by=['thread_year']).agg({"comm_id" : "count", "comm_score" : lambda x: x.abs().sum()}).reset_index()
    chart1['origin'] = 1

    chart2 = df2.groupby(by=['thread_year']).agg({"comm_id" : "count", "comm_score" : lambda x: x.abs().sum()}).reset_index()
    chart2['origin'] = 2

    chart = pd.concat([chart1,chart2],axis=0,ignore_index=True)

    # Alternative graph : num of comments instead of number of opinions
    # g = sns.catplot(data=chart,col='origin',x='thread_year',y="comm_id",kind="bar",legend=False)
    # g.set_ylabels("Number of comments")
    # g.set_xlabels("Year comments where posted")
    # g.set_titles("Interest on topic")
    # st.pyplot(fig=g)

    h = sns.catplot(data=chart,col='origin',x='thread_year',y="comm_score",kind="bar",legend=False)
    for ax in h.axes.ravel():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    h.set_titles("Interest on topic")
    h.set_ylabels("Number of opinions expressed")
    h.set_xlabels("Year comments where posted")
    st.pyplot(fig=h)


def get_global_sentiment(df : pd.DataFrame) -> None:
    """
    Compute global sentiment score : polarity * comm score normalized by total comm score -> result between -1 and 1.

    Parameters:
                - df : dataframe with all results for a keyword and subreddit, see get_dataframe function in get_data.py
    Return:
                - None, write directly in streamlit.
    """
    glob_sentiment = (df['comm_polarity'] * df['comm_score'] ).sum()/ (df['comm_score'].apply(lambda x: abs(x)).sum())
    st.header(round(glob_sentiment,2))