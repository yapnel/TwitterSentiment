import re
import twint
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import string
import anew


def vader(sid_obj, sentence):
    sentiment_dict = sid_obj.polarity_scores(sentence)
    val = sentiment_dict['compound']
    if val >= 0.05:
        return 'positive',val
    elif val <= - 0.05:
        return 'negative',val
    else:
        return 'neutral',val


def getTweets():
    twint.output.clean_lists()
    c = twint.Config()
    #c.Username = "NatWest_Help"
    c.Search = "bounce back loans"
    c.Hide_output = True
    c.Replies = True
    c.Since = "2020-05-01 00:00:00"
    #c.Until = "2020-05-13 00:01:00"
    #c.Limit = 100
    c.Pandas = True
    c.Store_object = True
    twint.run.Search(c)
    return twint.storage.panda.Tweets_df, twint.output.tweets_list

def createTweetDF(tweets):
    columns = ['conversation_id', 'datestamp', 'datetime', 'hashtags', 'likes_count', 'mentions',
               'replies_count', 'retweet', 'retweet_date', 'retweets_count', 'timestamp', 'timezone', 'tweet', 'username']
    tweetDF = pd.DataFrame(columns=columns)

    for tweet in tweets:
        tweetDF = tweetDF.append(
            {
                'conversation_id': tweet.conversation_id,
                'datestamp': tweet.datestamp,
                'datetime': tweet.datetime,
                'hashtags': tweet.hashtags,
                'likes_count': tweet.likes_count,
                'mentions': tweet.mentions,
                'replies_count': tweet.replies_count,
                'retweet': tweet.retweet,
                'retweet_date': tweet.retweet_date,
                'retweets_count': tweet.retweets_count,
                'timestamp': tweet.timestamp,
                'timezone': tweet.timezone,
                'tweet': tweet.tweet,
                'username': tweet.username
            }, ignore_index=True)

    return tweetDF

def vader_emolex(tweetDF):
    sid_obj = SentimentIntensityAnalyzer()
    new_words = {'debt': -4.0, 'pandemic': -4.0, 'fk': -4.0, 'refused': -4.0}
    sid_obj.lexicon.update(new_words)

    sentimentDF = pd.DataFrame(columns=['sentiment','valence','emotion','score'])
    
    for tweet in tweetDF['tweet']:
        words=tweet.split(' ')        
        keys=emolex.index.intersection(words)
        res=emolex.loc[keys].mean(axis=0).to_frame(name='score').reset_index().rename(columns={'index':'emotion'}).fillna(value=0.000)
        sent, val = vader(sid_obj, tweet)
        dictSent = {
                    'sentiment': sent,
                    'valence': val,
                    'emotion': list(res['emotion']),
                    'score': list(res['score'])
                   }
        sentimentDF = sentimentDF.append(dictSent, ignore_index=True)
    tweetDF=pd.concat([tweetDF, sentimentDF], axis=1)
    
    createBaseExportFile(tweetDF)
    createMentionExportFile(tweetDF)
    createHashtagExportFile(tweetDF)
    
    emotionDF  = tweetDF[['emotion']].copy().explode('emotion')
    scoreDF    = tweetDF[['score']].copy().explode('score')
    concatDF   = pd.concat([emotionDF, scoreDF], axis=1)
    explodedDF = pd.merge(explodedDF, concatDF, how='inner', left_on='ID', right_on=concatDF.index, suffixes=('_x', ''))
    explodedDF.drop(axis=1,columns=['emotion_x','score_x'],inplace=True)
    explodedDF.to_csv('emotion.csv', index=False, quoting=1)

def nrc_vad(tweetDF,nrc):
    sentimentDF = pd.DataFrame(columns=['valence','arousal'])
    
    for tweet in tweetDF['tweet']:
        words=tweet.split(' ')        
        keys=nrc.index.intersection(words)
        res=nrc.loc[keys].mean(axis=0).to_frame(name='score').reset_index().pivot_table(columns='index',values='score',dropna=False,fill_value=0)
        dictSent = {
                    'valence': res.Valence[0],
                    'arousal': res.Arousal[0]
                    }           
        sentimentDF = sentimentDF.append(dictSent, ignore_index=True)

    tweetDF=pd.concat([tweetDF, sentimentDF], axis=1)
    createBaseExportFile(tweetDF)
    createMentionExportFile(tweetDF)
    createHashtagExportFile(tweetDF)

def anew_vad(tweetDF):
    sentimentDF = pd.DataFrame(columns=['valence','arousal'])
    
    for tweet in tweetDF['tweet']:
        sentiment_attributes = anew.sentiment(tweet.split())
        dictSent = {
                    'valence': sentiment_attributes['valence'],
                    'arousal': sentiment_attributes['arousal']
                   }
        sentimentDF = sentimentDF.append(dictSent, ignore_index=True)

    tweetDF=pd.concat([tweetDF, sentimentDF], axis=1)
    createBaseExportFile(tweetDF)
    createMentionExportFile(tweetDF)
    createHashtagExportFile(tweetDF)

def createBaseExportFile(tweetDF):
    tweetDF.to_csv('base.csv',index=False, quoting=1)

def createMentionExportFile(tweetDF):
    explodedDF = tweetDF.explode('mentions')
    explodedDF = explodedDF.reset_index().rename(columns={'index':'ID'})
    explodedDF.to_csv('mention.csv', index=False, quoting=1)

def createHashtagExportFile(tweetDF):
    explodedDF = tweetDF.explode('hashtags')
    explodedDF = explodedDF.reset_index().rename(columns={'index':'ID'})
    explodedDF.to_csv('hashtags.csv', index=False, quoting=1)


def main():
    df, tweets = getTweets()
    tweetDF = createTweetDF(tweets)
    
    #emolex=pd.read_csv('emolex/combine_emolex.csv',index_col='word')  
    #vader_emolex(tweets)
    
    #anew_vad(tweets)

    nrc=pd.read_csv('emolex/NRC-VAD-Lexicon.csv',index_col='Word')
    nrc_vad(tweetDF,nrc)

if __name__ == "__main__":
    main()
