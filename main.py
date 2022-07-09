from operator import imod, index
import pandas as pd
import numpy as np
import Mecab
import neologdn
import unicodedata
from sklearn.feature_extraction.text import TfidVectorizer
import re
import warnings
from sklearn.metrics import accuracy_score
import time
import optuna
import shap
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassfier
warnings.filterwarnings('ignore')

# データ読み込み
cosme_data=pd.read_csv('cosme_data.csv', index_col=0)
cosme_data=cosme_data.reset_index
data_1=cosme_data[cosme_data['参考になった数']>1]
data_0=cosme_data[cosme_data['参考になった数']==0]

# データ前処理
def normalize(text):
    normalized_text = lower_text(text)
    normalized_text = normalize_unicode(normalized_text)
    normalized_text = normalized_number(normalized_text)
    return normalized_text

def lower_text(text):
    return text.lower()

def normalize_unicode(text):
    return unicodedata.normalize('NFKC'. text)

def normalize_number(text):
    return re.sub(r'|d+','0', text)

def makedata(data):
    data['文字数'] =0
    for i in range(len(data)):
        data['文字数'].iloc[i] = len(data['レビュー内容'].iloc[i])

    data['レビュー内容（正規化）']=0
    for i in range(len(data)):
        data['レビュー内容（正規化）'].iloc[i] = normalize(data['レビュー内容'].iloc[i])

    return data

data_1 = makedata(data_1)
data_0 = makedata(data_0)

def calcdata(x,y):
    data_75 = x[x['文字数']<75].sample(len(y[y['文字数']<75]))
    data_100 = x[(x['文字数']>75)&(x[x['文字数']<100])].sample(len(y[(y['文字数']>75)&(y['文字数']<100)]))
    data_150 = x[(x['文字数']>100)&(x[x['文字数']<150])].sample(len(y[(y['文字数']>100)&(y['文字数']<150)]))
    # data_200~500まで作る

    x = pd.concat([data_75,data_100,data_150])
    data = pd.concat([x,y])
    return data

data = calcdata(data_0,data_1)

# データの抽出
plususer = pd.merge(data, user_data, on = '投稿者ID')
plususer = plususer[plususer['投稿件数']<180]
user_0 = plususer[plususer['ラベル']==0]
user_1 = plususer[plususer['ラベル']==1]

user_0 = user_0.sample(len(user_1))
data = pd.concat([user_0, user_1])
data.groupby('ラベル')['文字数'].describe()

# dropwordの作成（ドロップワードの作成）
import MeCab

tagger = MeCab.Tagger()

def tokenize(text):
    stop_words = ['ある','いる','お','この','その','さん','する','なる','れる','られる','の']
    node = tagger.parseToNode(text)

    tokens = []
    while node:
        features = node.feature.split(',')
        if features[0] not in ['助詞','助動詞','記号']:
            token = features[6] if features[6]!='*' else node.surface
            if token not in stop_words:
                tokens.append(token)

        node = node.next

    return tokens

data_tdf = [data['レビュー内容（正規化）']].iloc[i] for i in range(len(data))
tfidf_vec = TfidfVectorizer(tokenizer=tokenize)
tfidf_vec.fit(data_tdf)
tfidf_dense = tfidf_vec.transform(data_tdf).todense()
new_cols = tfidf_vec.get_feature_names(1)
tf_data = pd.DataFrame(tfidf_dense, columns=new_cols)

a = [x for x in new_cols if len(x) ==1]
b = [x for x in new_cols if len(x) ==2]

drop_words1 = new_cols[:580]
drop_words2 = b[:720]
drop_words3 = a

# TF-IDF特徴量の作成
import Mecab
