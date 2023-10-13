import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import pickle

###########################################################
# # example
# ###########################################################
# model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
# sentences = ['This framework generates embeddings for each input sentence',
#              'Sentences are passed as a list of string.',
#              'The quick brown fox jumps over the lazy dog.']

# embeddings = model.encode(sentences)

# # Store sentences & embeddings on disc
# with open('./data_dump/embeddings.pkl', "wb") as fOut:
#     pickle.dump({'sentences': sentences, 'embeddings': embeddings},
#                 fOut, protocol=pickle.HIGHEST_PROTOCOL)

# # Load sentences & embeddings from disc
# with open('./data_dump/embeddings.pkl', "rb") as fIn:
#     stored_data = pickle.load(fIn)
#     stored_sentences = stored_data['sentences']
#     stored_embeddings = stored_data['embeddings']

###########################################################
# 載入資料
###########################################################
df = pd.read_excel(
    "./data_in/dialogue_dataset_20230914.xlsx", sheet_name='Sheet1')
print(df)
sentences = df['sentence'].tolist()
source_flag = df['source'].tolist()

###########################################################
# 設定模型
###########################################################
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
embeddings = model.encode(sentences)

# Store sentences & embeddings on disc
# file_name = "./data_dump/itrash_embeddings.pkl"
# file_name = "./data_dump/itrash_embeddings_0911.pkl"
file_name = "./data_dump/itrash_embeddings_0914.pkl"

with open(file_name, "wb") as fOut:
    pickle.dump({'sentences': sentences, 'embeddings': embeddings, 'source_flag': source_flag},
                fOut, protocol=pickle.HIGHEST_PROTOCOL)

# Load sentences & embeddings from disc
with open(file_name, "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_sentences = stored_data['sentences']
    stored_embeddings = stored_data['embeddings']
    stored_source = stored_data['source_flag']
