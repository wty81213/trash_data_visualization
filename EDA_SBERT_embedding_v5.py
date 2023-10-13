# !pip install -U sentence-transformers
# 引入所需程式庫項目
import plotly.express as px
from sklearn.manifold import TSNE
import numpy as np
from sentence_transformers import SentenceTransformer, util
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle

# 1. 對話資料 bar plot visualize
# 載入資料(對話資料)
df = pd.read_excel(
    "./data_in/dialogue_dataset_20230914.xlsx", sheet_name='Sheet1')
print(df)
# df.columns = ['problem_id', 'problem_name','sentence', 'source', '是否是出題方提供或確認', '後續調整', '0913調整', '0914調整']

# 1.1 Expore data analysis(problem_large_id)
plt.figure(figsize=(10, 5))
sns.countplot(df.problem_large_id, palette='Spectral')
plt.xlabel('problem_large_id')
# plt.title('problem_large_id Distrbution')
# save the plot as JPG file
plt.savefig("./images/problem_large_id_Distrbution.jpg", dpi=300)

# 1.2 Expore data analysis(Problem_id)
plt.figure(figsize=(10, 5))
sns.countplot(df.problem_id, palette='Spectral')
plt.xlabel('problem_id')
# plt.title('problem_id Distrbution')
# save the plot as JPG file
plt.savefig("./images/Problem_id_Distrbution.jpg", dpi=300)


# 2. 句子嵌入的可視化 embedding visualize
# 引入所需程式庫項目 Sentence Bert
# 載入所需使用之SBERT模型（以官方提供的「'distiluse-base-multilingual-cased-v2'」預訓練模型為例）
# Load sentences & embeddings from pickle
file_name = "./data_dump/itrash_embeddings_0914.pkl"
with open(file_name, "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_sentences = stored_data['sentences']
    stored_embeddings = stored_data['embeddings']
    stored_source = stored_data['source_flag']

# 2.1 句子嵌入的可視化 embedding visualize
# TSNE-句子嵌入視覺化
model = TSNE(perplexity=20, n_components=2,
             init='pca', n_iter=2500, random_state=23)
np.set_printoptions(suppress=True)
X_embedded = model.fit_transform(stored_embeddings)
df_embeddings = pd.DataFrame(X_embedded)
df_embeddings = df_embeddings.rename(columns={0: 'x', 1: 'y'})
df_embeddings = df_embeddings.assign(text=df.sentence.values)
df_embeddings = df_embeddings.assign(label_id=df.problem_id.values)
df_embeddings = df_embeddings.assign(label=df.problem_name.values)
df_embeddings = df_embeddings.assign(source=df.source.values)

# 2.2 Display Embedding
# 篩選資料視覺化資料
df_non = df_embeddings[df_embeddings.source == '非調整']  # 非調整的對話資料
df_change = df_embeddings[df_embeddings.source != '非調整']  # 調整的對話資料
df_all = df_embeddings  # 全部對話資料(含調整)

fig_non = px.scatter(
    df_non, x='x', y='y',
    color='label', labels={'color': 'label'},
    hover_data=['text'])
# title='Label Embedding Visualization')
fig_non.show()

fig_change = px.scatter(
    df_change, x='x', y='y',
    color='label', labels={'color': 'label'},
    hover_data=['text'])
# title='Label Embedding Visualization')
fig_change.show()

fig_all = px.scatter(
    df_all, x='x', y='y',
    color='label', labels={'color': 'label'},
    hover_data=['text'])
# title='Label Embedding Visualization')
fig_all.show()

# 2.3 Expore data analysis
# save the plot as JPG file
if not os.path.exists("images"):
    os.mkdir("images")
fig_non.write_image("images/fig_non_0914.jpeg")
fig_non.write_image("images/fig_non_0914.svg")
fig_non.write_html("images/fig_non_0914.html")
fig_change.write_image("images/fig_change_0914.jpeg")
fig_change.write_image("images/fig_change_0914.svg")
fig_change.write_html("images/fig_change_0914.html")
fig_all.write_image("images/fig_all_0914.jpeg")
fig_all.write_image("images/fig_all_0914.svg")
fig_all.write_html("images/fig_all_0914.html")
