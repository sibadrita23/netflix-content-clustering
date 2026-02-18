#!/usr/bin/env python
# coding: utf-8

# In[34]:


get_ipython().system('pip install wordcloud')


# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity

from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')


# In[36]:


from google.colab import files
uploaded = files.upload()

df = pd.read_csv(list(uploaded.keys())[0])
df.head()


# In[37]:


print("Shape:", df.shape)
df.info()
df.isnull().sum()


# In[38]:


df['director'].fillna('Unknown', inplace=True)
df['cast'].fillna('Unknown', inplace=True)
df['country'].fillna('Unknown', inplace=True)
df['rating'].fillna('Unknown', inplace=True)

# Drop rows with missing description
df.dropna(subset=['description'], inplace=True)

# Convert to lowercase
text_cols = ['title', 'director', 'cast', 'listed_in', 'description']
for col in text_cols:
    df[col] = df[col].str.lower()


# In[39]:


df['combined_features'] = (
df['description'] + " " +
df['listed_in'] + " " +
df['director'] + " " +
df['cast']
)

df[['title','combined_features']].head()


# In[40]:


tfidf = TfidfVectorizer(
stop_words='english',
max_features=5000
)

X = tfidf.fit_transform(df['combined_features'])
print("TF-IDF Shape:", X.shape)


# In[41]:


svd = TruncatedSVD(n_components=100, random_state=42)
X_reduced = svd.fit_transform(X)

print("Reduced Shape:", X_reduced.shape)


# In[42]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []
K_range = range(2, 15)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_reduced)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()


# In[43]:


kmeans = KMeans(n_clusters=8, random_state=42)
df['cluster_kmeans'] = kmeans.fit_predict(X_reduced)

df[['title','cluster_kmeans']].head()


# In[44]:


agg = AgglomerativeClustering(n_clusters=8)
df['cluster_agg'] = agg.fit_predict(X_reduced)


# In[45]:


print("KMeans Silhouette:", silhouette_score(X_reduced, df['cluster_kmeans']))
print("KMeans Calinski:", calinski_harabasz_score(X_reduced, df['cluster_kmeans']))
print("KMeans Davies:", davies_bouldin_score(X_reduced, df['cluster_kmeans']))


# In[46]:


svd_2d = TruncatedSVD(n_components=2, random_state=42)
X_vis = svd_2d.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_vis[:,0], X_vis[:,1], c=df['cluster_kmeans'], cmap='viridis')
plt.title("Cluster Visualization (2D)")
plt.show()


# In[47]:


for i in range(8):
    text = " ".join(df[df['cluster_kmeans']==i]['combined_features'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Cluster {i} WordCloud")
    plt.show()


# In[48]:


cosine_sim = cosine_similarity(X)

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def recommend(title, n=5):
    title = title.lower()

    if title not in indices:
       return "Title not found"

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]

    movie_indices = [i[0] for i in sim_scores]
    return df[['title','listed_in']].iloc[movie_indices]

# Example
recommend("inception")


# In[49]:


get_ipython().system('git init')


# In[60]:


get_ipython().system('mkdir -p data notebooks src models reports')


# In[61]:


get_ipython().system('ls')


# In[62]:


get_ipython().system('ls -R')



# In[63]:


from google.colab import files
files.upload()


# In[64]:


get_ipython().system('ls')



# In[65]:


get_ipython().system('mv "NETFLIX MOVIES AND TV SHOWS CLUSTERING (1) (1) (2) (2).csv" data/netflix_titles.csv')


# In[66]:


get_ipython().system('ls -R')


# In[ ]:


import json
from google.colab import _message
from nbconvert import PythonExporter


nb_dict = _message.blocking_request("get_ipynb", timeout_sec=30)


nb_content = nb_dict["ipynb"]


notebook_path = "notebooks/netflix-content-clustering.ipynb"

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb_content, f)

print("Notebook saved inside notebooks/")


exporter = PythonExporter()
body, _ = exporter.from_filename(notebook_path)

with open("src/netflix_project.py", "w", encoding="utf-8") as f:
    f.write(body)

print("Python script saved inside src/")


# In[ ]:





# In[69]:


get_ipython().system('jupyter nbconvert --to script *.ipynb')


# In[71]:


get_ipython().system('ls /content/drive/MyDrive')



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




