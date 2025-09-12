import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from tqdm import tqdm

df = pd.read_csv("../Data/all_waybill_info_meituan_0322.csv")
df_restaurants = df[["sender_lng", "sender_lat", "da_id"]].drop_duplicates()
df_customers = df[["recipient_lng", "recipient_lat", "da_id"]].drop_duplicates()

R = 6371
lng_coef = 0.767

id_restaurants = df_restaurants["da_id"].copy()
id_customers = df_customers["da_id"].copy()
X_restaurants = np.array(df_restaurants[["sender_lng", "sender_lat"]]) / 1e6
X_customers = np.array(df_customers[["recipient_lng", "recipient_lat"]]) / 1e6
X_restaurants = R * np.radians(X_restaurants)
X_customers = R * np.radians(X_customers)
X_restaurants[:, 0] *= lng_coef
X_customers[:, 0] *= lng_coef

#X_restaurants = X_restaurants[X_restaurants[:, 0] < 14900,:]
#X_customers = X_customers[X_customers[:, 0] < 14900,:]

#thres = 5
#clustering = AgglomerativeClustering(distance_threshold = thres, n_clusters = None)
#labels = clustering.fit_predict(X_restaurants)
#n_clusters = len(set(labels))
## Plot clusters
#plt.figure(figsize=(8, 6))
#for cluster_id in range(n_clusters):
#    plt.scatter(X_restaurants[labels == cluster_id, 0], X_restaurants[labels == cluster_id, 1], label=f"Cluster {cluster_id}", s = 1)
#plt.title(f"Restaurant Clusters\nNum Clusters = {n_clusters}, Distance Threshold = {thres}km")
#plt.xlabel("Longitude")
#plt.ylabel("Latitude")
##plt.legend()
#plt.savefig("restaurant_clusters.png")
#plt.clf()
#plt.close()

clustering = KMeans(n_clusters = 50)
labels = clustering.fit_predict(X_restaurants)
n_clusters = len(set(labels))
# Plot clusters
plt.figure(figsize=(8, 6))
for id in tqdm(set(id_restaurants)):
    for cluster_id in range(n_clusters):
        idx_lst = (labels == cluster_id) * (id_restaurants == id)
        plt.scatter(X_restaurants[idx_lst, 0], X_restaurants[idx_lst, 1], label=f"Cluster {cluster_id}", s = 1)
    plt.title(f"Restaurant Clusters (Num Clusters = {n_clusters})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    #plt.legend()
    plt.savefig(f"Plots/restaurant_clusters_region={id}.png")
    plt.clf()
    plt.close()


clustering = KMeans(n_clusters = 100)
labels = clustering.fit_predict(X_customers)
n_clusters = len(set(labels))
# Plot clusters
plt.figure(figsize=(8, 6))
for id in tqdm(set(id_customers)):
    for cluster_id in range(n_clusters):
        idx_lst = (labels == cluster_id) * (id_customers == id)
        plt.scatter(X_customers[idx_lst, 0], X_customers[idx_lst, 1], label=f"Cluster {cluster_id}", s = 0.1)
    plt.title(f"Customer Clusters (Num Clusters = {n_clusters})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    #plt.legend()
    plt.savefig(f"Plots/customer_clusters_region={id}.png")
    plt.clf()
    plt.close()
