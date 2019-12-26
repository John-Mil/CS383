"""
John Milmore
=========================================
KMeans clustering on congress voting data
=========================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

FILE_NAME = 'congress_data.csv'


def dist_from_cluster(X, cluster_center):
    """Calculate distance between each row of X and the cluster center.

        Arguments:
            X -- array-like, shape (num_voters, num_votes)
            cluster_center -- array-like, shape(num_votes,)

        Returns:
            array-like, shape(num_voters,)
            euclidean distance between each voter and the cluster center
    """
    return np.sum((X - cluster_center[np.newaxis, :]) ** 2, axis=1) ** (1 / 2)


def avg_diff_dist_from_clusters(X, cluster_center_1, cluster_center_2):
    """Calculate difference in distance between cluster center and each row of X.

        Arguments:
            X -- array-like, shape(num_votes, num_votes)
            cluster_center_1 -- array-like, shape(num_votes,)
            cluster_center_2 -- array-like, shape(num_votes,)

        Returns:
            array-like, shape(num_voters)
            difference between distance from cluster_center_1 and cluster_center_2 for each voter
    """
    return np.mean(abs(dist_from_cluster(X, cluster_center_1) - dist_from_cluster(X, cluster_center_2)))


def main():

    # ########################################################################
    # Get voting data

    votes_df = pd.read_csv(FILE_NAME).drop(columns=['Name', 'State', 'District'])
    X = votes_df.drop(columns='Party').values
    # y = votes_df['Party'].values

    # Transform votes into binary ints
    for x in X:
        yes_mask = (x == 'Yea') | (x == 'Aye')
        x[yes_mask] = 1
        x[~yes_mask] = 0

    # #########################################################################
    # Compute clustering with KMeans

    k_means = KMeans(n_clusters=2, random_state=0)
    k_means.fit(X)

    # #########################################################################
    # Interpret clusters

    clust_assignments = k_means.labels_

    # Democrat cluster
    dem_clust_assignments = clust_assignments[votes_df['Party'] == 'Democrat']
    val, cts = np.unique(dem_clust_assignments, return_counts=True)
    dem_clust = val[np.argmax(cts)]  # Cluster assignment of democrats from the model

    # Republican cluster
    rep_clust_assignments = clust_assignments[votes_df['Party'] == 'Republican']
    val, cts = np.unique(rep_clust_assignments, return_counts=True)
    rep_clust = val[np.argmax(cts)]  # Cluster assignment of republicans from the model

    # #########################################################################
    # Are the clustering results aligned with our data?

    num_correct_dem = sum(dem_clust_assignments == dem_clust)  # 'correct' democrats
    num_dem = len(dem_clust_assignments)
    print(f'Percent of democrats assigned to the democrat cluster: {num_correct_dem / num_dem:.4f}')

    num_correct_rep = sum(rep_clust_assignments == rep_clust)  # 'correct' republicans
    num_rep = len(rep_clust_assignments)
    print(f'Percent of republicans assigned to the republican cluster: {num_correct_rep / num_rep:.4f}')

    num_correct = num_correct_dem + num_correct_rep  # 'correct' voters
    num_voters = len(votes_df)
    print(f"Percent of total voters assigned to the 'correct' cluster: {num_correct / num_voters:.4f}\n")

    # #########################################################################
    # What is happening when the clustering is 'incorrect'?

    # find voters assigned to 'right' cluster
    right_dem_idx = votes_df[(votes_df['Party'] == 'Democrat') & (clust_assignments == dem_clust)].index
    right_rep_idx = votes_df[(votes_df['Party'] == 'Republican') & (clust_assignments == rep_clust)].index
    right_idx = right_dem_idx.append(right_rep_idx)
    X_right = X[right_idx]

    # find voters assigned to 'wrong' cluster
    wrong_idx = votes_df.index.drop(right_idx)
    X_wrong = X[wrong_idx]

    dem_clust_center = k_means.cluster_centers_[dem_clust]
    rep_clust_center = k_means.cluster_centers_[rep_clust]

    # calculate the average difference in distances from the two cluster centers for 'wrong' and 'right' assignments
    wrong_avg_diff_dist = avg_diff_dist_from_clusters(X_wrong, dem_clust_center, rep_clust_center)
    right_avg_diff_dist = avg_diff_dist_from_clusters(X_right, dem_clust_center, rep_clust_center)

    print("Average difference in distances between the two cluster centers for voters assigned to the ...")
    print(f"'incorrect' cluster: {wrong_avg_diff_dist:.4f}")
    print(f"'correct' cluster: {right_avg_diff_dist:.4f}\n")

    # NOTE: We see that 'incorrect' assignments are due to similar distances between cluster centers

    # #########################################################################
    # Visualize clustering

    # Project data to a lower dimensional space
    # PCA: find combination of votes that account for the most variance in the data
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Plot clusters
    fig, ax = plt.subplots(dpi=120)
    sns.set()
    ax.scatter(X_pca[right_dem_idx][:, 0], X_pca[right_dem_idx][:, 1], c='b', alpha=0.75,
               label="consistent dem assignment")
    ax.scatter(X_pca[right_rep_idx][:, 0], X_pca[right_rep_idx][:, 1], c='r', alpha=0.75,
               label="consistent rep assignment")
    ax.scatter(X_pca[wrong_idx][:, 0], X_pca[wrong_idx][:, 1], marker='x', c='k',
               label="inconsistent assignment")
    ax.legend()
    ax.set_xlabel('First Principle Component')
    ax.set_ylabel('Second Principle Component')
    ax.set_title('KMeans Clustering Visualized w/ PCA', fontweight='bold')
    fig.savefig('congress_cluster.png')


if __name__ == '__main__':
    main()
