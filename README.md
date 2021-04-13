# Self-Organizing Clustering Method

This is an implementation of a self-organizing clustering method. 
Apdated from 2016 MAICS paper "Real-time Unsupervised Clustering" by Dr. Gabriel Ferrer.

We presented this work on September 24, and I created this helpful visualization of how the SOC centers its clusters. 

The algorithm:
-Take k data samples for initial centroids
 -For each remaining data sample n
  -Find the distance between n and each k centroids
  -Average the closest centroid with n and replace centroid with this averaged sample
	
The GREEN dots are the original data points.
The BLUE dots represent the centroids from Kmeans.
The RED dots show how the Self-Organizing Clusters iteratively find the best centroids after averaging each image with the closest existing centroid.

![SOC](https://github.com/davidavzP/soc/blob/02d1f5241422299b5907e76a16600e17c3dccb2c/SOC.gif)
