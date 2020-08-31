pub mod alts {
    use std::collections::BTreeSet;
    use crate::{Edge, shift_edge, initial_plus_plus};
    use crate::means::traits::{Means};
    use std::fmt::Debug;
    use std::cmp::{max, min};
    use crate::PartialCmp;


    #[allow(dead_code)]
    pub struct SOCluster2<T: Means + Debug, V: PartialCmp + Into<f64>, D: Fn(&T, &T) -> V> {
        centroids: Vec<T>,
        counts: Vec<usize>,
        edges: BTreeSet<Edge>,
        unused_data: Vec<T>,
        average_edge: f64,
        k: usize,
        distance: D,
        weighted: bool
    }

    impl<T: Means + Debug, V: PartialCmp + Into<f64>, D: Fn(&T, &T) -> V> SOCluster2<T, V, D> {
        fn new(centroids: Vec<T>, edges: BTreeSet<Edge>, average_edge: f64, k: usize, distance: D, weighted: bool) -> SOCluster2<T, V, D> {
            SOCluster2 {
                centroids,
                counts: (0..k).map(|_| 1).collect(),
                edges,
                unused_data: Vec::new(),
                average_edge,
                k,
                distance,
                weighted
            }
        }

        pub fn new_trained_weighted(k: usize, data: &[T], distance: D) -> SOCluster2<T, V, D> {
            let centroids = initial_plus_plus(k, &distance, data);
            let mut soc = socluster_setup2(k, &centroids, distance, true);
            soc.train_all(data);
            soc
        }

        pub fn new_trained_unweighted(k: usize, data: &[T], distance: D) -> SOCluster2<T, V, D> {
            let centroids = initial_plus_plus(k, &distance, data);
            let mut soc = socluster_setup2(k, &centroids, distance, false);
            soc.train_all(data);
            soc
        }

        pub fn move_clusters(self) -> Vec<T> {
            self.centroids
        }

        pub fn copy_clusters(&self) -> Vec<T> {
            self.centroids.clone()
        }

        pub fn copy_extra_values(&self) -> Vec<T> {
            self.unused_data.clone()
        }

        pub fn copy_counts(&self) -> Vec<usize> {
            self.counts.clone()
        }

        pub fn train_all(&mut self, data: &[T]) {
            for v in data {
                self.train(v);
            }
        }

        pub fn train(&mut self, val: &T) {
            if self.centroids.len() == self.k {
                self.insert_last(val);
            }
            if self.centroids.len() > self.k && !self.edges.is_empty() {
                let edge = self.edges.pop_first().unwrap();
                let (e1, e2) = edge.indices;
                assert_ne!(e1, e2);
                let max = max(e1, e2);
                let min = min(e1, e2);
                let (n1, n2) = (self.centroids.remove(max), self.centroids.remove(min));

                let (c1, c2) = (self.counts.remove(max) as f64, self.counts.remove(min) as f64);
                assert!(c1 > 0.0 && c2 > 0.0);

                self.edges.drain_filter(|v| v.contains_index(e1) || v.contains_index(e2));
                self.edges = self.edges.iter().map(|edge| shift_edge(edge, max)).collect();

                let sum = c1 + c2;
                let mut mean;

                if self.weighted {
                    let w1 = c1 / sum;
                    let w2 = c2 / sum;
                    mean = Means::calc_weighted(&n1, w1, w2, &n2);
                } else {
                    mean = Means::calc_means(&vec![n1, n2]);
                }

                self.insert(min, &mean, sum as usize);
            }
        }

        pub fn insert(&mut self, index: usize, val: &T, count: usize) {
            self.centroids.insert(index, val.clone());

            self.counts.insert(index, count);

            for (i, v) in self.centroids.iter().enumerate() {
                if i != index {
                    let dist = (self.distance)(val, v);
                    let mut weight = dist.into();
                    if self.weighted {
                        let max = max(self.counts[index], self.counts[i]) as f64;
                        weight *= max;
                    }
                    self.edges.insert(Edge::new(weight, (index, i)));
                }
            }

            self.average_edge = self.edges.iter().fold(0.0, |sum, edge| sum + edge.weight) / self.edges.len() as f64;
        }


        pub fn insert_last(&mut self, val: &T) {
            let last = self.centroids.len();
            //self.centroids.push(val.clone());
            //self.counts.push(1);

            let mut temp_edges = self.edges.clone();

            let mut too_far = true;
            let mut min_dist = std::f64::MAX;
            let mut min_index = 0;

            for (i, v) in self.centroids.iter().enumerate() {
                let dist = (self.distance)(val, v);

                let mut weight = dist.into();
                if self.weighted {
                    let max = max(1, self.counts[i]) as f64;
                    weight *= max;
                }

                if weight <= self.average_edge {
                    too_far = false;
                }

                if weight < min_dist {
                    min_dist = weight;
                    min_index = i;
                }

                temp_edges.insert(Edge::new(weight, (last, i)));
            }

            if too_far {
                self.unused_data.push(val.clone());
            } else {
                self.centroids.push(val.clone());
                self.counts.push(1);
                self.edges = temp_edges;
                self.average_edge = self.edges.iter().fold(0.0, |sum, edge| sum + edge.weight) / self.edges.len() as f64;
            }
        }
    }

    fn socluster_setup2<T: Means + Debug, V: PartialCmp + Into<f64>, D: Fn(&T, &T) -> V>
    (k: usize, data: &[T], distance: D, weighted: bool) -> SOCluster2<T, V, D> {
        let mut nodes: Vec<T> = Vec::new();
        let mut vertices: BTreeSet<Edge> = BTreeSet::new();
        let mut dist_sum = 0.0;
        let mut totalnum_dist = 0.0;
        for i in 0..k {
            let img = data[i].clone();
            if !nodes.is_empty() {
                for (ni, v) in nodes.iter().enumerate() {
                    let dist = distance(&img, &v);

                    totalnum_dist += 1.0;
                    dist_sum += dist.into();

                    let node = Edge::new(dist.into(), (i, ni));
                    vertices.insert(node);
                }
            }
            nodes.push(img);
        }
        let average_dist = dist_sum / totalnum_dist;
        println!("average dist: {}", average_dist);
        SOCluster2::new(nodes, vertices, average_dist, k, distance, weighted)
    }

    ///****************************************************************
    #[allow(dead_code)]
    pub struct SOClusterF<T: Means + Debug, V: PartialCmp + Into<f64>, D: Fn(&T, &T) -> V> {
        centroids: Vec<T>,
        filtered: Vec<T>,
        counts: Vec<usize>,
        edges: BTreeSet<Edge>,
        f_edges: BTreeSet<Edge>,
        average_edge: f64,
        k: usize,
        distance: D,
        weighted: bool
    }

    impl<T: Means + Debug, V: PartialCmp + Into<f64>, D: Fn(&T, &T) -> V> SOClusterF<T, V, D> {
        fn new(centroids: Vec<T>, edges: BTreeSet<Edge>, average_edge: f64, k: usize, distance: D, weighted: bool) -> SOClusterF<T, V, D> {
            SOClusterF {
                centroids,
                filtered: Vec::new(),
                counts: (0..k).map(|_| 1).collect(),
                edges,
                f_edges: BTreeSet::new(),
                average_edge,
                k,
                distance,
                weighted
            }
        }

        pub fn new_trained_weighted(k: usize, data: &[T], distance: D) -> SOClusterF<T, V, D> {
            let mut socf = socluster_setupf(k, data, distance, true);
            socf.train_all(&data);
            socf
        }

        pub fn new_trained_unweighted(k: usize, data: &[T], distance: D) -> SOClusterF<T, V, D> {
            unimplemented!()
        }

        pub fn move_clusters(self) -> Vec<T> {
            self.centroids
        }

        pub fn copy_clusters(&self) -> Vec<T> {
            self.centroids.clone()
        }
        pub fn copy_counts(&self) -> Vec<usize> {
            self.counts.clone()
        }

        pub fn copy_filtered(&self) -> Vec<T> {
            self.filtered.clone()
        }

        pub fn train_all(&mut self, data: &[T]) {
            for val in data {
                self.filter_val(val);
                if self.filtered.len() >= self.k && !self.f_edges.is_empty() {
                    let val = self.f_edges.first().unwrap();
                    let min_centroid = self.edges.last().unwrap();
                    if val.weight <= min_centroid.weight {
                        self.train_from_filtered();
                    }
                }
            }
        }

        fn train_from_filtered(&mut self) {
            let f_edge = self.f_edges.pop_first().unwrap();
            let (e1, e2) = f_edge.indices;
            assert_ne!(e1, e2);
            let max = max(e1, e2);
            let min = min(e1, e2);
            let (n1, n2) = (self.filtered.remove(max), self.filtered.remove(min));
            self.f_edges.drain_filter(|v| v.contains_index(e1) || v.contains_index(e2));
            self.f_edges = self.f_edges.iter().map(|edge| shift_edge(edge, max)).collect();

            let mean = Means::calc_means(&vec![n1, n2]);

            self.train(&mean);
        }

        fn filter_val(&mut self, val: &T) {
            let last = self.filtered.len();
            self.filtered.push(val.clone());

            for (i, v) in self.filtered.iter().enumerate() {
                if i != last {
                    let dist = (self.distance)(val, v);
                    let mut weight = dist.into();

                    self.f_edges.insert(Edge::new(weight, (last, i)));
                }
            }
        }

        fn train(&mut self, val: &T) {
            if self.centroids.len() == self.k {
                self.insert_last(val);
            }
            if self.centroids.len() > self.k && !self.edges.is_empty() {
                let edge = self.edges.pop_first().unwrap();
                let (e1, e2) = edge.indices;
                assert_ne!(e1, e2);
                let max = max(e1, e2);
                let min = min(e1, e2);
                let (n1, n2) = (self.centroids.remove(max), self.centroids.remove(min));

                let (c1, c2) = (self.counts.remove(max) as f64, self.counts.remove(min) as f64);
                assert!(c1 > 0.0 && c2 > 0.0);

                self.edges.drain_filter(|v| v.contains_index(e1) || v.contains_index(e2));
                self.edges = self.edges.iter().map(|edge| shift_edge(edge, max)).collect();

                let sum = c1 + c2;
                let mut mean;

                if self.weighted {
                    let w1 = c1 / sum;
                    let w2 = c2 / sum;
                    mean = Means::calc_weighted(&n1, w1, w2, &n2);
                } else {
                    mean = Means::calc_means(&vec![n1, n2]);
                }

                self.insert(min, &mean, sum as usize);
            }
        }

        fn insert(&mut self, index: usize, val: &T, count: usize) {
            self.centroids.insert(index, val.clone());

            self.counts.insert(index, count);

            for (i, v) in self.centroids.iter().enumerate() {
                if i != index {
                    let dist = (self.distance)(val, v);
                    let mut weight = dist.into();
                    if self.weighted {
                        let max = max(self.counts[index], self.counts[i]) as f64;
                        weight *= max;
                    }
                    self.edges.insert(Edge::new(weight, (index, i)));
                }
            }
            self.average_edge = self.edges.iter().fold(0.0, |sum, edge| sum + edge.weight) / self.edges.len() as f64;
        }


        fn insert_last(&mut self, val: &T) {
            let last = self.centroids.len();
            self.centroids.push(val.clone());
            self.counts.push(1);

            for (i, v) in self.centroids.iter().enumerate() {
                if i != last {
                    let dist = (self.distance)(val, v);
                    assert_eq!(1, self.counts[last]);
                    let mut weight = dist.into();
                    if self.weighted {
                        let max = max(self.counts[last], self.counts[i]) as f64;
                        weight *= max;
                    }

                    self.edges.insert(Edge::new(weight, (last, i)));
                }
            }
        }
    }

    fn socluster_setupf<T: Means + Debug, V: PartialCmp + Into<f64>, D: Fn(&T, &T) -> V>
    (k: usize, data: &[T], distance: D, weighted: bool) -> SOClusterF<T, V, D> {
        let mut nodes: Vec<T> = Vec::new();
        let mut vertices: BTreeSet<Edge> = BTreeSet::new();
        let mut dist_sum = 0.0;
        let mut totalnum_dist = 0.0;
        for i in 0..k {
            let img = data[i].clone();
            if !nodes.is_empty() {
                for (ni, v) in nodes.iter().enumerate() {
                    let dist = distance(&img, &v);

                    totalnum_dist += 1.0;
                    dist_sum += dist.into();

                    let node = Edge::new(dist.into(), (i, ni));
                    vertices.insert(node);
                }
            }
            nodes.push(img);
        }
        let average_dist = dist_sum / totalnum_dist;
        println!("average dist: {}", average_dist);
        SOClusterF::new(nodes, vertices, average_dist, k, distance, weighted)
    }

    #[cfg(test)]
    mod tests {
        use crate::soc_alternates::alts::{SOCluster2, SOClusterF};

        #[allow(dead_code)]
        fn manhattan(n1: &f64, n2: &f64) -> f64 {
            let mut diff = n1 - n2;
            if diff < 0.0 { diff = -diff; }
            diff
        }

        #[allow(dead_code)]
        fn mean(nums: &Vec<f64>) -> f64 {
            let total: f64 = nums.iter().sum();
            total / (nums.len() as f64)
        }

        #[test]
        fn test_socf() {
            let data = vec![2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 24.0, 25.0, 26.0, 35.0, 40.0, 45.0, 100.0, 1000.0, 10000.0];
            let clust = 4;
            let socf = SOClusterF::new_trained_weighted(clust, &data, manhattan);
            println!("socf: {:?}, filtered len: {}", socf.centroids, socf.filtered.len());
        }

        #[test]
        fn test_soc2() {
            let data = vec![2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 24.0, 25.0, 26.0, 35.0, 40.0, 45.0];
            let clust = 4;
            let soc2 = SOCluster2::new_trained_weighted(clust, &data, manhattan);
        }
    }
}

