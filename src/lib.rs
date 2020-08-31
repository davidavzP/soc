#![feature(map_first_last)]
#![feature(btree_drain_filter)]
#![feature(option_unwrap_none)]
#![feature(associated_type_defaults)]
#![feature(trait_alias)]

pub mod means;
mod soc_alternates;

use std::collections::*;
use std::cmp::{Ordering, min, max};
use crate::means::traits::*;
use rand::thread_rng;
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use std::path::Iter;
use std::borrow::BorrowMut;
use std::fmt::{Debug, Display};


#[derive(Copy, Clone)]
pub struct Edge {
    weight: f64,
    indices: (usize, usize)

}


impl Edge{
    pub fn new(weight: f64, indices: (usize, usize)) -> Self {
        let indices = if indices.0 < indices.1 {indices}else {(indices.1, indices.0)};
        Edge {weight, indices}
    }

    pub fn contains_index(&self, index: usize) -> bool{
        self.indices.0 == index || self.indices.1 == index
    }
}

impl PartialEq for Edge{
    fn eq(&self, other: &Self) -> bool {
        self.cmp(&other) == Ordering::Equal
    }
}

impl Eq for Edge {}

impl PartialOrd for Edge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl Ord for Edge {
    //TODO: Needs to be cleaned up
    fn cmp(&self, other: &Self) -> Ordering {
        if self.weight < other.weight{
            return Ordering::Less;
        }else if self.weight > other.weight {
            return Ordering::Greater;
        }else if self.indices.0 < other.indices.0 {
            return Ordering::Less;
        }else if self.indices.0 > other.indices.0{
            return Ordering::Greater;
        }else if self.indices.1 < other.indices.1 {
            return Ordering::Less;
        }else if self.indices.1 > other.indices.1{
            return Ordering::Greater;
        }
        return Ordering::Equal;

    }

    fn max(self, other: Self) -> Self where
        Self: Sized, {
        if self.weight < other.weight {
            return other;
        }
        return self;
    }

    fn min(self, other: Self) -> Self where
        Self: Sized, {
        if self.weight > other.weight {
            return other;
        }
        return self;
    }
}

#[derive(Copy, Clone, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct Point<N>{
    x: N,
    y: N
}

impl Point<f64>{
    pub fn euclidean_distance(&self, other: &Self) -> f64{
        ((self.x - other.x).powf(2.0) + (self.y - other.y).powf(2.0)).powf(0.5)
    }
}

impl Means for Point<f64>{
    fn calc_mean(&self, other: &Self) -> Self {
        Point {
            x: self.x.calc_mean(&other.x),
            y: self.y.calc_mean(&other.y)
        }
    }

    fn calc_means(vals: &Vec<Self>) -> Self {
        unimplemented!()
    }

    fn calc_weighted(&self, w1: f64, w2: f64, other: &Self) -> Self {
        Point{
            x: self.x.calc_weighted(w1,w2, &other.x),
            y: self.y.calc_weighted(w1,w2, &other.y)
        }
    }
}

//**************************************************************************************************
//ORIGINAL CLUSTERING
//**************************************************************************************************

#[allow(dead_code)]
pub struct SOCluster<T: Means + Debug , V: PartialCmp + Into<f64>, D: Fn(&T, &T) -> V>{
    centroids: Vec<T>,
    counts: Vec<usize>,
    edges: BTreeSet<Edge>,
    k: usize,
    distance: D,
    weighted: bool
}

impl<T: Means + Debug, V: PartialCmp + Into<f64>, D: Fn(&T, &T) -> V> SOCluster<T,V,D>{
    fn new(centroids: Vec<T>, edges: BTreeSet<Edge>, k: usize, distance: D, weighted: bool) -> SOCluster<T,V,D>{
        SOCluster{
            centroids,
            counts: (0..k).map(|_| 1).collect(),
            edges,
            k,
            distance,
            weighted
        }
    }

    pub fn new_untrained(k: usize, data: &[T], distance: D) -> SOCluster<T,V,D>{
        socluster_setup(k, data, distance, true)
    }

    pub fn new_untrained_unweighted(k: usize, data: &[T], distance: D) -> SOCluster<T,V,D>{
        socluster_setup(k, data, distance, false)
    }

    pub fn new_trained(k: usize, data: &[T], distance: D) -> SOCluster<T,V,D>{
        let mut soc = socluster_setup(k, data, distance, true);
        soc.train_all(data);
        soc
    }

    pub fn new_trained_unweighted(k: usize, data: &[T], distance: D) -> SOCluster<T,V,D>{
        let mut soc = socluster_setup(k, data, distance, false);
        soc.train_all(data);
        soc
    }

    pub fn new_trained_plus(k: usize, data: &[T], distance: D) -> SOCluster<T,V,D>{
        let centroids = initial_plus_plus(k, &distance, data);
        let mut soc = socluster_setup(k, &centroids, distance, true);
        soc.train_all(data);
        soc
    }

    #[cfg(test)]
    pub fn k(&self) -> usize { self.centroids.len()}

    #[cfg(test)]
    pub fn classification(&self, sample: &T) -> usize {
        classify(sample, &self.centroids, &self.distance)
    }

    pub fn move_clusters(self) -> Vec<T>{
        self.centroids
    }

    pub fn copy_clusters(&self) -> Vec<T>{
        self.centroids.clone()
    }

    pub fn copy_counts(&self) -> Vec<usize>{
        self.counts.clone()
    }

    pub fn train_all(&mut self, data: &[T]){
        for  v in data{
            self.train(v);
        }
    }

    pub fn train(&mut self, val: &T){
        if self.centroids.len() == self.k {
            self.insert_last(val);
        }
        if self.centroids.len() > self.k && !self.edges.is_empty(){
            let edge = self.edges.pop_first().unwrap();
            let (e1, e2) = edge.indices;
            assert_ne!(e1, e2);
            let max = max(e1, e2);
            let min = min(e1, e2);
            let (n1, n2) = (self.centroids.remove(max), self.centroids.remove(min));

            let (c1, c2) = (self.counts.remove(max) as f64, self.counts.remove(min) as f64);
            assert!(c1 > 0.0 && c2 > 0.0);

            self.edges.drain_filter(|v| v.contains_index(e1) || v.contains_index(e2));
            self.edges = self.edges.iter().map(|edge | shift_edge(edge, max)).collect();

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

    pub fn insert(&mut self, index: usize, val: &T, count: usize){
        self.centroids.insert(index, val.clone());

        self.counts.insert(index, count);

        for (i, v) in self.centroids.iter().enumerate(){
            if i != index {
                let dist = (self.distance)(val, v);
                let mut weight= dist.into();
                if self.weighted {
                    let max = max(self.counts[index], self.counts[i]) as f64;
                    weight *= max;
                }
                self.edges.insert(Edge::new(weight, (index, i)));
            }
        }
    }


    pub fn insert_last(&mut self, val: &T){
        let last = self.centroids.len();
        self.centroids.push(val.clone());
        self.counts.push(1);

        for (i, v) in self.centroids.iter().enumerate(){
            if i != last {
                let dist = (self.distance)(val, v);
                assert_eq!(1, self.counts[last]);
                let mut weight= dist.into();
                if self.weighted{
                    let max = max(self.counts[last], self.counts[i]) as f64;
                    weight *= max;
                }

                self.edges.insert(Edge::new(weight, (last, i)));
            }
        }
    }

}

fn shift_edge(edge: &Edge, max: usize) -> Edge {
    let (e1, e2) = edge.indices;
    let mut new_indices = (e1,e2);
    if e1 > max {
        new_indices.0 -= 1;
    }
    if e2 > max {
        new_indices.1 -= 1;
    }
    Edge{
        weight: edge.weight.clone(),
        indices: new_indices
    }
}


fn socluster_setup<T: Means + Debug, V: PartialCmp + Into<f64>, D: Fn(&T, &T) -> V>
(k: usize, data: &[T], distance: D, weighted: bool) -> SOCluster<T,V,D> {
    let mut nodes: Vec<T> = Vec::new();
    let mut vertices: BTreeSet<Edge> = BTreeSet::new();
    for i in 0..k{
        let img = data[i].clone();
        if !nodes.is_empty() {
            for (ni, v) in nodes.iter().enumerate(){
                let dist = distance(&img, v);
                let node = Edge::new(dist.into(), (i, ni));
                vertices.insert(node);
            }
        }
        nodes.push(img);
    }
    SOCluster::new(nodes, vertices, k, distance, weighted)

}



#[cfg(test)]
fn classify<T: Means + Debug, V: PartialCmp, D: Fn(&T,&T) -> V>(target: &T, means: &Vec<T>, distance: &D) -> usize {
    let distances: Vec<(V,usize)> = (0..means.len())
        .map(|i| (distance(&target, &means[i]).into(), i))
        .collect();
    distances.iter()
        .fold(None, |m:Option<&(V, usize)>, d| m.map_or(Some(d), |m|
            Some(if m.0 < d.0 {m} else {d}))).unwrap().1
}

pub fn initial_plus_plus<T: Clone + PartialEq + Debug, V: Copy + PartialEq + PartialOrd + Into<f64>, D: Fn(&T,&T) -> V>(k: usize, distance: &D, data: &[T]) -> Vec<T> {
    let mut result = Vec::new();
    let mut rng = thread_rng();
    let range = Uniform::new(0, data.len());
    result.push(data[range.sample(&mut rng)].clone());
    while result.len() < k {
        let squared_distances: Vec<f64> = data.iter()
            .map(|datum| 1.0f64 + distance(datum, result.last().unwrap()).into())
            .map(|dist| dist.powf(2.0))
            .collect();
        let dist = WeightedIndex::new(&squared_distances).unwrap();
        result.push(data[dist.sample(&mut rng)].clone());
    }
    result
}


#[cfg(test)]
mod tests {
    use crate::{SOCluster, shift_edge, initial_plus_plus};
    use std::cmp::{max, min};
    use crate::means::traits::Means;

    fn manhattan_32(n1: &i32, n2: &i32) -> i32 {
        let mut diff = n1 - n2;
        if diff < 0 {diff = -diff;}
        diff
    }

    fn mean_32(nums: &Vec<i32>) -> i32 {
        let total: i32 = nums.iter().sum();
        total / (nums.len() as i32)
    }

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
    fn test_soc_insert(){
        let data = vec![2, 3, 4, 10, 11, 12, 24, 25, 26, 35, 40, 45];
        let clust = 4;
        let mut soc = SOCluster::new_untrained(clust, &data, manhattan_32);
        let val1 = 7;
        println!("soc1: {:?}, len: {}, clust: {}", soc.centroids, soc.centroids.len(), clust);
        println!("soc counts: {:?}", soc.counts);
        print!("Edges: [");
        for v in &soc.edges{
            print!(" ({}, [{},{}]) ", v.weight, v.indices.0, v.indices.1);
        }
        print!("]");
        println!();
        soc.insert_last(&val1);
        println!("soc1 after insert: {:?}, len: {}, clust: {}", soc.centroids, soc.centroids.len(), clust);
        println!("soc counts after insert: {:?}", soc.counts);

        print!("Edges after insert: [");
        for v in &soc.edges{
            print!(" ({}, [{},{}]) ", v.weight, v.indices.0, v.indices.1);
        }
        print!("]");
        println!();
        let edge = soc.edges.pop_first().unwrap();
        let dist = edge.weight;
        println!("closest: ({}, {}), dist: {}", edge.indices.0, edge.indices.1, dist);
        let (e1, e2) = edge.indices;
        let max = max(e1, e2);
        let min = min(e1, e2);
        let (n1, n2) = (soc.centroids.remove(max), soc.centroids.remove(min));
        let (c1, c2) = (soc.counts.remove(max) as f64, soc.counts.remove(min) as f64);
        println!("soc1 after remove: {:?}, len: {}, clust: {}", soc.centroids, soc.centroids.len(), clust);
        println!("soc1 counts after remove: {:?}", soc.counts);

        soc.edges.drain_filter(|v| v.contains_index(e1) || v.contains_index(e2));
        print!("Edges after drain: [");
        for v in &soc.edges{
            print!(" ({}, [{},{}]) ", v.weight, v.indices.0, v.indices.1);
        }
        print!("]");
        println!();

        soc.edges = soc.edges.iter().map(|edge| shift_edge(edge, max)).collect();
        print!("Edges after shift: [");
        for v in &soc.edges{
            print!(" ({}, [{},{}]) ", v.weight, v.indices.0, v.indices.1);
        }
        print!("]");
        println!();
        let w1 = c1 / (c1 + c2);
        let w2 = c2 / (c1 + c2);

        let mean = Means::calc_weighted(&n1, w1, w2, &n2);

        //soc.insert(min, &(soc.mean)(&vec![n1, n2]), 0);
        soc.insert(min, &mean, (c1 + c2) as usize);
        print!("Edges after insert: [");
        for v in &soc.edges{
            print!(" ({}, [{},{}]) ", v.weight, v.indices.0, v.indices.1);
        }
        print!("]");
        println!();



        println!("soc1 after shift: {:?}, len: {}, clust: {}", soc.centroids, soc.centroids.len(), clust);
        println!("soc1 counts after shift: {:?}", soc.counts);
    }


    #[test]
    fn test_soc_print() {
        println!("Standard Test: ");
        let data = vec![2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 24.0, 25.0, 26.0, 35.0, 40.0, 45.0];
        let clust = 4;
        let mut soc = SOCluster::new_untrained(clust, &data, manhattan);
        println!("centroids: {:?}", soc.centroids);
        soc.train_all(&data);
        println!("centroids after train: {:?}", soc.centroids);
        let soc2 = SOCluster::new_trained_plus(clust, &data, manhattan);
        println!("centroids 2 with train: {:?}", soc2.centroids);
        print!("Edges: [");
        for v in &soc2.edges{
            print!(" ({}, [{},{}]) ", v.weight, v.indices.0, v.indices.1);
        }
        print!("]");
        println!();
        println!("soc counts: {:?}", soc2.counts);
    }

    #[test]
    fn test_soc(){
        println!("Dr. Ferrer's Test: ");
        //Adapted From "https://github.com/gjf2a/kmeans"
        let candidate_target_means =
            vec![vec![3, 11, 25, 40], vec![2, 3, 11, 32], vec![7, 25, 35, 42],
                 vec![7, 25, 37, 45], vec![7, 25, 40, 40], vec![7, 24, 25, 40], vec![3, 10, 25, 41]];
        let num_target_means = candidate_target_means[0].len();
        let data = vec![2, 3, 4, 10, 11, 12, 24, 25, 26, 35, 40, 45];
        let socluster =
            SOCluster::new_trained(num_target_means, &data, manhattan_32);
        let mut sorted_means = socluster.copy_clusters();
        sorted_means.sort();
        let unsorted_means = socluster.copy_clusters();
        assert_eq!(socluster.k, sorted_means.len());
        assert_eq!(sorted_means.len(), num_target_means);
        println!("sorted_means: {:?}", sorted_means);
        println!("counts: {:?}", socluster.counts);
        println!("candidate_means: {:?}", candidate_target_means);
        for i in 0..sorted_means.len() {
            let target = find_best_match(i, sorted_means[i], &candidate_target_means).unwrap();
            let matching_mean = unsorted_means[socluster.classification(&target)];
            let sorted_index = sorted_means.binary_search(&matching_mean).unwrap();
            assert_eq!(i, sorted_index);
        }
        print!("Edges: [");
        for v in &socluster.edges{
            print!(" ({}, [{},{}]) ", v.weight, v.indices.0, v.indices.1);
        }
        print!("]");
        println!();
    }

    fn find_best_match(i: usize, mean: i32, candidates: &Vec<Vec<i32>>) -> Option<i32> {
        println!("i: {}, mean: {}, candidates: {:?}", i, mean, candidates);

        candidates.iter()
            .find(|target| mean == target[i])
            .map(|v| v[i])
    }
}

