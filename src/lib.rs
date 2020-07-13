#![feature(map_first_last)]
#![feature(btree_drain_filter)]
#![feature(option_unwrap_none)]
#![feature(associated_type_defaults)]
#![feature(trait_alias)]

pub mod means;
use std::collections::*;
use std::cmp::{Ordering, min, max};
use crate::means::traits::*;
use rand::thread_rng;
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use std::path::Iter;
use std::borrow::BorrowMut;


#[derive(Copy, Clone)]
pub struct Edge<V> where V: PartialCmp{
    distance: V,
    indices: (usize, usize),
    weight: f64
}


impl<V: PartialCmp> Edge<V>{
    pub fn new(distance: V, indices: (usize, usize), weight: f64) -> Self {
        let indices = if indices.0 < indices.1 {indices}else {(indices.1, indices.0)};
        Edge {distance, indices, weight}
    }

    pub fn contains_index(&self, index: usize) -> bool{
        self.indices.0 == index || self.indices.1 == index
    }
}

impl<V: PartialCmp> PartialEq for Edge<V>{
    fn eq(&self, other: &Self) -> bool {
        self.cmp(&other) == Ordering::Equal
    }
}

impl<V: PartialCmp> Eq for Edge<V> {}

impl<V: PartialCmp> PartialOrd for Edge<V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl<V: PartialCmp> Ord for Edge<V> {
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

fn f64_max(f1: f64, f2: f64) -> f64 {
    if f1 < f2{
        return f2;
    }
    return f1;
}

fn f64_min(f1: f64, f2: f64) -> f64 {
    if f1 > f2{
        return f2;
    }
    return f1;
}

#[allow(dead_code)]
pub struct SOCluster<T: Means , V: PartialCmp + Into<f64>, D: Fn(&T, &T) -> V, M: Fn(&Vec<T>) -> T>{
    centroids: Vec<T>,
    counts: Vec<usize>,
    edges: BTreeSet<Edge<V>>,
    k: usize,
    distance: D,
    mean: M
}

impl<T: Means  , V: PartialCmp + Into<f64>, D: Fn(&T, &T) -> V, M: Fn(&Vec<T>) -> T> SOCluster<T,V,D,M>{
    pub fn new(centroids: Vec<T>, edges: BTreeSet<Edge<V>>, k: usize, distance: D, mean: M) -> SOCluster<T,V,D,M>{
        SOCluster{
            centroids,
            counts: (0..k).map(|_| 1).collect(),
            edges,
            k,
            distance,
            mean
        }
    }

    pub fn new_untrained(k: usize, data: &[T], distance: D, mean: M) -> SOCluster<T,V,D,M>{
        socluster_setup(k, data, distance, mean)
    }

    pub fn new_trained(k: usize, data: &[T], distance: D, mean: M) -> SOCluster<T,V,D,M>{
        let mut soc = socluster_setup(k, data, distance, mean);
        soc.train_all(data);
        soc
    }

    pub fn new_trained_plus(k: usize, data: &[T], distance: D, mean: M) -> SOCluster<T,V,D,M>{
        let centroids = initial_plus_plus(k, &distance, data);
        let mut soc = socluster_setup(k, &centroids, distance, mean);
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

    pub fn train_all(&mut self, data: &[T]){
        for v in data{
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
            //TODO: GET COUNTS
            let (c1, c2) = (self.counts.remove(max) as f64, self.counts.remove(min) as f64);
            assert_ne!(c1, 0.0);
            assert_ne!(c2, 0.0);

            self.edges.drain_filter(|v| v.contains_index(e1) || v.contains_index(e2));
            self.edges = self.edges.iter().map(|edge | shift_edge(edge, max)).collect();

            //self.insert(min, &n1.calc_mean(&n2));
            //let mean = Means::calc_means(&vec![n1, n2]);

            let w1 = c1 / (c1 + c2);
            let w2 = c1 / (c1 + c2);

            let mean = Means::calc_weighted(&n1, w1, w2, &n2);
            //TODO: TRANSFER COUNTS AND CALCULATE WEIGHTED MEAN
            self.insert(min, &mean, (c1 + c2) as usize);
        }
    }

    pub fn insert(&mut self, index: usize, val: &T, count: usize){
        self.centroids.insert(index, val.clone());
        //TODO: insert correct count
        self.counts.insert(index, count);
        for (i, v) in self.centroids.iter().enumerate(){
            if i != index {
                let dist = (self.distance)(val, v);
                //TODO: DIFFERENT DISTANCE CALCULATION
                let weight = dist.into() * max(self.counts[index], self.counts[i]) as f64;
                self.edges.insert(Edge::new(dist, (index, i), weight));
            }
        }
    }


    pub fn insert_last(&mut self, val: &T){
        let last = self.centroids.len();
        self.centroids.push(val.clone());
        //TODO: PUSH ONE
        self.counts.push(1);
        for (i, v) in self.centroids.iter().enumerate(){
            if i != last {
                let dist = (self.distance)(val, v);
                //TODO: DIFFERNET DISTANCE CALCULATION
                let weight = dist.into() * max(self.counts[last], self.counts[i]) as f64;
                self.edges.insert(Edge::new(dist, (last, i), weight));
            }
        }
    }

}

fn shift_edge<V: PartialCmp>(edge: &Edge<V>, max: usize) -> Edge<V> {
    let (e1, e2) = edge.indices;
    let mut new_indices = (e1,e2);
    if e1 > max {
        new_indices.0 -= 1;
    }
    if e2 > max {
        new_indices.1 -= 1;
    }
    Edge{
        distance: edge.distance.clone(),
        indices: new_indices,
        weight: edge.weight.clone()
    }
}


fn socluster_setup<T: Means, V: PartialCmp + Into<f64>, D: Fn(&T, &T) -> V, M: Fn(&Vec<T>) -> T>
(k: usize, data: &[T], distance: D, mean: M) -> SOCluster<T,V,D,M> {
    let mut nodes: Vec<T> = Vec::new();
    let mut vertices: BTreeSet<Edge<V>> = BTreeSet::new();
    for i in 0..k{
        let img = data[i].clone();
        if !nodes.is_empty() {
            for (ni, v) in nodes.iter().enumerate(){
                let dist = distance(&img, v);
                let node = Edge::new(dist, (i, ni),  dist.into());
                vertices.insert(node);
            }
        }
        nodes.push(img);
    }
    SOCluster::new(nodes, vertices, k, distance, mean)

}

#[cfg(test)]
fn classify<T: Means, V: PartialCmp, D: Fn(&T,&T) -> V>(target: &T, means: &Vec<T>, distance: &D) -> usize {
    let distances: Vec<(V,usize)> = (0..means.len())
        .map(|i| (distance(&target, &means[i]).into(), i))
        .collect();
    distances.iter()
        .fold(None, |m:Option<&(V, usize)>, d| m.map_or(Some(d), |m|
            Some(if m.0 < d.0 {m} else {d}))).unwrap().1
}

fn initial_plus_plus<T: Clone + PartialEq, V: Copy + PartialEq + PartialOrd + Into<f64>, D: Fn(&T,&T) -> V>(k: usize, distance: &D, data: &[T]) -> Vec<T> {
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
        let mut soc = SOCluster::new_untrained(clust, &data, manhattan_32, mean_32);
        let val1 = 7;
        println!("soc1: {:?}, len: {}, clust: {}", soc.centroids, soc.centroids.len(), clust);
        println!("soc counts: {:?}", soc.counts);
        print!("Edges: [");
        for v in &soc.edges{
            print!(" ({}, [{},{}]) ", v.distance, v.indices.0, v.indices.1);
        }
        print!("]");
        println!();
        soc.insert_last(&val1);
        println!("soc1 after insert: {:?}, len: {}, clust: {}", soc.centroids, soc.centroids.len(), clust);
        println!("soc counts after insert: {:?}", soc.counts);

        print!("Edges after insert: [");
        for v in &soc.edges{
            print!(" ({}, [{},{}]) ", v.distance, v.indices.0, v.indices.1);
        }
        print!("]");
        println!();
        let edge = soc.edges.pop_first().unwrap();
        let dist = edge.distance;
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
            print!(" ({}, [{},{}]) ", v.distance, v.indices.0, v.indices.1);
        }
        print!("]");
        println!();

        soc.edges = soc.edges.iter().map(|edge| shift_edge(edge, max)).collect();
        print!("Edges after shift: [");
        for v in &soc.edges{
            print!(" ({}, [{},{}]) ", v.distance, v.indices.0, v.indices.1);
        }
        print!("]");
        println!();
        let w1 = c1 / (c1 + c2);
        let w2 = c1 / (c1 + c2);

        let mean = Means::calc_weighted(&n1, w1, w2, &n2);

        //soc.insert(min, &(soc.mean)(&vec![n1, n2]), 0);
        soc.insert(min, &mean, (c1 + c2) as usize);
        print!("Edges after insert: [");
        for v in &soc.edges{
            print!(" ({}, [{},{}]) ", v.distance, v.indices.0, v.indices.1);
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
        let mut soc = SOCluster::new_untrained(clust, &data, manhattan, mean);
        println!("centroids: {:?}", soc.centroids);
        soc.train_all(&data);
        println!("centroids after train: {:?}", soc.centroids);
        let soc2 = SOCluster::new_trained_plus(clust, &data, manhattan, mean);
        println!("centroids 2 with train: {:?}", soc2.centroids);
        print!("Edges: [");
        for v in &soc2.edges{
            print!(" ({}, [{},{}]) ", v.distance, v.indices.0, v.indices.1);
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
            vec![vec![3, 10, 25, 41], vec![2, 3, 10, 32], vec![7, 25, 35, 42],
                 vec![7, 25, 37, 45], vec![7, 25, 41, 41], vec![7, 24, 25, 41]];
        let num_target_means = candidate_target_means[0].len();
        let data = vec![2, 3, 4, 10, 11, 12, 24, 25, 26, 35, 40, 45];
        let socluster =
            SOCluster::new_trained(num_target_means, &data, manhattan_32, mean_32);
        //
        // let socluster_plus =
        //     SOCluster::new_trained_plus(num_target_means, &data, manhattan_32, mean_32);
        //
        // assert_ne!(socluster.copy_clusters(), socluster_plus.copy_clusters());

        let mut sorted_means = socluster.copy_clusters();
        sorted_means.sort();
        let unsorted_means = socluster.copy_clusters();
        assert_eq!(socluster.k, sorted_means.len());
        assert_eq!(sorted_means.len(), num_target_means);
        println!("sorted_means: {:?}", sorted_means);
        println!("candidate_means: {:?}", candidate_target_means);
        for i in 0..sorted_means.len() {
            let target = find_best_match(i, sorted_means[i], &candidate_target_means).unwrap();
            let matching_mean = unsorted_means[socluster.classification(&target)];
            let sorted_index = sorted_means.binary_search(&matching_mean).unwrap();
            assert_eq!(i, sorted_index);
        }
        print!("Edges: [");
        for v in &socluster.edges{
            print!(" ({}, [{},{}]) ", v.distance, v.indices.0, v.indices.1);
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

