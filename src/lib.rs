#![feature(map_first_last)]
#![feature(btree_drain_filter)]
#![feature(option_unwrap_none)]

use std::collections::*;
use std::cmp::{Ordering, min, max};
use rand::thread_rng;
use rand::distributions::{Distribution, Uniform};

#[derive(Copy, Clone)]
pub struct Edge<V: Copy + PartialEq + PartialOrd> {
    distance: V,
    indices: (usize, usize)
}

impl<V: Copy + PartialEq + PartialOrd> Edge<V> {
    pub fn new(distance: V, indices: (usize, usize)) -> Self {
        Edge {distance, indices}
    }

    pub fn contains_index(&self, index: usize) -> bool{
        self.indices.0 == index || self.indices.1 == index
    }

    pub fn eq_indices(&self, i2: &(usize,usize)) -> bool {
        (self.indices.0 == i2.0 || self.indices.0 == i2.1) &&
            (self.indices.1 == i2.0 || self.indices.1 == i2.1)
    }

}

impl<V:  Copy + PartialEq + PartialOrd> PartialEq for Edge<V>{
    fn eq(&self, other: &Self) -> bool {
        self.distance.eq(&other.distance) && self.eq_indices( &other.indices)
    }
}

impl<V:  Copy + PartialEq + PartialOrd> Eq for Edge<V>{}

impl<V:  Copy + PartialEq + PartialOrd> PartialOrd for Edge<V>{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl<V: Copy + PartialEq + PartialOrd> Ord for Edge<V> {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.distance < other.distance{
            return Ordering::Less;
        }else if self.distance > other.distance {
            return Ordering::Greater;
        }
        return Ordering::Equal;

    }

    fn max(self, other: Self) -> Self where
        Self: Sized, {
        if self.distance < other.distance {
            return other;
        }
        return self;
    }

    fn min(self, other: Self) -> Self where
        Self: Sized, {
        if self.distance > other.distance {
            return other;
        }
        return self;
    }
}




/*

setup(max-nodes)
    edges = new red-black-tree
    nodes = array of inputs

lookup(input)
    for each node
        find distance from input to node
    return (node-index, distance) of closest node

train(input)
    insert(input)
    if number of nodes exceeds max-nodes
        edge = edges.remove(smallest edge)
        (n1, n2) = endpoints of edge
        Remove n1 and n2 and their edges
        insert(merged(n1,n2))

insert(image)
    add image to nodes
    for each existing node n
        Create and insert a new edge:
            - First vertex is image
            - Second vertex is n
            - Weight = distance from image to n

merged(n1, n2) -> [MEAN]
    img = (n1.image + n2.image) / 2
    return img

*/


pub struct SOCluster<T: Clone, V: Copy + PartialEq + PartialOrd, D: Fn(&T, &T) -> V, M: Fn(&Vec<T>) -> T>{
    centroids: Vec<T>,
    edges: BTreeSet<Edge<V>>,
    k: usize,
    distance: D,
    mean: M
}

impl<T: Clone, V: Copy + PartialEq + PartialOrd, D: Fn(&T, &T) -> V, M: Fn(&Vec<T>) -> T> SOCluster<T,V,D,M>{
    pub fn new_rnd_k(k: usize, data: &[T], distance: D, mean: M) -> SOCluster<T,V,D,M>{
        socluster_setup(k, data, distance, mean)
    }

    pub fn new(k: usize, data: &[T], distance: D, mean: M) -> SOCluster<T,V,D,M>{
        let mut soc = socluster_setup(k, data, distance, mean);
        soc.train_all(data);
        soc

    }

    pub fn clusters(self) -> Vec<T>{
        self.centroids
    }

    pub fn train_all(&mut self, data: &[T]){
        for v in data{
            self.train(v);
        }
    }

    fn train(&mut self, val: &T){
        self.insert(self.centroids.len(), val);
        if self.centroids.len() > self.k && !self.edges.is_empty(){
            let edge = self.edges.pop_first().unwrap();

            let (e1, e2) = edge.indices;
            let max = max(e1, e2);
            let min = min(e1, e2);
            let (n1, n2) = (self.centroids.remove(e1), self.centroids.remove(e2));

            self.edges.drain_filter(|v| v.contains_index(e1) || v.contains_index(e2));

            self.edges = self.edges.iter().map(|edge| shift_edges(edge, max)).collect();

            self.insert(min, &(self.mean)(&vec![n1, n2]));
        }

    }

    fn insert(&mut self, index: usize, val: &T){
        self.centroids.insert(index, val.clone());
        for (i, v) in self.centroids.iter().enumerate(){
            if i != index {
                self.edges.insert(Edge::new((self.distance)(val, v), (index, i)));
            }
        }
    }

}

fn shift_edges<V: Copy + PartialEq + PartialOrd >(edge: &Edge<V>, max: usize) -> Edge<V>{
    let (e1, e2) = edge.indices;
    let mut new_indices = (e1,e2);
    if e1 > max {
        new_indices.0 -= 1;
    }
    if e2 > max {
        new_indices.1 -= 1;
    }
    Edge{
        distance: edge.distance,
        indices: new_indices
    }
}


fn socluster_setup<T: Clone, V: Copy + PartialEq + PartialOrd , D: Fn(&T, &T) -> V, M: Fn(&Vec<T>) -> T>
(k: usize, data: &[T], distance: D, mean: M) -> SOCluster<T,V,D,M> {
    //pick random k nodes
    let mut nodes: Vec<T> = Vec::new();
    let mut vertices: BTreeSet<Edge<V>> = BTreeSet::new();
    let mut rng = thread_rng();
    let range = Uniform::new(0, data.len());
    let mut taken: HashSet<usize> = HashSet::new();
    for i in 0..k{
        //gets a new index
        let mut index = range.sample(&mut rng);
        while taken.contains(&index) {
            index = range.sample(&mut rng);
        }
        taken.insert(index);
        //grabs the image
        let img = data[index].clone();
        //for each value in nodes find distance and add new vertex
        if !nodes.is_empty() {
            for (ni, v) in nodes.iter().enumerate(){
                let node = Edge::new(distance(&img, v), (i, ni));
                vertices.insert(node);
            }
        }
        nodes.push(img);
    }
    SOCluster{
        centroids: nodes,
        edges: vertices,
        k,
        distance,
        mean
    }

}

#[cfg(test)]
mod tests {
    use crate::SOCluster;

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
    fn test_soc(){
        let data = vec![2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 24.0, 25.0, 26.0, 35.0, 40.0, 45.0];
        let clust = 4;
        let mut soc = SOCluster::new_rnd_k(clust, &data, manhattan, mean);
        println!("centroids: {:?}", soc.centroids);
        soc.train_all(&data);
        println!("centroids after train: {:?}", soc.centroids);
        let soc2 = SOCluster::new(clust, &data, manhattan, mean);
        println!("centroids 2 with train: {:?}", soc2.centroids);
    }
}


