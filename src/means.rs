pub mod traits {
    pub trait Means: Clone + PartialEq {
        fn calc_mean(&self, other: &Self) -> Self;
        fn calc_means(vals: &Vec<Self>) -> Self;
        fn calc_weighted(&self, w1: f64, w2: f64, other: &Self) -> Self;

    }

    impl Means for f64{
        fn calc_mean(&self, other: &f64) -> Self {
            (self + other) / 2.0
        }

        fn calc_means(vals: &Vec<f64>) -> Self {
            let mut sums = 0.0;
            for v in vals{
                sums += v.clone();
            }
            sums / (vals.len() as f64)
        }

        fn calc_weighted(&self, w1: f64, w2: f64, other: &Self) -> Self {
            (w1 * self.clone()) + (w2 * other.clone())
        }
    }

    impl Means for i32{
        fn calc_mean(&self, other: &i32) -> Self {
            (self + other) / 2
        }

        fn calc_means(vals: &Vec<i32>) -> Self {
            let mut sums = 0;
            for v in vals{
                sums += v.clone();
            }
            sums / (vals.len() as i32)
        }

        fn calc_weighted(&self, w1: f64, w2: f64, other: &Self) -> Self {
            let mean = (w1 as i32 * self.clone()) + (w2 as i32 * other.clone());
            if mean == 0 {
                self.calc_mean(other)
            }else {
                mean
            }

        }
    }

    pub trait PartialCmp = Copy + PartialEq + PartialOrd;
}
