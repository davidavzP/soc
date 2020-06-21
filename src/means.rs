pub mod traits {
    pub trait Means: Clone + PartialEq {
        fn calc_mean(&self, other: &Self) -> Self;
        fn calc_means(vals: &Vec<Self>) -> Self;

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
    }

    pub trait PartialCmp = Copy + PartialEq + PartialOrd;
}
