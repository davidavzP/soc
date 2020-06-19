pub mod traits {
    pub trait Means: Clone + PartialEq {
        type T: Clone + PartialEq;
        fn calc_mean(&self, other: &Self) -> Self;

    }

    impl Means for f64{
        type T = f64;

        fn calc_mean(&self, other: &f64) -> Self {
            (self + other) / 2.0
        }
    }

    impl Means for i32{
        type T = i32;

        fn calc_mean(&self, other: &i32) -> Self {
            (self + other) / 2
        }
    }

    pub trait PartialCmp: Copy + PartialEq + PartialOrd{}

    impl PartialCmp for f64{}

    impl PartialCmp for i32 {}
}

