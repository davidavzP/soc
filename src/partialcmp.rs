pub trait PartialCmp: Copy + PartialEq + PartialOrd{}

impl PartialCmp for f64{}

impl PartialCmp for i32 {}