pub trait PushFeature {
    fn push(&self, out: &mut Vec<f32>);
}

pub fn push_feature<T: PushFeature>(v: &T, out: &mut Vec<f32>) {
    v.push(out);
}

impl PushFeature for u64 {
    fn push(&self, out: &mut Vec<f32>) {
        out.push(*self as f32);
    }
}

impl PushFeature for Option<u64> {
    fn push(&self, out: &mut Vec<f32>) {
        out.push(self.map(|v| v as f32).unwrap_or(f32::NAN));
    }
}

impl PushFeature for i64 {
    fn push(&self, out: &mut Vec<f32>) {
        out.push(*self as f32);
    }
}

impl PushFeature for i32 {
    fn push(&self, out: &mut Vec<f32>) {
        out.push(*self as f32);
    }
}

impl PushFeature for Option<i64> {
    fn push(&self, out: &mut Vec<f32>) {
        out.push(self.map(|v| v as f32).unwrap_or(f32::NAN));
    }
}

impl PushFeature for f64 {
    fn push(&self, out: &mut Vec<f32>) {
        out.push(*self as f32);
    }
}

impl PushFeature for Option<f64> {
    fn push(&self, out: &mut Vec<f32>) {
        out.push(self.map(|v| v as f32).unwrap_or(f32::NAN));
    }
}

impl PushFeature for bool {
    fn push(&self, out: &mut Vec<f32>) {
        out.push(if *self { 1.0 } else { 0.0 });
    }
}

impl<T, const N: usize> PushFeature for [T; N]
where
    T: PushFeature,
{
    fn push(&self, out: &mut Vec<f32>) {
        for item in self {
            item.push(out);
        }
    }
}

pub trait FeatureLen {
    const LEN: usize;
}

impl FeatureLen for u64 {
    const LEN: usize = 1;
}

impl FeatureLen for Option<u64> {
    const LEN: usize = 1;
}

impl FeatureLen for i64 {
    const LEN: usize = 1;
}

impl FeatureLen for i32 {
    const LEN: usize = 1;
}

impl FeatureLen for Option<i64> {
    const LEN: usize = 1;
}

impl FeatureLen for f64 {
    const LEN: usize = 1;
}

impl FeatureLen for Option<f64> {
    const LEN: usize = 1;
}

impl FeatureLen for bool {
    const LEN: usize = 1;
}

impl<T, const N: usize> FeatureLen for [T; N]
where
    T: FeatureLen,
{
    const LEN: usize = N * T::LEN;
}