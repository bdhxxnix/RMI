// < begin copyright >
// Copyright Ryan Marcus 2020
//
// See root directory of this project for license terms.
//
// < end copyright >

use crate::models::*;
use log::trace;

const MAX_SEGMENT_ABS_ERROR: f64 = 1.0;

fn simple_lr<T: Iterator<Item = (f64, f64)>>(loc_data: T) -> (f64, f64) {
    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    let mut c = 0.0;
    let mut n: u64 = 0;
    let mut m2 = 0.0;
    let mut data_size = 0;

    for (x, y) in loc_data {
        n += 1;
        let dx = x - mean_x;
        mean_x += dx / (n as f64);
        mean_y += (y - mean_y) / (n as f64);
        c += dx * (y - mean_y);

        let dx2 = x - mean_x;
        m2 += dx * dx2;
        data_size += 1;
    }

    if data_size == 0 {
        return (0.0, 0.0);
    }

    if data_size == 1 {
        return (mean_y, 0.0);
    }

    let cov = c / ((n - 1) as f64);
    let var = m2 / ((n - 1) as f64);

    if var == 0.0 {
        return (mean_y, 0.0);
    }

    let beta: f64 = cov / var;
    let alpha = mean_y - beta * mean_x;

    return (alpha, beta);
}

fn segment_error<T: TrainingKey>(data: &RMITrainingData<T>,
                                 start: usize,
                                 end: usize,
                                 params: (f64, f64)) -> f64 {
    let (alpha, beta) = params;
    data.iter()
        .skip(start)
        .take(end - start)
        .map(|(key, pos)| {
            let prediction = beta.mul_add(key.as_float(), alpha);
            (prediction - pos as f64).abs()
        })
        .fold(0.0, f64::max)
}

fn fit_segment<T: TrainingKey>(data: &RMITrainingData<T>, start: usize, end: usize) -> (f64, f64) {
    simple_lr(data
        .iter()
        .skip(start)
        .take(end - start)
        .map(|(inp, offset)| (inp.as_float(), offset as f64)))
}

fn build_segments<T: TrainingKey>(data: &RMITrainingData<T>) -> (Vec<f64>, Vec<f64>, Vec<u64>) {
    if data.len() == 0 {
        return (vec![0.0], vec![0.0], vec![u64::MAX]);
    }

    let mut intercepts = Vec::new();
    let mut slopes = Vec::new();
    let mut boundaries = Vec::new();

    let mut start = 0;
    while start < data.len() {
        let mut end = start + 1;
        let mut best_params = fit_segment(data, start, end);
        let mut best_err = segment_error(data, start, end, best_params);

        while end < data.len() {
            let candidate_params = fit_segment(data, start, end + 1);
            let candidate_err = segment_error(data, start, end + 1, candidate_params);

            if candidate_err <= MAX_SEGMENT_ABS_ERROR {
                best_params = candidate_params;
                best_err = candidate_err;
                end += 1;
            } else {
                break;
            }
        }

        trace!("PLA segment {}:{} err {}", start, end, best_err);
        intercepts.push(best_params.0);
        slopes.push(best_params.1);
        boundaries.push(data.get_key(end - 1).as_uint());
        start = end;
    }

    return (intercepts, slopes, boundaries);
}

pub struct OptimalPLAModel {
    intercepts: Vec<f64>,
    slopes: Vec<f64>,
    boundaries: Vec<u64>,
}

impl OptimalPLAModel {
    pub fn new<T: TrainingKey>(data: &RMITrainingData<T>) -> OptimalPLAModel {
        let (intercepts, slopes, boundaries) = build_segments(data);
        OptimalPLAModel { intercepts, slopes, boundaries }
    }
}

impl Model for OptimalPLAModel {
    fn predict_to_float(&self, inp: &ModelInput) -> f64 {
        if self.boundaries.is_empty() {
            return 0.0;
        }

        let key = inp.as_int();
        let idx = match self.boundaries.binary_search(&key) {
            Ok(exact) => exact + 1,
            Err(insert) => insert,
        };

        let bounded_idx = idx.min(self.intercepts.len() - 1);
        self.slopes[bounded_idx].mul_add(inp.as_float(), self.intercepts[bounded_idx])
    }

    fn input_type(&self) -> ModelDataType { ModelDataType::Float }
    fn output_type(&self) -> ModelDataType { ModelDataType::Float }

    fn params(&self) -> Vec<ModelParam> {
        vec![
            ModelParam::Int(self.intercepts.len() as u64),
            ModelParam::FloatArray(self.intercepts.clone()),
            ModelParam::FloatArray(self.slopes.clone()),
            ModelParam::IntArray(self.boundaries.clone()),
        ]
    }

    fn code(&self) -> String {
        String::from(
            "inline double optimal_pla(uint64_t length, const double intercepts[], const double slopes[], const uint64_t boundaries[], double inp) {\n    uint64_t idx = bs_upper_bound(boundaries, length, (uint64_t)inp);\n    if (idx >= length) { idx = length - 1; }\n    return std::fma(slopes[idx], inp, intercepts[idx]);\n}"
        )
    }

    fn standard_functions(&self) -> HashSet<StdFunctions> {
        let mut to_r = HashSet::new();
        to_r.insert(StdFunctions::BinarySearch);
        to_r
    }

    fn function_name(&self) -> String { String::from("optimal_pla") }

    fn model_name(&self) -> &'static str { "optimal_pla" }

    fn needs_bounds_check(&self) -> bool { false }

    fn set_to_constant_model(&mut self, constant: u64) -> bool {
        self.intercepts = vec![constant as f64];
        self.slopes = vec![0.0];
        self.boundaries = vec![u64::MAX];
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_linear_run() {
        let mut pts = Vec::new();
        for i in 0..50u64 { pts.push((i, i * 2)); }
        let md = ModelData::IntKeyToIntPos(pts);
        let pla = OptimalPLAModel::new(&md);

        assert_eq!(pla.boundaries.len(), 1);
        assert_eq!(pla.predict_to_int(10.into()), 20);
        assert_eq!(pla.predict_to_int(49.into()), 98);
    }

    #[test]
    fn makes_multiple_segments_when_needed() {
        let md = ModelData::IntKeyToIntPos(vec![
            (0, 0), (1, 1), (2, 2), // first segment
            (100, 120), (101, 121), (102, 122), // second far apart
        ]);

        let pla = OptimalPLAModel::new(&md);
        assert!(pla.boundaries.len() >= 2);

        assert_eq!(pla.predict_to_int(0.into()), 0);
        assert_eq!(pla.predict_to_int(101.into()), 121);
    }

    #[test]
    fn empty_defaults_to_zero() {
        let md: ModelData<u64> = ModelData::empty();
        let pla = OptimalPLAModel::new(&md);
        assert_eq!(pla.predict_to_int(5.into()), 0);
    }
}
