use window::{hann, kaiser, sinc};

pub const TOTAL_FILTER_TAP: usize = 63;
pub const NUM_OS_FILTER_TAPS: usize = 32;
pub const UP_DELAY: usize = NUM_OS_FILTER_TAPS / 2;
pub const DOWN_DELAY: usize = (NUM_OS_FILTER_TAPS / 2) + 1;

pub fn build_filter_coefs(num_taps: usize) -> Vec<f32> {
    let sinc = sinc(num_taps, 0.5);
    let hann = hann(num_taps);
    let kaiser = kaiser(num_taps, 10.0);
    let res = sinc
        .iter()
        .zip(hann.iter())
        .zip(kaiser.iter())
        .map(|((v, h), k)| v * h * k)
        .collect::<Vec<f32>>();
    let sum: f32 = res.iter().sum();
    res.into_iter().map(|v| v / sum).collect::<Vec<f32>>()
}

#[cfg(test)]
mod tests {
    use nih_plug::util::window::hann;

    use super::*;

    #[test]
    fn test_create_filter_kernel() {
        let s = sinc(TOTAL_FILTER_TAP, 0.5);
        let h = hann(TOTAL_FILTER_TAP);
        let k = kaiser(TOTAL_FILTER_TAP, 10.);

        assert_eq!(s.len(), TOTAL_FILTER_TAP);
        assert_eq!(h.len(), TOTAL_FILTER_TAP);
        assert_eq!(k.len(), TOTAL_FILTER_TAP);

        let mut res = s
            .into_iter()
            .zip(h.into_iter())
            .zip(k.into_iter())
            .map(|((sn, hn), kn)| sn * hn * kn)
            .collect::<Vec<f32>>();

        let sum: f32 = res.iter().sum();

        res.iter_mut().for_each(|v| *v /= sum);

        let expected_result = [
            -0.00000000e+00,
            0.00000000e+00,
            3.02847249e-07,
            0.00000000e+00,
            -4.27782121e-06,
            0.00000000e+00,
            2.51688452e-05,
            0.00000000e+00,
            -9.81020621e-05,
            0.00000000e+00,
            2.97363887e-04,
            0.00000000e+00,
            -7.57236871e-04,
            0.00000000e+00,
            1.69370522e-03,
            0.00000000e+00,
            -3.42579205e-03,
            0.00000000e+00,
            6.40195196e-03,
            0.00000000e+00,
            -1.12548285e-02,
            0.00000000e+00,
            1.89561539e-02,
            0.00000000e+00,
            -3.13042338e-02,
            0.00000000e+00,
            5.26963461e-02,
            0.00000000e+00,
            -9.91563379e-02,
            0.00000000e+00,
            3.15929813e-01,
            5.00000007e-01,
            3.15929813e-01,
            0.00000000e+00,
            -9.91563379e-02,
            0.00000000e+00,
            5.26963461e-02,
            0.00000000e+00,
            -3.13042338e-02,
            0.00000000e+00,
            1.89561539e-02,
            0.00000000e+00,
            -1.12548285e-02,
            0.00000000e+00,
            6.40195196e-03,
            0.00000000e+00,
            -3.42579205e-03,
            0.00000000e+00,
            1.69370522e-03,
            0.00000000e+00,
            -7.57236871e-04,
            0.00000000e+00,
            2.97363887e-04,
            0.00000000e+00,
            -9.81020621e-05,
            0.00000000e+00,
            2.51688452e-05,
            0.00000000e+00,
            -4.27782121e-06,
            0.00000000e+00,
            3.02847249e-07,
            0.00000000e+00,
            -0.00000000e+00,
        ];

        assert_eq!(res.len(), TOTAL_FILTER_TAP);

        res.iter()
            .zip(expected_result.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!((a - b).abs() < 1e-6, "a: {}, b: {} at index: {}", a, b, idx)
            });
    }

    #[test]
    fn test_create_filter_with_method() {
        let res = build_filter_coefs(TOTAL_FILTER_TAP);

        dbg!(&res);

        assert_eq!(res.len(), TOTAL_FILTER_TAP);

        let expected_result = [
            -0.00000000e+00,
            3.02847249e-07,
            -4.27782121e-06,
            2.51688452e-05,
            -9.81020621e-05,
            2.97363887e-04,
            -7.57236871e-04,
            1.69370522e-03,
            -3.42579205e-03,
            6.40195196e-03,
            -1.12548285e-02,
            1.89561539e-02,
            -3.13042338e-02,
            5.26963461e-02,
            -9.91563379e-02,
            3.15929813e-01,
            3.15929813e-01,
            -9.91563379e-02,
            5.26963461e-02,
            -3.13042338e-02,
            1.89561539e-02,
            -1.12548285e-02,
            6.40195196e-03,
            -3.42579205e-03,
            1.69370522e-03,
            -7.57236871e-04,
            2.97363887e-04,
            -9.81020621e-05,
            2.51688452e-05,
            -4.27782121e-06,
            3.02847249e-07,
            -0.00000000e+00,
        ];

        res.iter()
            .step_by(2)
            .zip(expected_result.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-6, "a: {}, b: {}", a, b));
    }
}
