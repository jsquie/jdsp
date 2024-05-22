pub const TOTAL_FILTER_TAPS: usize = 63;
pub const NUM_OS_FILTER_TAPS: usize = 32;
pub const UP_DELAY: usize = NUM_OS_FILTER_TAPS / 2;
pub const DOWN_DELAY: usize = (NUM_OS_FILTER_TAPS / 2) + 1;

fn sinc(size: i32, cutoff: f32) -> Vec<f32> {
    (((size * -1) / 2)..=(size / 2))
        .map(|i| {
            if i == 0 {
                1.0
            } else {
                let pi_i_cutoff = std::f32::consts::PI * (i as f32 * cutoff);
                pi_i_cutoff.sin() / pi_i_cutoff
            }
        })
        .collect::<Vec<f32>>()
}

fn hann(size: u32) -> Vec<f32> {
    (0..size)
        .map(|n| {
            ((std::f32::consts::PI * n as f32) / (size - 1) as f32)
                .sin()
                .powf(2.)
        })
        .collect::<Vec<f32>>()
}

fn zeroth_order_bessel(val: f32) -> f32 {
    const EPS: f32 = 1e-6;
    let mut bessel_value: f32 = 0.0;
    let mut term: f32 = 1.;
    let mut m: f32 = 0.;

    while term > EPS * bessel_value {
        bessel_value += term;
        m += 1.;
        term *= (val * val) / (4. * m * m);
    }

    bessel_value
}

fn kaiser(size: u32, beta: f32) -> Vec<f32> {
    let one_over_denom = 1. / zeroth_order_bessel(beta);
    let n_size: u32 = size - 1;
    let n_recip: f32 = 1. / n_size as f32;

    (0..size)
        .map(|n| {
            let k = (2. * (n as f32) * n_recip) - 1.;
            let arg = (1. - (k * k)).sqrt();
            zeroth_order_bessel(beta * arg) * one_over_denom
        })
        .collect::<Vec<f32>>()
}

pub fn build_filter_coefs() -> Vec<f32> {
    let sinc = sinc(TOTAL_FILTER_TAPS as i32, 0.5);
    let hann = hann(TOTAL_FILTER_TAPS as u32);
    let kaiser = kaiser(TOTAL_FILTER_TAPS as u32, 10.0);
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
    fn test_create_kaiser() {
        let res = kaiser(10, 1.);
        let expected_result = [
            0.78984831, 0.86980546, 0.93237871, 0.97536552, 0.99724655, 0.99724655, 0.97536552,
            0.93237871, 0.86980546, 0.78984831,
        ];

        res.iter()
            .zip(expected_result.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-6, "a: {}, b: {}", a, b));
    }

    #[test]
    fn test_create_hann() {
        let res = hann(10);
        let expected_result = [
            0., 0.11697778, 0.41317591, 0.75, 0.96984631, 0.96984631, 0.75, 0.41317591, 0.11697778,
            0.,
        ];

        res.iter()
            .zip(expected_result.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-6, "a: {}, b: {}", a, b));
    }

    #[test]
    fn test_create_sinc() {
        let res = sinc(11, 0.5);
        let expected_result = [
            1.27323954e-01,
            -3.89817183e-17,
            -2.12206591e-01,
            3.89817183e-17,
            6.36619772e-01,
            1.00000000e+00,
            6.36619772e-01,
            3.89817183e-17,
            -2.12206591e-01,
            -3.89817183e-17,
            1.27323954e-01,
        ];

        assert_eq!(res.len(), 11);

        res.iter()
            .zip(expected_result.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-6, "a: {}, b: {}", a, b));
    }

    #[test]
    fn test_create_filter_kernel() {
        let s = sinc(TOTAL_FILTER_TAPS as i32, 0.5);
        let h = hann(TOTAL_FILTER_TAPS);
        let k = kaiser(TOTAL_FILTER_TAPS as u32, 10.);

        assert_eq!(s.len(), TOTAL_FILTER_TAPS);
        assert_eq!(h.len(), TOTAL_FILTER_TAPS);
        assert_eq!(k.len(), TOTAL_FILTER_TAPS);

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

        assert_eq!(res.len(), TOTAL_FILTER_TAPS);

        res.iter()
            .zip(expected_result.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!((a - b).abs() < 1e-6, "a: {}, b: {} at index: {}", a, b, idx)
            });
    }

    #[test]
    fn test_create_filter_with_method() {
        let res = build_filter_coefs();

        dbg!(&res);

        assert_eq!(res.len(), TOTAL_FILTER_TAPS);

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
