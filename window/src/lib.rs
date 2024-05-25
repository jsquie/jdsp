pub fn sinc(size: usize, cutoff: f32) -> Vec<f32> {
    (((size as i32 * -1) / 2)..=(size as i32 / 2))
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

pub fn hann(size: usize) -> Vec<f32> {
    (0..size)
        .map(|n| {
            ((std::f32::consts::PI * n as f32) / (size - 1) as f32)
                .sin()
                .powf(2.)
        })
        .collect::<Vec<f32>>()
}

pub fn kaiser(size: usize, beta: f32) -> Vec<f32> {
    let one_over_denom = 1. / zeroth_order_bessel(beta);
    let n_size: u32 = size as u32 - 1;
    let n_recip: f32 = 1. / n_size as f32;

    (0..size)
        .map(|n| {
            let k = (2. * (n as f32) * n_recip) - 1.;
            let arg = (1. - (k * k)).sqrt();
            zeroth_order_bessel(beta * arg) * one_over_denom
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

#[cfg(test)]
mod tests {
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
}
