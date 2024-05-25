use crate::oversample::os_filter_constants::*;
use crate::oversample::SampleRole;
use circular_buffer::circular_buffer::{DelayBuffer, SizedCircularConvBuff32, SizedDelayBuffer32};

#[derive(Debug)]
pub struct OversampleStage<const CAP: usize> {
    filter_buff: SizedCircularConvBuff32,
    delay_buff: SizedDelayBuffer32,
    kernel: [f32; CAP],
    pub data: Vec<f32>,
    scratch_buff: Vec<f32>,
    delay_coef: Option<f32>,
}

impl<const CAP: usize> OversampleStage<CAP> {
    const CAPACITY: usize = CAP;

    pub fn new(target_size: usize, role: SampleRole, _kernel_size: usize) -> Self {
        todo!("implement kernel_size as parameter");
        OversampleStage {
            filter_buff: SizedCircularConvBuff32::new(),
            delay_buff: match role {
                SampleRole::UpSampleStage => SizedDelayBuffer32::new(UP_DELAY),
                SampleRole::DownSampleStage => SizedDelayBuffer32::new(DOWN_DELAY),
            },
            kernel: [0.0f32; CAP],
            data: vec![0.0_f32; target_size],
            scratch_buff: match role {
                SampleRole::UpSampleStage => vec![0.0_f32; target_size / 2],
                SampleRole::DownSampleStage => vec![0.0_f32; target_size],
            },
            delay_coef: None,
        }
    }

    #[cold]
    pub fn initialize_kernel(&mut self, num_coefs: usize) {
        let new_kernel = build_filter_coefs(num_coefs);
        self.kernel
            .iter_mut()
            .zip(new_kernel.iter().step_by(2))
            .for_each(|(k, n)| *k = *n);
        self.delay_coef = Some(new_kernel[new_kernel.len() / 2]);
    }

    #[cold]
    pub fn reset(&mut self) {
        // self.filter_buff.reset();
        self.delay_buff.reset();
        self.data.iter_mut().for_each(|x| *x = 0.0);
        self.scratch_buff.iter_mut().for_each(|x| *x = 0.0);
    }

    #[inline]
    pub fn process_up(&mut self, input: &mut [f32]) {
        input.clone_into(&mut self.scratch_buff);
        self.filter_buff.convolve(input, &self.kernel);
        self.delay_buff.delay(&mut self.scratch_buff);

        let mut output = self.data.iter_mut();

        input
            .iter()
            .zip(self.scratch_buff.iter())
            .for_each(|(c, d)| {
                *output.next().unwrap() = *c * 2.0;
                *output.next().unwrap() = *d * 2.0 * self.delay_coef.unwrap();
            });
    }

    #[inline]
    pub fn process_down(&mut self, input: &[f32]) {
        self.data
            .iter_mut()
            .zip(input.iter().step_by(2))
            .for_each(|(a, b)| *a = *b);

        self.filter_buff.convolve(&mut self.data, &self.kernel);

        self.scratch_buff
            .iter_mut()
            .zip(input.iter().skip(1).step_by(2))
            .for_each(|(a, b)| *a = *b * self.delay_coef.unwrap());

        self.delay_buff.delay(&mut self.scratch_buff);

        self.data
            .iter_mut()
            .zip(self.scratch_buff.iter())
            .for_each(|(o, d)| *o = *o + *d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_os_stage() {
        let _buf: &mut [f32] = &mut [0.0; 8];
        let os_stage = OversampleStage::new(8, SampleRole::UpSampleStage);
        assert_eq!(os_stage.data, &[0.0_f32; 8]);

        let _buf_64: &mut [f32] = &mut [0.0; 8];
        let os_stage_64 = OversampleStage::new(8, SampleRole::UpSampleStage);

        assert_eq!(os_stage_64.data, &[0.0_f32; 8]);
    }

    #[test]
    fn test_os_stage_up() {
        let _buf: &mut [f32] = &mut [0.0; 8];
        let mut os_stage = OversampleStage::new(8, SampleRole::UpSampleStage);
        os_stage.initialize_kernel();

        let signal: &mut [f32] = &mut [1., 0., 0., 0.];

        os_stage.process_up(signal);

        let expected_result: &[f32] = &[
            0.00000000e+00,
            0.00000000e+00,
            6.05694498e-07,
            0.00000000e+00,
            -8.55564241e-06,
            0.00000000e+00,
            5.03376904e-05,
            0.00000000e+00,
        ];
        check_results(&os_stage.data, expected_result);
    }

    #[test]
    fn test_os_stage_down() {
        let _buf: &mut [f32] = &mut [0.0; 8];
        let mut os_stage = OversampleStage::new(8, SampleRole::DownSampleStage);

        os_stage.initialize_kernel();

        let mut signal_vec: Vec<f32> = vec![vec![1.], vec![0.; 15]].into_iter().flatten().collect();

        let signal: &mut [f32] = signal_vec.as_mut_slice();

        os_stage.process_down(signal);

        let expected_result: &[f32] = &[
            0.00000000e+00,
            3.02847249e-07,
            -4.27782121e-06,
            2.51688452e-05,
            -9.81020621e-05,
            2.97363887e-04,
            -7.57236871e-04,
            1.69370522e-03,
        ];
        check_results(expected_result, &os_stage.data);
    }

    const ERR_TOL: f32 = 1e-5;

    fn check_results(results: &[f32], expected: &[f32]) {
        results
            .iter()
            .zip(expected.iter())
            .enumerate()
            .for_each(|(idx, (r, e))| {
                assert!(
                    (r - e).abs() < ERR_TOL,
                    "Elements: {} -- r: {}, e: {}",
                    idx,
                    r,
                    e
                );
            })
    }

    #[test]
    fn test_multi_stage_up_small() {
        let _buf_0: &mut [f32] = &mut [0.0; 8];
        let _buf_1: &mut [f32] = &mut [0.0; 16];

        let mut os_stage_0 = OversampleStage::new(8, SampleRole::UpSampleStage);
        let mut os_stage_1 = OversampleStage::new(16, SampleRole::UpSampleStage);

        os_stage_0.initialize_kernel();
        os_stage_1.initialize_kernel();

        let signal: &mut [f32] = &mut [1., 0., 0., 0.];

        os_stage_0.process_up(signal);
        os_stage_1.process_up(&mut os_stage_0.data);

        let expected_result: &[f32] = &[
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            3.66865824e-13,
            0.00000000e+00,
            -5.18210553e-12,
            0.00000000e+00,
            2.53071566e-11,
            0.00000000e+00,
            -4.56407414e-11,
            0.00000000e+00,
            -3.99586764e-11,
            0.00000000e+00,
        ];

        check_results(&os_stage_1.data, expected_result);
    }

    #[test]
    fn test_os_multi_stage_down() {
        let mut os_stage_0 = OversampleStage::new(16, SampleRole::DownSampleStage);
        let mut os_stage_1 = OversampleStage::new(8, SampleRole::DownSampleStage);

        os_stage_0.initialize_kernel();
        os_stage_1.initialize_kernel();

        let mut signal: Vec<f32> = vec![vec![1.], vec![0.; 31]].into_iter().flatten().collect();

        os_stage_0.process_down(&mut signal[..]);
        os_stage_1.process_down(&os_stage_0.data);

        let expected_result: &[f32] = &[
            0.00000000e+00,
            0.00000000e+00,
            -1.29552638e-12,
            -1.14101853e-11,
            8.26681591e-11,
            1.52379713e-10,
            5.39599503e-10,
            7.96325155e-10,
        ];

        check_results(&os_stage_1.data, expected_result);
    }

    const RAND_TEST_DATA: [f32; 64] = [
        2.542913180922093,
        0.5862253930784518,
        -0.3781017981702173,
        0.9719707041773196,
        -0.723438732761163,
        0.46736570485331075,
        1.1230884239093122,
        -1.5254702810737608,
        -1.0369418785020552,
        -0.7551785221976162,
        0.355800385238018,
        -0.5496093998991464,
        1.8590280730593869,
        2.087612808960415,
        -0.46388441828587534,
        -0.3128312459772416,
        1.0149635867592255,
        0.3506127205242301,
        -0.029696502629996187,
        -0.2829972749482435,
        1.5890446250525057,
        -0.008351621752887797,
        -1.1129456636678388,
        -1.5361537403069268,
        -1.041077829067189,
        0.728072050969641,
        -0.8438129596070822,
        0.4402591181465738,
        -0.42971887492521316,
        0.15334135252187214,
        -0.7635461199104077,
        1.762123972569591,
        0.3301145895344221,
        0.2014572416094927,
        0.7764788596949593,
        0.39160350248318776,
        -0.17756425187058367,
        0.27433281620925154,
        0.4080337326257071,
        -1.6414140095086096,
        -1.336345359235497,
        -1.1008971772091236,
        -1.3157063518867227,
        1.0175764232139108,
        0.17440562421604466,
        -1.0313598972790907,
        0.23651759499372582,
        -1.3146833809038507,
        -1.124634252813258,
        1.1031223369325283,
        -0.0004804077318948717,
        0.7880973652232677,
        0.5194246080123744,
        0.34318373072487535,
        1.4884272487515111,
        0.7985053660609663,
        -0.01826882502069841,
        -0.2802137889249463,
        -1.24550136558111,
        0.6171569477185286,
        -0.5509759086738899,
        0.4617242023903233,
        -0.5468219478819403,
        -0.3319779579370651,
    ];

    #[test]
    fn test_big_rand_os_stage_up() {
        let mut os_stage_0 =
            OversampleStage::new(RAND_TEST_DATA.len() * 2, SampleRole::UpSampleStage);
        os_stage_0.initialize_kernel();

        let mut sig = RAND_TEST_DATA.clone();

        os_stage_0.process_up(&mut sig);

        let expected_rand_conv_upsample: &[f32] = &[
            0.00000000e+00,
            0.00000000e+00,
            1.54022852e-06,
            0.00000000e+00,
            -2.14011824e-05,
            0.00000000e+00,
            1.22759827e-04,
            0.00000000e+00,
            -4.65597200e-04,
            0.00000000e+00,
            1.36953447e-03,
            0.00000000e+00,
            -3.37294626e-03,
            0.00000000e+00,
            7.27076077e-03,
            0.00000000e+00,
            -1.41315792e-02,
            0.00000000e+00,
            2.53368422e-02,
            0.00000000e+00,
            -4.27662146e-02,
            0.00000000e+00,
            6.94745742e-02,
            0.00000000e+00,
            -1.11932984e-01,
            0.00000000e+00,
            1.88293174e-01,
            0.00000000e+00,
            -3.71570704e-01,
            0.00000000e+00,
            1.37471584e+00,
            2.54291322e+00,
            2.17622230e+00,
            5.86225401e-01,
            -5.99519818e-01,
            -3.78101804e-01,
            5.99430715e-01,
            9.71970718e-01,
            2.44760301e-01,
            -7.23438743e-01,
            -7.02282454e-01,
            4.67365712e-01,
            1.51482065e+00,
            1.12308844e+00,
            -4.10416501e-01,
            -1.52547030e+00,
            -1.46742631e+00,
            -1.03694189e+00,
            -9.55780169e-01,
            -7.55178533e-01,
            -5.51275106e-02,
            3.55800390e-01,
            -1.23837709e-01,
            -5.49609408e-01,
            2.67498998e-01,
            1.85902810e+00,
            2.68548587e+00,
            2.08761284e+00,
            7.18373446e-01,
            -4.63884425e-01,
            -8.48771780e-01,
            -3.12831250e-01,
            5.97988995e-01,
            1.01496360e+00,
            7.30090084e-01,
            3.50612726e-01,
            2.17627150e-01,
            -2.96965031e-02,
            -4.39040248e-01,
            -2.82997279e-01,
            7.22761023e-01,
            1.58904465e+00,
            1.24437925e+00,
            -8.35162187e-03,
            -9.62331526e-01,
            -1.11294568e+00,
            -1.10934684e+00,
            -1.53615376e+00,
            -1.82142975e+00,
            -1.04107784e+00,
            3.40973917e-01,
            7.28072061e-01,
            -2.04807048e-01,
            -8.43812972e-01,
            -2.25223485e-01,
            4.40259124e-01,
            6.43392725e-02,
            -4.29718881e-01,
            -9.73500203e-02,
            1.53341355e-01,
            -4.72298018e-01,
            -7.63546131e-01,
            3.96581986e-01,
            1.76212400e+00,
            1.58188466e+00,
            3.30114594e-01,
            -2.60878939e-01,
            2.01457244e-01,
            7.39841033e-01,
            7.76478871e-01,
            5.81145068e-01,
            3.91603508e-01,
            1.18211532e-01,
            -1.77564254e-01,
            -1.62966221e-01,
            2.74332820e-01,
            6.80684778e-01,
            4.08033738e-01,
            -6.13981300e-01,
            -1.64141403e+00,
            -1.86519802e+00,
            -1.33634538e+00,
            -8.89895711e-01,
            -1.10089719e+00,
            -1.54812474e+00,
            -1.31570637e+00,
            -1.68312972e-01,
            1.01757644e+00,
            1.15620621e+00,
            1.74405627e-01,
            -8.91458669e-01,
            -1.03135991e+00,
            -3.17903015e-01,
            2.36517598e-01,
            -1.97639294e-01,
            -1.31468340e+00,
            -1.89367207e+00,
            -1.12463427e+00,
        ];

        assert_eq!(&os_stage_0.data.len(), &expected_rand_conv_upsample.len());

        check_results(&os_stage_0.data, expected_rand_conv_upsample);
    }

    #[test]
    fn test_big_rand_os_stage_down() {
        let mut os_stage_0 = OversampleStage::new(32, SampleRole::DownSampleStage);
        os_stage_0.initialize_kernel();

        let mut sig = RAND_TEST_DATA.clone();

        os_stage_0.process_down(&mut sig);

        let expected_rand_downsample: &[f32] = &[
            0.00000000e+00,
            7.70114261e-07,
            -1.09926350e-05,
            6.54005487e-05,
            -2.55546547e-04,
            7.69936590e-04,
            -1.93424016e-03,
            4.24089842e-03,
            -8.36751618e-03,
            1.52049978e-02,
            -2.59674979e-02,
            4.25792076e-02,
            -6.89373185e-02,
            1.15476369e-01,
            -2.22297417e-01,
            7.59723472e-01,
            1.11829073e+00,
            -2.40952203e-01,
            5.67634446e-01,
            -6.53720730e-01,
            -9.46126776e-01,
            7.12938242e-01,
            1.19977700e+00,
            1.38604408e-02,
            3.88947073e-01,
            2.74318556e-01,
            2.98666934e-01,
            -1.58265234e+00,
            9.75286224e-03,
            -1.50350209e-01,
            -1.66944847e-01,
            6.19026387e-01,
        ];

        assert_eq!(os_stage_0.data.len(), expected_rand_downsample.len());

        check_results(&os_stage_0.data, expected_rand_downsample);
    }
}
