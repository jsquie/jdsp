mod os_filter_constants;
mod oversample_stage;

use nih_plug::prelude::*;

use crate::oversample::oversample_stage::{OsFactor, OversampleStage, TwoTimes};

use os_filter_constants::*;

const MAX_OVER_SAMPLE_FACTOR: usize = 4;

#[derive(Enum, Debug, Copy, Clone, PartialEq)]
pub enum OversampleFactor {
    #[id = "2x"]
    #[name = "2x"]
    TwoTimes = 1,
    #[id = "4x"]
    #[name = "4x"]
    FourTimes = 2,
    #[id = "8x"]
    #[name = "8x"]
    EightTimes = 3,
    #[id = "16x"]
    #[name = "16x"]
    SixteenTimes = 4,
}

#[derive(Debug)]
pub struct Oversample {
    buff_size: usize,
    factor: OversampleFactor,
    stage_0: OversampleStage<TwoTimes>,
}

impl Oversample {
    pub fn new(initial_factor: OversampleFactor, init_buff_size: usize) -> Self {
        Oversample {
            factor: initial_factor,
            buff_size: init_buff_size,
            stage_0: OversampleStage::new(init_buff_size),
        }
    }

    pub fn get_oversample_factor(&self) -> OversampleFactor {
        self.factor
    }

    pub fn set_oversample_factor(&mut self, new_factor: OversampleFactor) {
        self.factor = new_factor;
    }

    #[cold]
    pub fn reset(&mut self) {
        self.stage_0.reset();
        // self.up_stages
        // .iter_mut()
        // .zip(self.down_stages.iter_mut())
        // .for_each(|(u, d)| {
        // u.reset();
        // d.reset();
        // });
    }

    #[inline]
    pub fn process_up(&mut self, input: &mut [f32], output: &mut [f32]) {
        self.stage_0.process_up(&mut input);

        /*
                let mut last_stage = input;

                self.up_stages
                    .iter_mut()
                    .take(self.factor as usize)
                    .for_each(|s| {
                        s.process_up(last_stage);
                        last_stage = &mut s.data;
                    });

                output
                    .iter_mut()
                    .zip(last_stage.iter())
                    .for_each(|(out, st)| {
                        *out = *st;
                    });
        */
    }

    #[inline]
    pub fn process_down(&mut self, input: &mut [f32], output: &mut [f32]) {
        let mut last_stage = input;

        self.down_stages
            .iter_mut()
            .rev()
            .take(self.factor as usize)
            .rev()
            .for_each(|s| {
                s.process_down(last_stage);
                last_stage = &mut s.data;
            });

        output
            .iter_mut()
            .zip(last_stage.iter())
            .for_each(|(out, st)| {
                *out = *st;
            })
    }
}

/*
#[cfg(test)]
mod tests {

    use core::panic;

    use crate::oversample::*;

    #[test]
    fn test_create_os_2x() {
        let mut os = Oversample::new(OversampleFactor::TwoTimes, 4);
        os.initialize_oversample_stages();

        assert_eq!(os.up_stages.len(), 4);
        assert_eq!(os.down_stages.len(), 4);
        assert_eq!(
            match &os.up_stages[0] {
                Some(stage) => stage.data.len(),
                None => panic!("wrong amount of stages"),
            },
            8
        );
        assert_eq!(
            match &os.down_stages[3] {
                Some(stage) => stage.data.len(),
                None => panic!("Wrong amount of stages"),
            },
            4
        );
        /*
                let os_64 = Oversample::new(OversampleFactor::TwoTimes, 4);
                assert_eq!(os_64.up_stages.len(), 4);
                assert_eq!(os_64.down_stages.len(), 4);
                assert_eq!(os_64.up_stages[0].data.len(), 8);
                assert_eq!(os_64.down_stages[3].data.len(), 4);
        */
    }

    #[test]
    fn test_create_os_4x() {
        let mut os = Oversample::new(OversampleFactor::FourTimes, 4);
        os.initialize_oversample_stages();
        assert_eq!(os.up_stages.len(), 4);
        assert_eq!(os.down_stages.len(), 4);

        assert_eq!(
            match &os.up_stages[0] {
                Some(stage) => stage.data.len(),
                None => panic!("Incorrect first up sample stage"),
            },
            8
        );

        assert_eq!(
            match &os.up_stages[1] {
                Some(stage) => stage.data.len(),
                None => panic!("Wrong number of stages"),
            },
            16
        );

        assert_eq!(
            match &os.down_stages[2] {
                Some(stage) => stage.data.len(),
                None => panic!("Wrong number of stages"),
            },
            8
        );

        assert_eq!(
            match &os.down_stages[3] {
                Some(stage) => stage.data.len(),
                None => panic!("Wrong number of stages"),
            },
            4
        );
    }

    #[test]
    fn test_create_os_8x() {
        let mut os = Oversample::new(OversampleFactor::EightTimes, 4);
        os.initialize_oversample_stages();
        assert_eq!(os.up_stages.len(), 4);
        assert_eq!(os.down_stages.len(), 4);

        assert_eq!(
            match &os.up_stages[0] {
                Some(stage) => stage.data.len(),
                None => panic!("Wrong number of stages"),
            },
            8
        );

        assert_eq!(
            match &os.up_stages[1] {
                Some(stage) => stage.data.len(),
                None => panic!("Wrong number of stages"),
            },
            16
        );

        assert_eq!(
            match &os.up_stages[2] {
                Some(stage) => stage.data.len(),
                None => panic!("Wrong number of stages"),
            },
            32
        );

        assert_eq!(
            match &os.down_stages[1] {
                Some(stage) => stage.data.len(),
                None => panic!("Wrong number of stages"),
            },
            16
        );

        assert_eq!(
            match &os.down_stages[2] {
                Some(stage) => stage.data.len(),
                None => panic!("Wrong number of stages"),
            },
            8
        );

        assert_eq!(
            match &os.down_stages[3] {
                Some(stage) => stage.data.len(),
                None => panic!("Wrong number of stages"),
            },
            4
        );
    }

    #[test]
    fn test_create_os_16x() {
        let mut os = Oversample::new(OversampleFactor::SixteenTimes, 4);
        os.initialize_oversample_stages();

        assert_eq!(os.up_stages.len(), 4);
        assert_eq!(os.down_stages.len(), 4);
        assert_eq!(
            match &os.up_stages[0] {
                Some(stage) => stage.data.len(),
                None => panic!("Wrong number of stages"),
            },
            8
        );
        assert_eq!(
            match &os.up_stages[1] {
                Some(stage) => stage.data.len(),
                None => panic!("Wrong number of stages"),
            },
            16
        );
        assert_eq!(
            match &os.up_stages[2] {
                Some(stage) => stage.data.len(),
                None => panic!("Wrong number of stages"),
            },
            32
        );
        assert_eq!(
            match &os.up_stages[3] {
                Some(stage) => stage.data.len(),
                None => panic!("Wrong number of stages"),
            },
            64
        );
        assert_eq!(
            match &os.down_stages[0] {
                Some(stage) => stage.data.len(),
                None => panic!("Wrong number of stages"),
            },
            32
        );
        assert_eq!(
            match &os.down_stages[1] {
                Some(stage) => stage.data.len(),
                None => panic!("Wrong number of stages"),
            },
            16
        );
        assert_eq!(
            match &os.down_stages[2] {
                Some(stage) => stage.data.len(),
                None => panic!("Wrong number of stages"),
            },
            8
        );
        assert_eq!(
            match &os.down_stages[3] {
                Some(stage) => stage.data.len(),
                None => panic!("Wrong number of stages"),
            },
            4
        );
        /*
                let os_64 = Oversample::new(OversampleFactor::SixteenTimes, 4);
                assert_eq!(os_64.up_stages.len(), 4);
                assert_eq!(os_64.down_stages.len(), 4);
                assert_eq!(os_64.up_stages[0].data.len(), 8);
                assert_eq!(os_64.up_stages[1].data.len(), 16);
                assert_eq!(os_64.up_stages[2].data.len(), 32);
                assert_eq!(os_64.up_stages[3].data.len(), 64);
                assert_eq!(os_64.down_stages[0].data.len(), 32);
                assert_eq!(os_64.down_stages[1].data.len(), 16);
                assert_eq!(os_64.down_stages[2].data.len(), 8);
                assert_eq!(os_64.down_stages[3].data.len(), 4);
        */
    }

    const ERR_TOL: f32 = 1e-5;

    fn check_results(result: &[f32], expected: &[f32]) {
        result
            .iter()
            .zip(expected.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a - b).abs() < ERR_TOL,
                    "Wrong at index: {} -- result: {} expected: {}",
                    idx,
                    a,
                    b
                );
            })
    }

    #[test]
    fn test_small_up_sample_2x() {
        let mut os = Oversample::new(OversampleFactor::TwoTimes, 4);
        os.initialize_oversample_stages();

        let sig: &mut [f32] = &mut [1., 0., 0., 0.];

        let result: &mut [f32] = &mut [0.0; 8];
        os.process_up(sig, result);

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

        check_results(result, expected_result);
    }

    #[test]
    fn test_small_up_sample_4x() {
        let mut os = Oversample::new(OversampleFactor::FourTimes, 4);
        os.initialize_oversample_stages();
        let sig: &mut [f32] = &mut [1., 0., 0., 0.];

        const E_RESULT: &[f32] = &[
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            1.21283720e-14,
            0.00000000e+00,
            -1.48023410e-13,
            0.00000000e+00,
            6.42866764e-13,
            0.00000000e+00,
            -1.07393190e-12,
            0.00000000e+00,
            -5.38648737e-13,
            0.00000000e+00,
        ];

        let result: &mut [f32] = &mut [0.0; E_RESULT.len()];
        os.process_up(sig, result);
        check_results(result, E_RESULT);
    }

    #[test]
    fn test_small_up_sample_8x() {
        let mut os = Oversample::new(OversampleFactor::EightTimes, 4);
        os.initialize_oversample_stages();

        let sig: &mut [f32] = &mut [1., 0., 0., 0.];
        let result: &mut [f32] = &mut [0.0; 32];

        os.process_up(sig, result);

        let expected_result: &[f32] = &[
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            1.33568412e-21,
            0.00000000e+00,
            -1.63016536e-20,
            0.00000000e+00,
            7.07982020e-20,
            0.00000000e+00,
            -1.18270926e-19,
            0.00000000e+00,
            -7.56224401e-20,
            0.00000000e+00,
            7.48833181e-19,
            0.00000000e+00,
            -1.74266397e-18,
            0.00000000e+00,
            2.32105594e-18,
            0.00000000e+00,
            -1.24340739e-18,
            0.00000000e+00,
        ];

        check_results(result, expected_result);
    }

    #[test]
    fn test_small_up_sample_16x() {
        let mut os = Oversample::new(OversampleFactor::SixteenTimes, 4);
        os.initialize_oversample_stages();
        let sig: &mut [f32] = &mut [1., 0., 0., 0.];

        let result: &mut [f32] = &mut [0.0; 64];
        os.process_up(sig, result);

        let expected_result: &[f32] = &[
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            0.00000000e+00,
            1.47097407e-28,
            0.00000000e+00,
            -1.79528298e-27,
            0.00000000e+00,
            7.79692727e-27,
            0.00000000e+00,
            -1.30250456e-26,
            0.00000000e+00,
            -8.32821524e-27,
            0.00000000e+00,
            8.24681656e-26,
            0.00000000e+00,
            -1.91917646e-25,
            0.00000000e+00,
            2.55615310e-25,
            0.00000000e+00,
            -1.38730356e-25,
            0.00000000e+00,
            -2.32264440e-25,
            0.00000000e+00,
            6.95180122e-25,
            0.00000000e+00,
            -1.01314568e-24,
            0.00000000e+00,
            1.24101812e-24,
            0.00000000e+00,
            -1.53628649e-24,
            0.00000000e+00,
            1.70225760e-24,
            0.00000000e+00,
            -1.44474565e-24,
            0.00000000e+00,
            1.65800609e-24,
            0.00000000e+00,
        ];

        assert_eq!(result.len(), expected_result.len());

        for (r, e) in result.iter().zip(expected_result.iter()) {
            assert!(
                (r - e).abs() < 1e-7,
                "Assertion failed: res: {}, expected: {}",
                r,
                e
            )
        }
    }

    #[test]
    fn down_sample_2x() {
        let mut os = Oversample::new(OversampleFactor::TwoTimes, 4);
        os.initialize_oversample_stages();
        let sig_vec = &mut vec![vec![1.], vec![0.; 7]]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        let result: &mut [f32] = &mut [0.0; 4];
        os.process_down(sig_vec, result);

        let expected_result: &[f32] = &[
            0.00000000e+00,
            3.02847249e-07,
            -4.27782121e-06,
            2.51688452e-05,
        ];

        for (r, e) in result.iter().zip(expected_result.iter()) {
            assert!(
                (r - e).abs() < 1e-7,
                "Assertion failed: res: {}, expected: {}",
                r,
                e
            )
        }
    }

    #[test]
    fn down_sample_4x() {
        let mut os = Oversample::new(OversampleFactor::FourTimes, 4);
        os.initialize_oversample_stages();
        let sig_vec = &mut vec![vec![1.], vec![0.; 15]]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        let result: &mut [f32] = &mut [0.0; 4];
        os.process_down(sig_vec, result);
        let expected_result: &[f32] =
            &[0.0000000e+00, 0.0000000e+00, -1.4802341e-13, -1.0739319e-12];

        for (r, e) in result.iter().zip(expected_result.iter()) {
            assert!(
                (r - e).abs() < 1e-7,
                "Assertion failed: res: {}, expected: {}",
                r,
                e
            )
        }
    }

    #[test]
    fn down_sample_8x() {
        let mut os = Oversample::new(OversampleFactor::EightTimes, 4);
        os.initialize_oversample_stages();
        let mut sig_vec = &mut vec![vec![1.], vec![0.; 31]]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        // let sig: &mut [f32] = sig_vec.as_slice();

        let result: &mut [f32] = &mut [0.0; 4];
        os.process_down(&mut sig_vec, result);

        let expected_result: &[f32] = &[
            0.00000000e+00,
            0.00000000e+00,
            -1.63016536e-20,
            7.48833181e-19,
        ];

        for (r, e) in result.iter().zip(expected_result.iter()) {
            assert!(
                (r - e).abs() < 1e-7,
                "Assertion failed: res: {}, expected: {}",
                r,
                e
            )
        }
    }

    #[test]
    fn down_sample_16x() {
        let mut os = Oversample::new(OversampleFactor::SixteenTimes, 4);
        os.initialize_oversample_stages();
        let sig_vec = &mut vec![vec![1.], vec![0.; 63]]
            .into_iter()
            .flatten()
            .collect::<Vec<f32>>();

        let result: &mut [f32] = &mut [0.0; 4];
        os.process_down(sig_vec, result);

        let expected_result: &[f32] = &[
            0.00000000e+00,
            0.00000000e+00,
            -1.79528298e-27,
            -2.32264440e-25,
        ];
        for (r, e) in result.iter().zip(expected_result.iter()) {
            assert!(
                (r - e).abs() < 1e-7,
                "Assertion failed: res: {}, expected: {}",
                r,
                e
            )
        }
    }
}
*/
