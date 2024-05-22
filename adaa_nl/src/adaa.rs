use polylog::Li2;
use std::f64::consts::PI;
use std::fmt::Debug;

use nih_plug::prelude::*;

const ERR_TOL: f64 = 1e-5;

#[derive(Enum, Debug, Clone, Copy, PartialEq)]
pub enum ProcessorStyle {
    #[id = "hard clip"]
    #[name = "Hard Clip"]
    HardClip = 0,
    #[id = "tanh"]
    #[name = "Tanh"]
    Tanh = 1,
}
use ProcessorStyle::*;

#[derive(Enum, Copy, Clone, Debug, PartialEq)]
pub enum AntiderivativeOrder {
    #[id = "first order ad"]
    #[name = "First Order"]
    FirstOrder,

    #[id = "second order ad"]
    #[name = "Second Order"]
    SecondOrder,
}
use AntiderivativeOrder::*;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ProcessorState {
    State(ProcessorStyle, AntiderivativeOrder),
}
use ProcessorState::*;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ProcStateTransition {
    NoChange,
    ChangeOrder(AntiderivativeOrder),
    ChangeStyle(ProcessorStyle),
}
use ProcStateTransition::*;

type H = fn(f64) -> f64;
type H1 = fn(f64) -> f64;
type H2 = fn(f64) -> f64;

#[derive(Debug, Copy, Clone)]
struct ProcState {
    x1: f64,
    x2: f64,
    d2: f64,
    ad1_x1: f64,
    ad2_x0: f64,
    ad2_x1: f64,
    nl_func: H,
    nl_func_ad1: H1,
    nl_func_ad2: H2,
}

impl ProcState {
    #[inline]
    fn tanh_ad2(x: f64) -> f64 {
        let expval = (-2. * x).exp();
        0.5 * (Li2::li2(&(-expval)) - x * (x + 2.0 * (expval + 1.).ln() - 2.0 * x.cosh().ln()))
            + (PI.powi(2) / 24.0)
    }

    #[inline]
    fn hard_clip_ad1(val: f64) -> f64 {
        if val.abs() <= 1.0 {
            0.5 * val.powi(2)
        } else {
            (val * val.signum()) - 0.5
        }
    }

    #[inline]
    fn hard_clip_ad2(val: f64) -> f64 {
        if val.abs() <= 1. {
            val.powi(3) * (1. / 6.)
        } else {
            (((val.powi(2) * 0.5) + (1. / 6.)) * val.signum()) - (val * 0.5)
        }
    }

    fn first_order_tanh() -> ProcState {
        ProcState {
            x1: 0.0,
            x2: 0.0,
            d2: 0.0,
            ad1_x1: 0.0,
            ad2_x0: 0.0,
            ad2_x1: 0.0,
            nl_func: |x| x.tanh(),
            nl_func_ad1: |x| x.cosh().ln(),
            nl_func_ad2: |x| x,
        }
    }

    fn second_order_tanh() -> ProcState {
        ProcState {
            x1: 0.0,
            x2: 0.0,
            d2: 0.0,
            ad1_x1: 0.0,
            ad2_x0: 0.0,
            ad2_x1: 0.0,
            nl_func: |x| x.tanh(),
            nl_func_ad1: |x| x.cosh().ln(),
            nl_func_ad2: |x| ProcState::tanh_ad2(x),
        }
    }

    fn first_order_hard_clip() -> ProcState {
        ProcState {
            x1: 0.0,
            x2: 0.0,
            d2: 0.0,
            ad1_x1: 0.0,
            ad2_x0: 0.0,
            ad2_x1: 0.0,
            nl_func: |x| x.clamp(-1.0, 1.0),
            nl_func_ad1: |x| ProcState::hard_clip_ad1(x),
            nl_func_ad2: |x| x,
        }
    }

    fn second_order_hard_clip() -> ProcState {
        ProcState {
            x1: 0.0,
            x2: 0.0,
            d2: 0.0,
            ad1_x1: 0.0,
            ad2_x0: 0.0,
            ad2_x1: 0.0,
            nl_func: |x| x.clamp(-1.0, 1.0),
            nl_func_ad1: |x| ProcState::hard_clip_ad1(x),
            nl_func_ad2: |x| ProcState::hard_clip_ad2(x),
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct ADAA {
    current_proc_state: ProcState,
    proc_alg: fn(f64, &mut ProcState) -> f32,
}

impl ADAA {
    fn first_order_tanh() -> ADAA {
        let new_state = ProcState::first_order_tanh();
        ADAA {
            current_proc_state: new_state,
            proc_alg: |x: f64, y: &mut ProcState| ADAA::process_first_order(y, x),
        }
    }

    fn second_order_tanh() -> ADAA {
        let new_state = ProcState::second_order_tanh();
        ADAA {
            current_proc_state: new_state,
            proc_alg: |x: f64, y: &mut ProcState| ADAA::process_second_order(y, x),
        }
    }

    fn first_order_hard_clip() -> ADAA {
        let new_state = ProcState::first_order_hard_clip();
        ADAA {
            current_proc_state: new_state,
            proc_alg: |x: f64, y: &mut ProcState| ADAA::process_first_order(y, x),
        }
    }

    fn second_order_hard_clip() -> ADAA {
        let new_state = ProcState::second_order_hard_clip();
        ADAA {
            current_proc_state: new_state,
            proc_alg: |x: f64, y: &mut ProcState| ADAA::process_second_order(y, x),
        }
    }

    fn process(&mut self, val: f64) -> f32 {
        (self.proc_alg)(val, &mut self.current_proc_state)
    }

    fn process_first_order(state: &mut ProcState, val: f64) -> f32 {
        let diff = val - state.x1;
        let ad1_x0 = (state.nl_func_ad1)(val);

        let result = if diff.abs() < 1e-5 {
            (state.nl_func)((val + state.x1) / 2.)
        } else {
            (ad1_x0 - state.ad1_x1) / diff
        };

        state.x1 = val;
        state.ad1_x1 = ad1_x0;

        result as f32
    }

    fn process_second_order(state: &mut ProcState, val: f64) -> f32 {
        state.ad2_x0 = (state.nl_func_ad2)(val);
        let d1 = if (val - state.x1).abs() < ERR_TOL {
            (state.nl_func_ad1)(0.5 * (val + state.x1))
        } else {
            (state.ad2_x0 - state.ad2_x1) / (val - state.x1)
        };

        let result = if (val - state.x2).abs() < ERR_TOL {
            let xbar = 0.5 * (val + state.x2);
            let delta = xbar - state.x1;
            if delta.abs() < ERR_TOL {
                (state.nl_func)(0.5 * (xbar + state.x1))
            } else {
                (2.0 / delta)
                    * ((state.nl_func_ad1)(xbar)
                        + (state.ad2_x1 - (state.nl_func_ad2)(xbar)) / delta)
            }
        } else {
            (2.0 / (val - state.x2)) * (d1 - state.d2)
        };

        state.d2 = d1;
        state.x2 = state.x1;
        state.x1 = val;
        state.ad2_x1 = state.ad2_x0;

        result as f32
    }
}

#[derive(Debug, Clone)]
pub struct NonlinearProcessor {
    state: ProcessorState,
    proc: ADAA,
}

impl NonlinearProcessor {
    pub fn new() -> Self {
        NonlinearProcessor {
            state: State(HardClip, FirstOrder),
            proc: ADAA::first_order_hard_clip(),
        }
    }

    pub fn from_state(new_state: ProcessorState) -> Self {
        NonlinearProcessor {
            state: new_state,
            proc: match new_state {
                State(HardClip, FirstOrder) => ADAA::first_order_hard_clip(),
                State(HardClip, SecondOrder) => ADAA::second_order_hard_clip(),
                State(Tanh, FirstOrder) => ADAA::first_order_tanh(),
                State(Tanh, SecondOrder) => ADAA::second_order_tanh(),
            },
        }
    }

    pub fn change_state(&mut self, transition: ProcStateTransition) {
        nih_dbg!("Changing State!");
        match (self.state, transition) {
            (_, NoChange) => (),
            (State(old_style, _), ChangeOrder(new_order)) => {
                self.reset_with_new_state(State(old_style, new_order))
            }
            (State(_, old_order), ChangeStyle(new_style)) => {
                self.reset_with_new_state(State(new_style, old_order))
            }
        };
    }

    pub fn compare_and_change_state(&mut self, other_state: ProcessorState) {
        match (self.state, other_state) {
            (State(Tanh, _), State(HardClip, _)) => self.change_state(ChangeStyle(HardClip)),
            (State(HardClip, _), State(Tanh, _)) => self.change_state(ChangeStyle(Tanh)),
            (_, _) => (),
        }
        match (self.state, other_state) {
            (State(_, FirstOrder), State(_, SecondOrder)) => {
                self.change_state(ChangeOrder(SecondOrder))
            }
            (State(_, SecondOrder), State(_, FirstOrder)) => {
                self.change_state(ChangeOrder(FirstOrder))
            }
            (_, _) => (),
        }
    }

    fn reset_with_new_state(&mut self, new_state: ProcessorState) {
        match new_state {
            State(HardClip, FirstOrder) => self.initialize_as_first_order_hc(),
            State(HardClip, SecondOrder) => self.initialize_as_second_order_hc(),
            State(Tanh, FirstOrder) => self.initialize_as_first_order_tanh(),
            State(Tanh, SecondOrder) => self.initialize_as_second_order_tanh(),
        }
    }

    pub fn process(&mut self, val: f32) -> f32 {
        self.proc.process(val as f64)
    }

    fn initialize_as_first_order_hc(&mut self) {
        self.state = State(HardClip, FirstOrder);
        self.proc = ADAA::first_order_hard_clip();
    }

    fn initialize_as_first_order_tanh(&mut self) {
        self.state = State(Tanh, FirstOrder);
        self.proc = ADAA::first_order_tanh();
    }

    fn initialize_as_second_order_hc(&mut self) {
        self.state = State(HardClip, SecondOrder);
        self.proc = ADAA::second_order_hard_clip();
    }

    fn initialize_as_second_order_tanh(&mut self) {
        self.state = State(Tanh, SecondOrder);
        self.proc = ADAA::second_order_tanh();
    }
}

#[cfg(test)]

mod test {

    use super::*;

    const ERR_TOL: f64 = 1e-5;
    const INPUT_LINSPACE: [f64; 50] = [
        -2., -1.92, -1.84, -1.76, -1.68, -1.6, -1.52, -1.44, -1.36, -1.28, -1.2, -1.12, -1.04,
        -0.96, -0.88, -0.8, -0.72, -0.64, -0.56, -0.48, -0.4, -0.32, -0.24, -0.16, -0.08, 0., 0.08,
        0.16, 0.24, 0.32, 0.4, 0.48, 0.56, 0.64, 0.72, 0.8, 0.88, 0.96, 1.04, 1.12, 1.2, 1.28,
        1.36, 1.44, 1.52, 1.6, 1.68, 1.76, 1.84, 1.92,
    ];

    fn check_results_64(input: &[f64], expected: &[f64]) {
        input
            .iter()
            .zip(expected.iter())
            .for_each(|(r, e)| assert!((r - e).abs() <= ERR_TOL, "failure. r: {}, e: {}", r, e));
    }

    /*
    fn check_results_32(input: &[f32], expected: &[f32]) {
        input.iter().zip(expected.iter()).for_each(|(r, e)| {
            assert!(
                (r - e).abs() <= ERR_TOL as f32,
                "failure. r: {}, e: {}",
                r,
                e
            )
        });
    }
    */

    #[test]
    fn tanh_ad1_test() {
        let expected_results = [
            1.32500275, 1.24811869, 1.17176294, 1.09602265, 1.02099842, 0.94680615, 0.87357884,
            0.80146861, 0.73064865, 0.66131513, 0.59368897, 0.5280172, 0.46457382, 0.40365993,
            0.34560279, 0.29075356, 0.23948351, 0.19217836, 0.1492307, 0.11103042, 0.07795349,
            0.05034933, 0.02852769, 0.01274576, 0.00319659, 0., 0.00319659, 0.01274576, 0.02852769,
            0.05034933, 0.07795349, 0.11103042, 0.1492307, 0.19217836, 0.23948351, 0.29075356,
            0.34560279, 0.40365993, 0.46457382, 0.5280172, 0.59368897, 0.66131513, 0.73064865,
            0.80146861, 0.87357884, 0.94680615, 1.02099842, 1.09602265, 1.17176294, 1.24811869,
        ];

        let ps = ProcState::first_order_tanh();

        let result: Vec<_> = INPUT_LINSPACE
            .iter()
            .map(|v| (ps.nl_func_ad1)(*v))
            .collect::<Vec<f64>>();

        check_results_64(&result, &expected_results);
    }

    #[test]
    fn tanh_ad2_test() {
        let expected_results = [
            -1.01582293e+00,
            -9.12901331e-01,
            -8.16109863e-01,
            -7.25402860e-01,
            -6.40727157e-01,
            -5.62020942e-01,
            -4.89212458e-01,
            -4.22218557e-01,
            -3.60943093e-01,
            -3.05275157e-01,
            -2.55087166e-01,
            -2.10232828e-01,
            -1.70545014e-01,
            -1.35833587e-01,
            -1.05883267e-01,
            -8.04516148e-02,
            -5.92672653e-02,
            -4.20285287e-02,
            -2.84025257e-02,
            -1.80250081e-02,
            -1.05010128e-02,
            -5.40647439e-03,
            -2.29087261e-03,
            -6.80927511e-04,
            -8.52787865e-05,
            -4.99600361e-16,
            8.52787865e-05,
            6.80927511e-04,
            2.29087261e-03,
            5.40647439e-03,
            1.05010128e-02,
            1.80250081e-02,
            2.84025257e-02,
            4.20285287e-02,
            5.92672653e-02,
            8.04516148e-02,
            1.05883267e-01,
            1.35833587e-01,
            1.70545014e-01,
            2.10232828e-01,
            2.55087166e-01,
            3.05275157e-01,
            3.60943093e-01,
            4.22218557e-01,
            4.89212458e-01,
            5.62020942e-01,
            6.40727157e-01,
            7.25402860e-01,
            8.16109863e-01,
            9.12901331e-01,
        ];

        let ps = ProcState::second_order_tanh();

        dbg!(&ps);

        let result: Vec<_> = INPUT_LINSPACE
            .iter()
            .map(|v| (ps.nl_func_ad2)(*v))
            .collect();

        check_results_64(&result, &expected_results);
    }

    #[test]
    fn hard_clip_ad1_test() {
        let expected_result = [
            1.5,
            1.42,
            1.34,
            1.26,
            1.18,
            1.1,
            1.02,
            0.94,
            0.8599999999999999,
            0.78,
            0.7,
            0.6200000000000001,
            0.54,
            0.4608,
            0.38719999999999993,
            0.32000000000000006,
            0.2592,
            0.20479999999999993,
            0.15680000000000002,
            0.1152,
            0.07999999999999996,
            0.05120000000000002,
            0.0288,
            0.012799999999999987,
            0.003200000000000006,
            0.0,
            0.003200000000000006,
            0.012800000000000023,
            0.02880000000000005,
            0.05119999999999995,
            0.07999999999999996,
            0.1152,
            0.15680000000000002,
            0.2048000000000001,
            0.25920000000000015,
            0.32000000000000023,
            0.38719999999999993,
            0.4608,
            0.54,
            0.6200000000000001,
            0.7000000000000002,
            0.7800000000000002,
            0.8599999999999999,
            0.94,
            1.02,
            1.1,
            1.1800000000000002,
            1.2600000000000002,
            1.3399999999999999,
            1.42,
        ];

        let ps = ProcState::first_order_hard_clip();

        let result: Vec<_> = INPUT_LINSPACE
            .iter()
            .map(|v| (ps.nl_func_ad1)(*v))
            .collect();
        check_results_64(&result, &expected_result);
    }

    #[test]
    fn hard_clip_ad2_test() {
        let expected_result = [
            -1.1666666666666665,
            -1.0498666666666665,
            -0.9394666666666668,
            -0.8354666666666667,
            -0.7378666666666666,
            -0.646666666666667,
            -0.5618666666666667,
            -0.4834666666666667,
            -0.41146666666666665,
            -0.34586666666666666,
            -0.2866666666666666,
            -0.23386666666666667,
            -0.18746666666666667,
            -0.147456,
            -0.11357866666666662,
            -0.08533333333333336,
            -0.06220799999999999,
            -0.04369066666666665,
            -0.02926933333333334,
            -0.018432,
            -0.01066666666666666,
            -0.005461333333333336,
            -0.002304,
            -0.0006826666666666656,
            -8.533333333333357e-05,
            0.0,
            8.533333333333357e-05,
            0.0006826666666666686,
            0.002304000000000006,
            0.005461333333333325,
            0.01066666666666666,
            0.018432,
            0.02926933333333334,
            0.04369066666666669,
            0.06220800000000005,
            0.08533333333333343,
            0.11357866666666662,
            0.147456,
            0.18746666666666667,
            0.23386666666666667,
            0.28666666666666674,
            0.3458666666666669,
            0.41146666666666665,
            0.4834666666666667,
            0.5618666666666667,
            0.646666666666667,
            0.7378666666666669,
            0.835466666666667,
            0.9394666666666665,
            1.0498666666666665,
        ];

        let ps = ProcState::second_order_hard_clip();

        let result: Vec<_> = INPUT_LINSPACE
            .iter()
            .map(|v| (ps.nl_func_ad2)(*v))
            .collect();

        check_results_64(&result, &expected_result);
    }

    #[test]
    fn process_hard_clip_ad1() {
        let expected_result = [
            -0.75, -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -0.99, -0.92, -0.84,
            -0.76, -0.68, -0.6, -0.52, -0.44, -0.36, -0.28, -0.2, -0.12, -0.04, 0.04, 0.12, 0.2,
            0.28, 0.36, 0.44, 0.52, 0.6, 0.68, 0.76, 0.84, 0.92, 0.99, 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.,
        ];

        let mut ad1_hc = NonlinearProcessor::new();

        assert_eq!(ad1_hc.state, State(HardClip, FirstOrder));

        let result: Vec<_> = INPUT_LINSPACE
            .into_iter()
            .map(|v| ad1_hc.process(v as f32) as f64)
            .collect();

        check_results_64(&result, &expected_result);
    }

    #[test]
    fn process_hard_clip_ad2() {
        let mut ad2_hc = NonlinearProcessor::new();
        ad2_hc.change_state(ChangeOrder(SecondOrder));

        assert_eq!(ad2_hc.state, State(HardClip, SecondOrder));

        let expected_result = [
            -0.58333333,
            -0.91319444,
            -1.,
            -1.,
            -1.,
            -1.,
            -1.,
            -1.,
            -1.,
            -1.,
            -1.,
            -1.,
            -1.,
            -0.99833333,
            -0.95833333,
            -0.88,
            -0.8,
            -0.72,
            -0.64,
            -0.56,
            -0.48,
            -0.4,
            -0.32,
            -0.24,
            -0.16,
            -0.08,
            0.,
            0.08,
            0.16,
            0.24,
            0.32,
            0.4,
            0.48,
            0.56,
            0.64,
            0.72,
            0.8,
            0.88,
            0.95833333,
            0.99833333,
            1.,
            1.,
            1.,
            1.,
            1.,
            1.,
            1.,
            1.,
            1.,
            1.,
        ];

        let result: Vec<_> = INPUT_LINSPACE
            .into_iter()
            .map(|v| ad2_hc.process(v as f32) as f64)
            .collect();

        check_results_64(&result, &expected_result);
    }

    #[test]
    fn test_close_hc_ad2() {
        let expected_result: &[f64] = &[
            -3.33333333e-06,
            -8.25000000e-06,
            -1.15000000e-05,
            -1.25000000e-05,
            -1.35000000e-05,
            -1.45000000e-05,
            -1.55000000e-05,
            -1.65000000e-05,
            -1.75000000e-05,
            -1.85000000e-05,
        ];

        let input_sig = [
            -1.0e-05, -1.1e-05, -1.2e-05, -1.3e-05, -1.4e-05, -1.5e-05, -1.6e-05, -1.7e-05,
            -1.8e-05, -1.9e-05,
        ];

        let mut ad2_hc = NonlinearProcessor::new();

        ad2_hc.change_state(ChangeOrder(SecondOrder));

        let result: Vec<_> = input_sig
            .into_iter()
            .map(|v| ad2_hc.process(v) as f64)
            .collect();

        check_results_64(&result, expected_result);
    }

    #[test]
    fn test_os_2x_hc_ad2() {
        let expected_result: &[f64] = &[
            0.00431436,
            0.00431436,
            0.00194573,
            -0.00236863,
            0.00012454,
            0.00249318,
            -0.00229388,
            -0.00478705,
            -0.00405785,
            0.00072921,
        ];
        let input: &[f32] = &[
            0.01294309,
            0.,
            -0.0071059,
            0.,
            0.00747953,
            0.,
            -0.01436116,
            0.,
            0.00218762,
            0.,
        ];

        let mut ad2_hc = NonlinearProcessor::new();

        ad2_hc.change_state(ChangeOrder(SecondOrder));

        let result: Vec<_> = input.iter().map(|v| ad2_hc.process(*v) as f64).collect();

        check_results_64(&result, expected_result);
    }
}
