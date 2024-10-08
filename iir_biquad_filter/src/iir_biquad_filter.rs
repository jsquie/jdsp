use std::f32::consts::PI;

#[derive(Debug)]
pub enum FilterType {
    Lowpass,
    Highpass,
    Bandpass,
    Bandreject,
}

#[derive(Debug, Clone, Copy)]
pub enum FilterOrder {
    First,
    Second,
}

const W1: usize = 0;
const W2: usize = 1;

const B0: usize = 0;
const B1: usize = 1;
const B2: usize = 2;
const A1: usize = 3;
const A2: usize = 4;

const FIRST_ORDER_Q_VALS: [f32; 1] = [0.70710677];
const SECOND_ORDER_Q_VALS: [f32; 2] = [0.54, 1.31];

#[derive(Debug)]
pub struct IIRBiquadFilter {
    coefs: [[f32; 5]; 2],
    filter_type: FilterType,
    states: [[f32; 2]; 2],
    order: FilterOrder,
    cutoff_freq: f32,
    sample_rate: f32,
}

impl Default for IIRBiquadFilter {
    fn default() -> Self {
        IIRBiquadFilter {
            coefs: [[0.0_f32; 5]; 2],
            filter_type: FilterType::Lowpass,
            states: [[0.0_f32; 2]; 2],
            order: FilterOrder::First,
            cutoff_freq: 1000.0,
            sample_rate: 44100.0,
        }
    }
}

impl IIRBiquadFilter {
    pub fn new(ft: FilterType) -> Self {
        let mut new_biquad = IIRBiquadFilter::default();
        new_biquad.set_filter_type(ft);
        new_biquad
    }

    pub fn init(&mut self, sample_rate: &f32, cutoff_freq: &f32, order: FilterOrder) {
        self.sample_rate = *sample_rate;
        self.gen_coefficients(cutoff_freq, order);
        self.order = order;
        self.cutoff_freq = *cutoff_freq;
    }

    pub fn set_filter_type(&mut self, new_filter_type: FilterType) {
        self.filter_type = new_filter_type;
        self.gen_coefficients(&self.cutoff_freq.clone(), self.order);
    }

    pub fn reset(&mut self) {
        self.states = [[0.0_f32; 2]; 2];
    }

    pub fn get_current_cutoff(&self) -> f32 {
        self.cutoff_freq
    }

    pub fn set_cutoff(&mut self, new_cutoff_freq: f32) {
        self.cutoff_freq = new_cutoff_freq;
        self.gen_coefficients(&new_cutoff_freq, self.order);
    }

    #[inline]
    fn gen_coefficients(&mut self, cutoff_freq: &f32, order: FilterOrder) {
        match order {
            FilterOrder::First => {
                self.coefs = [
                    match self.filter_type {
                        FilterType::Lowpass => Self::calculate_lowpass_sections(
                            &cutoff_freq,
                            &self.sample_rate,
                            &FIRST_ORDER_Q_VALS[0],
                        ),
                        FilterType::Highpass => Self::calculate_highpass_sections(
                            &cutoff_freq,
                            &self.sample_rate,
                            &FIRST_ORDER_Q_VALS[0],
                        ),
                        FilterType::Bandpass => Self::calculate_bandpass_sections(
                            &cutoff_freq,
                            &self.sample_rate,
                            &FIRST_ORDER_Q_VALS[0],
                        ),
                        FilterType::Bandreject => Self::calculate_bandreject_sections(
                            &cutoff_freq,
                            &self.sample_rate,
                            &FIRST_ORDER_Q_VALS[0],
                        ),
                    },
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            }
            FilterOrder::Second => {
                self.coefs = [
                    Self::calculate_lowpass_sections(
                        &cutoff_freq,
                        &self.sample_rate,
                        &SECOND_ORDER_Q_VALS[0],
                    ),
                    Self::calculate_lowpass_sections(
                        &cutoff_freq,
                        &self.sample_rate,
                        &SECOND_ORDER_Q_VALS[1],
                    ),
                ]
            }
        };
    }

    #[inline]
    fn calculate_lowpass_sections(fc: &f32, fs: &f32, q_value: &f32) -> [f32; 5] {
        let omega_0: f32 = 2. * PI * (*fc / *fs);
        let alpha: f32 = omega_0.sin() / (2. * q_value);
        let cos_omega: f32 = omega_0.cos();
        let a0: f32 = 1. + alpha;
        let b0: f32 = ((1. - cos_omega) / 2.) / a0;
        let b1: f32 = (1. - cos_omega) / a0;
        let b2: f32 = ((1. - cos_omega) / 2.) / a0;
        let a1: f32 = (-2. * cos_omega) / a0;
        let a2: f32 = (1. - alpha) / a0;
        [b0, b1, b2, a1, a2]
    }

    fn calculate_highpass_sections(_fc: &f32, _fs: &f32, _q_value: &f32) -> [f32; 5] {
        unimplemented!()
    }

    fn calculate_bandpass_sections(_fc: &f32, _fs: &f32, _q_value: &f32) -> [f32; 5] {
        unimplemented!()
    }

    fn calculate_bandreject_sections(_fc: &f32, _fs: &f32, _q_value: &f32) -> [f32; 5] {
        unimplemented!()
    }

    pub fn process_sample(&mut self, sample: &mut f32) {
        let mut y: f32 = 0.0;
        let num_sections: usize = match &self.order {
            FilterOrder::First => 1,
            FilterOrder::Second => 2,
        };
        for i in 0..num_sections {
            let state = self.states[i];
            let coefs = self.coefs[i];

            let x = if i == 0 { *sample } else { y };

            y = (coefs[B0] * x) + state[W1];
            self.states[i][W1] = (coefs[B1] * x) - (coefs[A1] * y) + state[W2];
            self.states[i][W2] = (coefs[B2] * x) - (coefs[A2] * y);
        }
        *sample = y;
    }

    pub fn process_block(&mut self, input_signal: &mut [f32]) {
        input_signal.iter_mut().for_each(|s| {
            let mut y: f32 = 0.0;
            let num_sections: usize = match &self.order {
                FilterOrder::First => 1,
                FilterOrder::Second => 2,
            };
            for i in 0..num_sections {
                let state = self.states[i];
                let coefs = self.coefs[i];

                let x = if i == 0 { *s } else { y };

                y = (coefs[B0] * x) + state[W1];
                self.states[i][W1] = (coefs[B1] * x) - (coefs[A1] * y) + state[W2];
                self.states[i][W2] = (coefs[B2] * x) - (coefs[A2] * y);
            }
            *s = y;
        });
    }
}

#[cfg(test)]
mod tests {

    use crate::iir_biquad_filter;

    use super::*;

    const FIRST_ORDER_1000_441_LPF_COEFS: [f32; 5] = [
        0.004604009,
        0.009208018,
        0.004604009,
        -1.7990962,
        0.81751233,
    ];

    #[test]
    fn test_calculate_lowpass() {
        assert_eq!(
            iir_biquad_filter::IIRBiquadFilter::calculate_lowpass_sections(
                &1000.0,
                &44100.0,
                &FIRST_ORDER_Q_VALS[0]
            ),
            FIRST_ORDER_1000_441_LPF_COEFS
        )
    }

    #[test]
    fn test_calculate_lowpass_order_2() {
        assert_eq!(
            iir_biquad_filter::IIRBiquadFilter::calculate_lowpass_sections(
                &2500.0,
                &48000.0,
                &SECOND_ORDER_Q_VALS[0]
            ),
            [
                0.020448789,
                0.040897578,
                0.020448789,
                -1.4594773,
                0.54127246
            ]
        );
        assert_eq!(
            iir_biquad_filter::IIRBiquadFilter::calculate_lowpass_sections(
                &2500.0,
                &48000.0,
                &SECOND_ORDER_Q_VALS[1]
            ),
            [0.023635214, 0.04727043, 0.023635214, -1.6868998, 0.7814407]
        )
    }

    #[test]
    fn test_gen_coefs() {
        let mut f = IIRBiquadFilter::default();
        f.init(&44100.0, &1000.0, FilterOrder::First);
        assert_eq!(f.coefs[0], FIRST_ORDER_1000_441_LPF_COEFS);
    }

    #[test]
    fn test_proc() {
        let mut input_signal = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.];
        assert_eq!(input_signal.len(), 11);
        let mut f = IIRBiquadFilter::default();
        f.init(&44100.0, &1000.0, FilterOrder::First);

        f.process_block(&mut input_signal);

        let expected_result: [f32; 11] = [
            0.004604, 0.01749103, 0.03230823, 0.04382648, 0.05243569, 0.05850817, 0.06239501,
            0.06442348, 0.06489536, 0.06408601, 0.06224416,
        ];

        input_signal
            .iter()
            .zip(expected_result.into_iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-5));
    }

    #[test]
    fn test_proc_100hz() {
        let mut input_signal = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.];
        assert_eq!(input_signal.len(), 11);
        let mut f = IIRBiquadFilter::default();
        f.init(&44100.0, &100.0, FilterOrder::First);

        f.process_block(&mut input_signal);

        let expected_result: [f32; 11] = [
            5.024142299431054e-05,
            0.00019995340480202317,
            0.0003968802473011646,
            0.0005897991339120711,
            0.0007787512432243895,
            0.0009637777296043322,
            0.0011449197154072583,
            0.0013222182833520274,
            0.0014957144690554308,
            0.0016654492537250094,
            0.0018314635570085642,
        ];

        input_signal
            .iter()
            .zip(expected_result.into_iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-6, "{} {}", a, b));
    }

    const RANDOM_NORMAL_480: [f32; 480] = [
        8.39230795e-01,
        -9.16539688e-01,
        3.18156358e-01,
        4.17256114e-01,
        8.21000542e-01,
        -4.10880524e-01,
        -6.67290289e-01,
        6.58536394e-01,
        -5.16765416e-02,
        1.09171495e+00,
        1.15067683e+00,
        3.16417324e+00,
        -2.18070921e+00,
        7.03969993e-01,
        -1.43694991e-01,
        5.85904548e-01,
        2.70337631e-01,
        -6.59921572e-01,
        1.79980190e-01,
        7.28376389e-02,
        -6.11702129e-01,
        1.68534955e-01,
        -1.97170269e-01,
        -1.00033774e+00,
        9.70550929e-01,
        6.80547089e-01,
        2.49604689e-01,
        -8.21201147e-01,
        3.53400408e-01,
        -6.06874722e-01,
        1.48535410e+00,
        9.15034526e-01,
        -2.08098471e+00,
        9.34496930e-01,
        -1.02108645e+00,
        -9.31705451e-01,
        2.30125386e-01,
        -1.02820454e+00,
        -1.30125298e+00,
        -4.28937070e-01,
        -7.97314961e-01,
        -3.30346604e-01,
        2.77625493e-01,
        2.46818166e+00,
        2.96065815e+00,
        -6.66584525e-01,
        -1.16431914e-03,
        -3.23383661e-01,
        1.19380545e+00,
        5.06839195e-01,
        7.35433095e-01,
        -9.15973158e-01,
        4.98753948e-01,
        -5.35441496e-01,
        -2.44866156e-01,
        8.20092394e-01,
        -3.68286276e-01,
        3.40082250e-01,
        -1.39645644e+00,
        1.73260172e-01,
        1.27809792e-01,
        7.31697910e-01,
        7.86043757e-01,
        6.00756770e-01,
        5.52220004e-01,
        9.98655737e-01,
        -1.97811307e-01,
        1.97769744e-01,
        8.70147828e-01,
        2.46946032e-02,
        6.62879845e-01,
        -1.15434953e+00,
        2.12702656e+00,
        -5.00580366e-01,
        6.33446060e-01,
        2.58499470e-01,
        4.51758695e-01,
        4.14941819e-01,
        1.28794886e+00,
        1.20134788e+00,
        2.12603441e-01,
        2.09122641e-02,
        -1.09172897e+00,
        -6.33725618e-01,
        -6.09181832e-01,
        4.52403888e-01,
        -2.14600675e-01,
        2.14284835e+00,
        1.14738059e-02,
        1.13210377e+00,
        -5.05107884e-03,
        1.85031943e+00,
        -2.74374166e-01,
        1.35775344e+00,
        -1.60019309e+00,
        4.02616256e-01,
        3.32135807e-01,
        -2.67666377e-01,
        1.12100536e+00,
        -1.37553002e+00,
        8.30829192e-01,
        4.83646716e-01,
        2.35154829e-03,
        1.06440253e+00,
        1.73536803e+00,
        -1.11141899e-01,
        9.56385463e-01,
        8.17751194e-01,
        1.09188499e+00,
        -7.59912593e-01,
        9.50557186e-01,
        1.55693719e+00,
        -1.04755407e-01,
        -1.25105718e+00,
        3.25210947e-02,
        1.64004651e+00,
        -1.52612016e-01,
        1.25829616e+00,
        -1.53466505e+00,
        -1.14510969e+00,
        -2.40511773e-01,
        -7.46539854e-01,
        3.54843442e-01,
        1.09230155e-01,
        -8.66497147e-01,
        -3.83789182e-01,
        1.18185335e+00,
        2.70612725e-01,
        -8.71301710e-01,
        -5.11281999e-01,
        -2.50349437e-01,
        4.48624616e-01,
        -3.08490915e-01,
        -6.59221967e-01,
        -2.70954861e-01,
        -8.64412840e-01,
        1.23125998e+00,
        1.05040287e+00,
        4.31810581e-01,
        5.40345138e-01,
        -1.40678747e-02,
        1.10501127e+00,
        -1.18013882e+00,
        -1.16334025e-01,
        -8.18066646e-01,
        -1.25246121e+00,
        1.65795797e+00,
        -4.40528019e-01,
        -7.13155612e-01,
        -4.77071329e-01,
        -4.05459631e-01,
        -3.26934076e-01,
        5.15284038e-01,
        -6.52381637e-01,
        -2.31315494e-01,
        -6.72452838e-01,
        -3.50980601e-01,
        -6.09190522e-02,
        1.42662591e-01,
        -9.38427660e-01,
        4.22427979e-01,
        -4.70280041e-01,
        -2.18344643e-01,
        -1.26764271e+00,
        -4.00898583e-02,
        8.88073718e-01,
        1.51630228e+00,
        4.49810116e-01,
        4.44810203e-01,
        -1.49564898e+00,
        4.87558477e-03,
        -6.72762048e-01,
        -2.38223137e-01,
        -3.49795861e-01,
        1.50441981e+00,
        -2.60777453e-01,
        1.66980102e+00,
        5.23152667e-02,
        -6.10950629e-01,
        -2.72897493e-02,
        6.85480971e-01,
        -3.78083170e-02,
        -2.97503227e-01,
        -2.93222391e+00,
        1.25751945e+00,
        -3.27174024e-01,
        2.32621569e+00,
        2.08116812e+00,
        2.31665018e+00,
        -3.66244068e-01,
        1.14454342e+00,
        -2.83349962e-01,
        7.24939662e-01,
        -2.28814469e-01,
        -1.50881689e-01,
        -2.70547661e-01,
        -9.69861533e-02,
        8.68966240e-01,
        -8.28147446e-01,
        1.29810076e+00,
        -5.83085707e-01,
        -2.64971658e-01,
        1.06044907e-01,
        -1.21532996e-01,
        -4.08418497e-01,
        5.43003813e-01,
        -1.28575372e+00,
        -4.31861033e-01,
        1.10421481e-01,
        -2.25900389e+00,
        -1.64622971e+00,
        3.96002570e-01,
        -7.77858480e-01,
        6.14523139e-01,
        -1.48456326e-01,
        -1.66988417e+00,
        3.45673283e-02,
        1.88155418e+00,
        6.33779868e-01,
        1.70300619e+00,
        4.08832827e-01,
        -1.09568941e+00,
        -2.78596442e-01,
        1.28555224e+00,
        1.99788107e-01,
        4.62171903e-01,
        1.39505465e-01,
        3.37853500e-01,
        -1.70237431e+00,
        -1.60210063e+00,
        -1.41866347e+00,
        -2.48773645e+00,
        -2.03776569e+00,
        -5.58335054e-02,
        1.37458343e+00,
        2.00639782e+00,
        6.15127787e-01,
        1.38525249e+00,
        -1.13793163e+00,
        -7.94551603e-01,
        1.33954951e-02,
        -1.77254018e+00,
        1.52338116e+00,
        4.60598267e-01,
        6.42921926e-01,
        1.35050797e-01,
        -2.98456520e-01,
        2.22608512e+00,
        5.20875163e-02,
        -2.28951510e-01,
        -3.53861646e-01,
        -9.94313112e-02,
        2.07355891e-01,
        9.92547252e-01,
        -1.16571277e+00,
        9.66882115e-01,
        1.17248600e+00,
        -1.60737786e+00,
        -1.10138579e-02,
        1.42002269e+00,
        3.02102011e+00,
        1.15705112e+00,
        -6.59704824e-01,
        3.09415675e-01,
        1.59654116e+00,
        4.44709451e-01,
        -5.62111767e-01,
        1.25519316e+00,
        2.74656248e-01,
        1.13162456e+00,
        4.61009388e-01,
        -1.93573924e-01,
        -3.95084239e-01,
        2.64969459e-01,
        -6.76073103e-01,
        -2.79246381e+00,
        -5.76186337e-01,
        8.10085041e-01,
        -3.36674061e-01,
        -1.16878462e-01,
        9.24566951e-01,
        -4.79810639e-01,
        2.04623380e+00,
        -1.02103745e+00,
        1.18395107e+00,
        -7.59174046e-01,
        2.39333464e-01,
        -1.97807490e+00,
        1.48741540e+00,
        -5.38422240e-02,
        5.11777585e-01,
        -2.28840326e+00,
        1.56174624e+00,
        1.52085241e+00,
        -8.09765221e-01,
        4.27392598e-01,
        2.05473949e-01,
        1.25803816e+00,
        8.89455506e-01,
        -3.97003193e-01,
        -9.09806705e-01,
        -1.01207995e+00,
        9.78876142e-01,
        -8.39682791e-01,
        -1.25410079e+00,
        1.26193487e+00,
        -1.24697183e+00,
        -6.37835909e-01,
        7.48785519e-01,
        -9.02506778e-01,
        8.41538519e-01,
        -1.73540193e-01,
        -2.74677198e+00,
        1.47813385e+00,
        1.72052599e+00,
        -3.79986437e-01,
        -5.46678810e-01,
        -9.27573899e-02,
        6.17909449e-01,
        9.83622769e-01,
        -1.43225028e-01,
        8.33204788e-01,
        4.45296684e-01,
        -7.62896447e-01,
        -6.04190906e-01,
        1.88439258e+00,
        7.94432665e-01,
        -1.12364924e-01,
        2.09141880e+00,
        9.31487605e-01,
        1.94097572e+00,
        -1.02002509e+00,
        -1.04651344e+00,
        -6.69337686e-01,
        1.02476751e+00,
        -1.53922961e-01,
        4.53402971e-01,
        -1.97174266e+00,
        -4.42900983e-01,
        9.15743935e-02,
        -4.85114614e-01,
        1.76888346e+00,
        -1.19867811e+00,
        4.43983037e-01,
        3.53794244e-01,
        -5.47493669e-01,
        -1.21392085e+00,
        6.37572258e-02,
        1.33719440e+00,
        -1.26628189e+00,
        -1.00613886e+00,
        -1.59225470e+00,
        -2.06527819e-01,
        -2.21080258e+00,
        -2.01948619e-02,
        1.29803880e+00,
        5.30007780e-01,
        -1.32911932e+00,
        -1.71305425e+00,
        7.23524566e-01,
        7.74191962e-02,
        6.18783834e-01,
        3.22047480e-02,
        9.53311737e-01,
        6.13092949e-01,
        1.20115058e+00,
        1.24894177e+00,
        -1.38592339e+00,
        8.71413004e-01,
        1.47976377e+00,
        5.23324350e-01,
        -1.42988713e-01,
        1.15051578e+00,
        9.09101550e-01,
        -4.44236367e-01,
        -4.69808592e-02,
        -5.07061946e-01,
        -1.61168224e+00,
        1.05406943e+00,
        2.34760031e-01,
        -2.01409413e-01,
        1.23477648e+00,
        -1.55647078e-01,
        -1.13564561e+00,
        2.06988103e+00,
        4.42305869e-01,
        1.32536323e-01,
        4.85465862e-02,
        -2.48946161e+00,
        1.38271926e-01,
        3.25215013e-02,
        -1.06562537e+00,
        3.41962867e-01,
        9.18018929e-01,
        1.61091527e+00,
        -3.22329843e-01,
        9.46032761e-01,
        1.08901091e+00,
        6.85717014e-01,
        -1.10979404e+00,
        1.36091124e+00,
        -1.16755273e+00,
        5.66016215e-01,
        -1.15595243e-01,
        -1.61551636e+00,
        -2.58929177e+00,
        -3.62166913e-01,
        2.63041950e-01,
        -1.65902856e+00,
        1.35261297e+00,
        -7.92921129e-01,
        4.16575096e-01,
        -2.76822903e-01,
        8.62883889e-01,
        -1.14330921e+00,
        1.27932948e+00,
        4.15486107e-01,
        1.07882250e+00,
        3.68825709e-01,
        -3.95949604e-01,
        -6.91757967e-01,
        -3.38157990e-02,
        -1.24942355e+00,
        -8.22903890e-02,
        9.89828296e-01,
        -4.40824163e-01,
        3.24659618e-01,
        8.13334705e-02,
        -1.29055939e+00,
        -1.09277496e+00,
        6.50939031e-01,
        -1.30524129e-02,
        5.26065127e-01,
        1.53320491e-01,
        6.76844783e-01,
        -3.50610123e-01,
        8.87082312e-01,
        1.45031488e+00,
        -1.07742060e+00,
        -8.64778276e-01,
        3.26651726e-01,
        2.41351091e-01,
        9.73160022e-01,
        1.60162070e+00,
        6.54670726e-01,
        -1.28382679e+00,
        1.86042924e-01,
        -5.99465079e-01,
        4.59024630e-02,
        3.77942940e-01,
        -1.13332301e+00,
        -3.12706851e-01,
        -1.78001972e+00,
        2.24875974e+00,
        7.13910883e-01,
        1.92170907e-01,
        2.54658118e-01,
        -6.95211854e-01,
        -1.92383252e-01,
        -9.87533127e-01,
        1.26653681e+00,
        -2.00824390e-01,
        1.22321539e+00,
        -1.98092860e-01,
        2.26180811e+00,
        -1.35526260e+00,
        8.27717075e-01,
        1.10330885e+00,
        -1.58955629e+00,
        8.35640096e-01,
        -1.39442878e-01,
        -1.85722181e-01,
        1.66212103e+00,
        1.19328271e+00,
        -1.97327131e+00,
        -1.44270135e-01,
        1.56360259e+00,
        -1.65613931e-01,
        -4.02321315e-03,
        -7.61884954e-01,
    ];

    #[test]
    fn test_rand_200() {
        let mut input_signal = RANDOM_NORMAL_480.clone();
        let expected_coefficients: [f32; 5] = [
            1.12479959e-04,
            2.24959919e-04,
            1.12479959e-04,
            -1.96977856e+00,
            9.70228476e-01,
        ];

        let mut f = IIRBiquadFilter::default();
        f.init(&44100.0, &150.0, FilterOrder::First);
        f.coefs[0]
            .iter()
            .zip(expected_coefficients.into_iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-6));

        f.process_block(&mut input_signal);

        let expected_result: [f32; 480] = [
            9.43966457e-05,
            2.71641433e-04,
            3.67485323e-04,
            4.75723487e-04,
            8.02523422e-04,
            1.30464229e-03,
            1.71608301e-03,
            1.99224517e-03,
            2.32656414e-03,
            2.83512620e-03,
            3.69647963e-03,
            5.26808444e-03,
            7.38648411e-03,
            9.38300890e-03,
            1.12127886e-02,
            1.30958076e-02,
            1.50629240e-02,
            1.70171889e-02,
            1.88078118e-02,
            2.05111163e-02,
            2.21223079e-02,
            2.35651198e-02,
            2.49013062e-02,
            2.60485922e-02,
            2.70119558e-02,
            2.81168505e-02,
            2.94665393e-02,
            3.08031185e-02,
            3.19691355e-02,
            3.30049257e-02,
            3.40653300e-02,
            3.54476446e-02,
            3.69117066e-02,
            3.80554710e-02,
            3.90093654e-02,
            3.96879203e-02,
            4.00298535e-02,
            4.01749151e-02,
            3.99457972e-02,
            3.92488991e-02,
            3.82225494e-02,
            3.69447906e-02,
            3.55556809e-02,
            3.44948482e-02,
            3.43695621e-02,
            3.51012154e-02,
            3.59782225e-02,
            3.67013189e-02,
            3.74477749e-02,
            3.84443526e-02,
            3.97249831e-02,
            4.10690377e-02,
            4.22873641e-02,
            4.33993377e-02,
            4.43667844e-02,
            4.52623996e-02,
            4.62265072e-02,
            4.71887603e-02,
            4.79791412e-02,
            4.84679984e-02,
            4.88167742e-02,
            4.92637446e-02,
            4.99426361e-02,
            5.09055485e-02,
            5.21025643e-02,
            5.35446303e-02,
            5.51841944e-02,
            5.68401919e-02,
            5.85414291e-02,
            6.03864500e-02,
            6.23273630e-02,
            6.42045075e-02,
            6.60510059e-02,
            6.81051629e-02,
            7.02654099e-02,
            7.24450000e-02,
            7.47073218e-02,
            7.70460547e-02,
            7.95695230e-02,
            8.24536009e-02,
            8.56537551e-02,
            8.89054055e-02,
            9.19260694e-02,
            9.45009198e-02,
            9.66227138e-02,
            9.84804298e-02,
            1.00247644e-01,
            1.02160780e-01,
            1.04430201e-01,
            1.06956021e-01,
            1.09613923e-01,
            1.12477703e-01,
            1.15590437e-01,
            1.18857614e-01,
            1.22068634e-01,
            1.24967164e-01,
            1.27671116e-01,
            1.30327022e-01,
            1.32948456e-01,
            1.35499384e-01,
            1.37823507e-01,
            1.40103011e-01,
            1.42454134e-01,
            1.44845820e-01,
            1.47536040e-01,
            1.50577400e-01,
            1.53738232e-01,
            1.57030418e-01,
            1.60568291e-01,
            1.64180730e-01,
            1.67670536e-01,
            1.71284494e-01,
            1.75159178e-01,
            1.78850539e-01,
            1.82061972e-01,
            1.85146952e-01,
            1.88412223e-01,
            1.91787185e-01,
            1.95068662e-01,
            1.97832172e-01,
            1.99967124e-01,
            2.01681668e-01,
            2.03099345e-01,
            2.04391578e-01,
            2.05520402e-01,
            2.06297341e-01,
            2.06907467e-01,
            2.07659476e-01,
            2.08391474e-01,
            2.08784841e-01,
            2.08831380e-01,
            2.08719210e-01,
            2.08554536e-01,
            2.08207846e-01,
            2.07564326e-01,
            2.06614245e-01,
            2.05513047e-01,
            2.04650072e-01,
            2.04144075e-01,
            2.03837360e-01,
            2.03616610e-01,
            2.03492726e-01,
            2.03395233e-01,
            2.03054854e-01,
            2.02382322e-01,
            2.01300762e-01,
            1.99973550e-01,
            1.98778425e-01,
            1.97536617e-01,
            1.95979261e-01,
            1.94146951e-01,
            1.92100195e-01,
            1.89966750e-01,
            1.87817115e-01,
            1.85532158e-01,
            1.83030698e-01,
            1.80304589e-01,
            1.77417073e-01,
            1.74498562e-01,
            1.71508117e-01,
            1.68381990e-01,
            1.65209752e-01,
            1.61974786e-01,
            1.58518654e-01,
            1.54779858e-01,
            1.51031021e-01,
            1.47691666e-01,
            1.44876871e-01,
            1.42402470e-01,
            1.39920094e-01,
            1.37162788e-01,
            1.34183054e-01,
            1.31054068e-01,
            1.27790665e-01,
            1.24630654e-01,
            1.21778405e-01,
            1.19254653e-01,
            1.17104573e-01,
            1.15096684e-01,
            1.12962164e-01,
            1.10842612e-01,
            1.08883176e-01,
            1.06968221e-01,
            1.04661154e-01,
            1.01824033e-01,
            9.89418397e-02,
            9.64304347e-02,
            9.46710066e-02,
            9.39117737e-02,
            9.38469396e-02,
            9.40487368e-02,
            9.43866219e-02,
            9.48185182e-02,
            9.53003697e-02,
            9.57380942e-02,
            9.60296019e-02,
            9.61804828e-02,
            9.63290903e-02,
            9.65213568e-02,
            9.67219239e-02,
            9.70062880e-02,
            9.72235765e-02,
            9.72773876e-02,
            9.72662114e-02,
            9.71502550e-02,
            9.69495702e-02,
            9.66428342e-02,
            9.60250070e-02,
            9.51530171e-02,
            9.39863485e-02,
            9.21311941e-02,
            8.97099328e-02,
            8.71368172e-02,
            8.45397795e-02,
            8.20160747e-02,
            7.93785001e-02,
            7.63952689e-02,
            7.34980662e-02,
            7.11524994e-02,
            6.93905170e-02,
            6.81501524e-02,
            6.70763353e-02,
            6.57724712e-02,
            6.44365155e-02,
            6.33916753e-02,
            6.25909487e-02,
            6.19280340e-02,
            6.13783625e-02,
            6.07176523e-02,
            5.95241259e-02,
            5.76278889e-02,
            5.49830107e-02,
            5.14437167e-02,
            4.72421310e-02,
            4.30572227e-02,
            3.95061588e-02,
            3.67182026e-02,
            3.45165997e-02,
            3.26178334e-02,
            3.05713740e-02,
            2.82668560e-02,
            2.57324963e-02,
            2.30361168e-02,
            2.06047808e-02,
            1.85838408e-02,
            1.68263362e-02,
            1.51827111e-02,
            1.37796280e-02,
            1.28851854e-02,
            1.22479296e-02,
            1.15386868e-02,
            1.07288266e-02,
            9.89940302e-03,
            9.23732312e-03,
            8.70628570e-03,
            8.14529883e-03,
            7.81562023e-03,
            7.68395969e-03,
            7.32180846e-03,
            6.94359341e-03,
            7.23152792e-03,
            8.47711416e-03,
            1.02076942e-02,
            1.18987005e-02,
            1.37089909e-02,
            1.59032001e-02,
            1.82413235e-02,
            2.05663826e-02,
            2.30630032e-02,
            2.58051748e-02,
            2.87914153e-02,
            3.18850177e-02,
            3.48360419e-02,
            3.76026887e-02,
            4.02091737e-02,
            4.22835871e-02,
            4.35081712e-02,
            4.43241257e-02,
            4.51754038e-02,
            4.59832464e-02,
            4.67861828e-02,
            4.76850395e-02,
            4.87618987e-02,
            5.00762645e-02,
            5.14626079e-02,
            5.28506271e-02,
            5.41628516e-02,
            5.51575950e-02,
            5.58471440e-02,
            5.65970961e-02,
            5.75120137e-02,
            5.82254908e-02,
            5.86099607e-02,
            5.92216107e-02,
            6.02151196e-02,
            6.11889321e-02,
            6.21343981e-02,
            6.32595613e-02,
            6.47289306e-02,
            6.64223728e-02,
            6.79439150e-02,
            6.90264256e-02,
            6.98257434e-02,
            7.05817699e-02,
            7.10636800e-02,
            7.12646425e-02,
            7.14301228e-02,
            7.13482186e-02,
            7.10371284e-02,
            7.06985279e-02,
            7.03140510e-02,
            6.99776637e-02,
            6.93664666e-02,
            6.82710835e-02,
            6.73946838e-02,
            6.70246227e-02,
            6.66819757e-02,
            6.61433732e-02,
            6.55781919e-02,
            6.52395413e-02,
            6.51562885e-02,
            6.52183359e-02,
            6.54706077e-02,
            6.57939947e-02,
            6.58886584e-02,
            6.59410863e-02,
            6.64075964e-02,
            6.72083727e-02,
            6.82543932e-02,
            6.98011833e-02,
            7.19336292e-02,
            7.43969075e-02,
            7.66245220e-02,
            7.83258991e-02,
            7.97883631e-02,
            8.13093202e-02,
            8.28800515e-02,
            8.42296331e-02,
            8.50587571e-02,
            8.55138111e-02,
            8.58330604e-02,
            8.62043201e-02,
            8.67342767e-02,
            8.71886809e-02,
            8.75951747e-02,
            8.80181024e-02,
            8.81689267e-02,
            8.79480976e-02,
            8.77224819e-02,
            8.76296703e-02,
            8.72525699e-02,
            8.62995708e-02,
            8.48415218e-02,
            8.29144810e-02,
            8.04846626e-02,
            7.79837613e-02,
            7.58715796e-02,
            7.39038792e-02,
            7.15294419e-02,
            6.87400268e-02,
            6.59815173e-02,
            6.34438454e-02,
            6.11047112e-02,
            5.89917985e-02,
            5.72022888e-02,
            5.58205743e-02,
            5.49345330e-02,
            5.43103330e-02,
            5.36070013e-02,
            5.31070782e-02,
            5.30879122e-02,
            5.33135188e-02,
            5.36645288e-02,
            5.43259361e-02,
            5.52271638e-02,
            5.60737486e-02,
            5.67523299e-02,
            5.70845397e-02,
            5.70801392e-02,
            5.71324354e-02,
            5.73061884e-02,
            5.75689697e-02,
            5.80356397e-02,
            5.84384408e-02,
            5.87627954e-02,
            5.94387084e-02,
            6.04149846e-02,
            6.14200402e-02,
            6.21133538e-02,
            6.22190647e-02,
            6.20483839e-02,
            6.17578750e-02,
            6.12506280e-02,
            6.07912498e-02,
            6.07443739e-02,
            6.11009579e-02,
            6.16345295e-02,
            6.24235409e-02,
            6.35894996e-02,
            6.48440569e-02,
            6.60126350e-02,
            6.71667170e-02,
            6.82103087e-02,
            6.91751445e-02,
            6.99360803e-02,
            6.99752243e-02,
            6.91767830e-02,
            6.80278590e-02,
            6.67143631e-02,
            6.52184697e-02,
            6.37662568e-02,
            6.23492114e-02,
            6.09196895e-02,
            5.95869570e-02,
            5.83014706e-02,
            5.70117814e-02,
            5.59407698e-02,
            5.52351876e-02,
            5.48566714e-02,
            5.46245237e-02,
            5.42493144e-02,
            5.36569101e-02,
            5.28320500e-02,
            5.17138472e-02,
            5.05579567e-02,
            4.95775635e-02,
            4.86527380e-02,
            4.77661560e-02,
            4.67941303e-02,
            4.54258987e-02,
            4.37601885e-02,
            4.21464321e-02,
            4.06912103e-02,
            3.93951258e-02,
            3.82896973e-02,
            3.73300236e-02,
            3.64791626e-02,
            3.59604730e-02,
            3.57458998e-02,
            3.53451166e-02,
            3.46613760e-02,
            3.39857570e-02,
            3.35154584e-02,
            3.34703015e-02,
            3.39548288e-02,
            3.45926742e-02,
            3.50017198e-02,
            3.52128592e-02,
            3.52931033e-02,
            3.53404887e-02,
            3.53332720e-02,
            3.50627586e-02,
            3.43864841e-02,
            3.35322062e-02,
            3.30742396e-02,
            3.30501837e-02,
            3.31641494e-02,
            3.32605069e-02,
            3.31896409e-02,
            3.28733984e-02,
            3.24504460e-02,
            3.21767390e-02,
            3.21315735e-02,
            3.23036002e-02,
            3.28034038e-02,
            3.36076634e-02,
            3.44154880e-02,
            3.53416417e-02,
            3.63868302e-02,
            3.72450370e-02,
            3.80544443e-02,
            3.88643664e-02,
            3.97621809e-02,
            4.11026172e-02,
            4.26180964e-02,
            4.37433687e-02,
            4.47369244e-02,
            4.59976645e-02,
            4.73383402e-02,
            4.85125734e-02,
        ];

        input_signal
            .iter()
            .zip(expected_result.into_iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-4, "{} {}", a, b));
    }
}
