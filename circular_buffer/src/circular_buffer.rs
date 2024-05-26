#[cfg(target_arch = "aarch64")]
use apple_sys::Accelerate::cblas_sdot;

#[inline]
#[cfg(target_arch = "aarch64")]
fn dot_prod(buf: &[f32], kernel: &[f32], n: i32) -> f32 {
    let result;
    unsafe {
        result = cblas_sdot(n, buf.as_ptr(), 1, kernel.as_ptr(), 1);
    }
    result
}

#[cfg(not(target_arch = "aarch64"))]
fn dot_prod(buf: &[f32], kernel: &[f32]) -> f32 {
    buf.iter()
        .zip(kernel.iter())
        .fold(0.0, |acc, (b, k)| acc + (b * k))
}

#[derive(Debug)]
pub struct SizedCircularConvBuff<const K_SIZE: usize, const B_SIZE: usize> {
    buff: [f32; B_SIZE],
}

impl<const K_SIZE: usize, const B_SIZE: usize> SizedCircularConvBuff<K_SIZE, B_SIZE> {
    // const KERNEL_SIZE: usize = SIZE;
    const KERNEL_SIZE_I32: i32 = K_SIZE as i32;
    const NUM_CONV_BLOCKS: usize = 1;
    const BLOCK_SIZE: usize = K_SIZE * Self::NUM_CONV_BLOCKS;
    // const BUF_SIZE: usize = Self::BLOCK_SIZE + Self::KERNEL_SIZE;

    pub fn new() -> Self {
        SizedCircularConvBuff {
            buff: [0.0_f32; B_SIZE],
        }
    }

    pub fn convolve(&mut self, input: &mut [f32], kernel: &[f32; K_SIZE]) {
        // for i in 0..input.len() / kernel.len()
        // copy input[i * k..(i + 1) * k] -> buff[k..k * 2]
        // for j in 0..k
        // input[i+j] = buf[i + 1 .. i + 1 + k] dot kernel
        // buf[k..k * 2] -> buf[0..k]

        for i in 0..(input.len() / Self::BLOCK_SIZE) {
            self.buff[K_SIZE..]
                .iter_mut()
                .zip(
                    input
                        .iter()
                        .skip(i * Self::BLOCK_SIZE)
                        .take(Self::BLOCK_SIZE),
                )
                .for_each(|(b, i)| *b = *i);

            for j in 0..Self::BLOCK_SIZE {
                unsafe {
                    input[(i * Self::BLOCK_SIZE) + j] = cblas_sdot(
                        Self::KERNEL_SIZE_I32,
                        self.buff[j + 1..].as_ptr(),
                        1,
                        kernel.as_ptr(),
                        1,
                    );
                }
            }
            for j in 0..K_SIZE {
                self.buff[j] = self.buff[j + Self::BLOCK_SIZE];
            }
        }
    }
}

pub trait DelayBuffer {
    fn push(&mut self, val: f32);
    fn decrement_pos(&mut self);
    fn delay(&mut self, input: &mut [f32]);
    fn reset(&mut self);
}

// const SIZED_DELAY_32_SIZE: usize = 32;

#[derive(Debug)]
pub struct SizedDelayBuffer<const DELAY_LEN: usize> {
    data: [f32; DELAY_LEN],
    pos: usize,
}

impl<const DELAY_LEN: usize> SizedDelayBuffer<DELAY_LEN> {
    pub fn new(delay_len: usize) -> Self {
        assert!(delay_len <= DELAY_LEN && delay_len > 0);
        SizedDelayBuffer {
            data: [0.0_f32; DELAY_LEN],
            pos: 0,
        }
    }
}

impl<const DELAY_LEN: usize> DelayBuffer for SizedDelayBuffer<DELAY_LEN> {
    #[inline]
    fn push(&mut self, val: f32) {
        self.data[self.pos] = val;
    }

    #[inline]
    fn decrement_pos(&mut self) {
        self.pos = if self.pos == 0 {
            DELAY_LEN - 1
        } else {
            self.pos - 1
        };
    }

    #[cold]
    fn reset(&mut self) {
        self.data.iter_mut().for_each(|x| *x = 0.0_f32.into());
        self.pos = 0;
    }

    fn delay(&mut self, input: &mut [f32]) {
        input.iter_mut().for_each(|v| {
            self.push(*v);
            self.decrement_pos();
            *v = self.data[self.pos];
        })
    }
}

#[derive(Debug)]
pub struct CircularDelayBuffer {
    data: Vec<f32>,
    pos: usize,
    size: usize,
}

impl CircularDelayBuffer {
    pub fn new(initial_size: usize) -> Self {
        CircularDelayBuffer {
            data: vec![0.0_f32; initial_size],
            pos: 0,
            size: initial_size,
        }
    }
}

impl DelayBuffer for CircularDelayBuffer {
    #[inline]
    fn push(&mut self, val: f32) {
        self.data[self.pos] = val;
    }

    #[inline]
    fn decrement_pos(&mut self) {
        self.pos = if self.pos == 0 {
            self.size - 1
        } else {
            self.pos - 1
        };
    }

    /// Resets the buffer's data to all zeros and resets the buffers position value to zero
    #[cold]
    fn reset(&mut self) {
        self.data.iter_mut().for_each(|x| *x = 0.0_f32.into());
        self.pos = 0;
    }

    /// delays the input by self.size number of samples
    #[inline]
    fn delay(&mut self, input: &mut [f32]) {
        input.iter_mut().for_each(|v| {
            self.push(*v);
            self.decrement_pos();
            *v = self.data[self.pos];
        })
    }
}

#[derive(Debug)]
pub struct CircularConvBuffer {
    data: Vec<f32>,
    kernel: Vec<f32>,
    pos: usize,
    size: usize,
}

impl CircularConvBuffer {
    pub fn new(initial_size: usize, kernel: &[f32]) -> Self {
        CircularConvBuffer {
            data: vec![0.0_f32; initial_size],
            pos: 0,
            size: initial_size,
            kernel: kernel.iter().rev().map(|v| *v).collect::<Vec<f32>>(),
        }
    }

    /// Resets the buffer's data to all zeros and resets the buffers position value to zero
    #[cold]
    pub fn reset(&mut self) {
        self.data.iter_mut().for_each(|x| *x = 0.0_f32.into());
        self.pos = 0;
    }

    #[inline]
    fn push(&mut self, val: f32) {
        self.data[self.pos] = val;
    }

    #[inline]
    fn increment_pos(&mut self) {
        self.pos = if self.pos == self.size - 1 {
            0
        } else {
            self.pos + 1
        };
    }

    /// Convolves the input signal with the buffer's interal kernel
    #[inline]
    pub fn convolve(&mut self, input: &mut [f32]) {
        input.iter_mut().for_each(|s| {
            self.push(*s);
            self.increment_pos();
            let o_pos = self.size - self.pos;
            *s = dot_prod(&self.data[self.pos..], &self.kernel[..o_pos], o_pos as i32)
                + dot_prod(
                    &self.data[..self.pos],
                    &self.kernel[o_pos..],
                    self.pos as i32,
                );
        })
    }
}

#[cfg(test)]
mod tests {

    // use crate::circular_buffer;
    use super::*;

    #[test]
    fn create_f32() {
        let new = CircularDelayBuffer::new(1);
        assert_eq!(new.pos, 0);
        assert_eq!(new.size, 1);
        assert_eq!(new.data, vec![0.0]);

        let new_conv = CircularConvBuffer::new(1, &[1.]);
        assert_eq!(new_conv.pos, 0);
        assert_eq!(new_conv.size, 1);
        assert_eq!(new_conv.data, vec![0.0]);
    }

    #[test]
    fn push_sucess() {
        let mut new = CircularDelayBuffer::new(1);
        new.push(1.);
        assert_eq!(new.data[0], 1.);
    }

    #[test]
    fn ptr_dec() {
        let mut new = CircularDelayBuffer::new(2);
        assert_eq!(new.pos, 0);
        new.decrement_pos();
        assert_eq!(new.pos, 1);
        new.decrement_pos();
        assert_eq!(new.pos, 0);
    }

    #[test]
    fn test_conv_01234_012() {
        let mut signal = vec![0., 1., 2., 3., 4., 0., 0.];
        let mut buf = CircularConvBuffer::new(3, &[0., 1., 2.]);

        buf.convolve(&mut signal);
        assert_eq!(signal, vec![0., 0., 1., 4., 7., 10., 8.])
    }

    const ERR_TOL: f32 = 1e-7;

    #[test]
    fn conv_sin_filter() {
        let mut sig: Vec<f32> = (0..30)
            .map(|x| (((x as f32) * std::f32::consts::PI * 2.0) / 44100.0).sin())
            .collect();

        let coefs: [f32; 10] = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let kernel = &coefs.into_iter().collect::<Vec<f32>>();

        let mut buf = CircularConvBuffer::new(10, kernel);

        buf.convolve(&mut sig);

        let expected_result = vec![
            0.00000000e+00,
            0.00000000e+00,
            1.42475857e-04,
            5.69903424e-04,
            1.42475855e-03,
            2.84951708e-03,
            4.98665483e-03,
            7.97864762e-03,
            1.19679712e-02,
            1.70971015e-02,
            2.35085141e-02,
            2.99199262e-02,
            3.63313377e-02,
            4.27427485e-02,
            4.91541584e-02,
            5.55655673e-02,
            6.19769751e-02,
            6.83883817e-02,
            7.47997868e-02,
            8.12111904e-02,
            8.76225924e-02,
            9.40339926e-02,
            1.00445391e-01,
            1.06856787e-01,
            1.13268181e-01,
            1.19679573e-01,
            1.26090962e-01,
            1.32502349e-01,
            1.38913733e-01,
            1.45325114e-01,
        ];

        sig.iter()
            .zip(expected_result.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < ERR_TOL, "result: {}, expected: {}", a, b));
    }

    #[test]
    fn delay_5_samples() {
        let mut sig: Vec<f32> = (1..10).map(|x| x as f32).collect();
        let mut delay_buf = CircularDelayBuffer::new(5);

        delay_buf.delay(&mut sig);
        let expected_result = vec![0., 0., 0., 0., 1., 2., 3., 4., 5.];
        dbg!(&sig);
        check_results(&sig, &expected_result);
    }

    #[test]
    fn delay_5_samples_sized() {
        let mut sig: Vec<f32> = (1..10).map(|x| x as f32).collect();
        let mut delay_buf: SizedDelayBuffer<5> = SizedDelayBuffer::new(5);

        delay_buf.delay(&mut sig);
        let expected_result = vec![0., 0., 0., 0., 1., 2., 3., 4., 5.];
        dbg!(&sig);
        check_results(&sig, &expected_result);
    }

    fn check_results(result: &[f32], expected: &[f32]) {
        result.iter().zip(expected.iter()).for_each(|(a, b)| {
            assert!((a - b).abs() < ERR_TOL, "result: {}, expected: {}", a, b);
        })
    }

    #[test]
    fn delay_9_samples() {
        let mut sig: Vec<f32> = (1..10).map(|x| x as f32).collect();
        let mut delay_buf = CircularDelayBuffer::new(9);

        delay_buf.delay(&mut sig);
        let expected_result = vec![0., 0., 0., 0., 0., 0., 0., 0., 1.];
        dbg!(&sig);
        check_results(&sig, &expected_result);
    }

    #[test]
    fn delay_over_2_blocks() {
        let mut sig_1: Vec<f32> = (1..9).map(|x| x as f32).collect();
        let mut sig_2: Vec<f32> = (1..9).map(|_| 0.0_f32).collect();
        let mut delay_buf = CircularDelayBuffer::new(4);

        dbg!(&sig_1);
        delay_buf.delay(&mut sig_1);
        delay_buf.delay(&mut sig_2);
        let expected_result_1 = vec![0., 0., 0., 1., 2., 3., 4., 5.];
        let expected_result_2 = vec![6., 7., 8., 0., 0., 0., 0., 0.];
        dbg!(&sig_1);
        dbg!(&delay_buf.data);
        check_results(&sig_1, &expected_result_1);
        check_results(&sig_2, &expected_result_2);
    }

    #[test]
    fn test_convolve_filter_taps() {
        let filter_taps = vec![
            -0.0064715474097890545,
            0.006788724784527351,
            -0.007134125572070907,
            0.007511871271766723,
            -0.007926929217098087,
            0.00838534118242672,
            -0.00889453036904902,
            0.009463720022395613,
            -0.010104514094437885,
            0.010831718180021,
            -0.011664525313602769,
            0.012628270948224513,
            -0.013757103575462731,
            0.015098181413680897,
            -0.01671851963595936,
            0.01871667093508393,
            -0.021243750540180146,
            0.024543868940610197,
            -0.0290386730354654,
            0.035524608815134716,
            -0.045708348639099484,
            0.06402724397938601,
            -0.10675158913607562,
            0.32031404953367254,
            0.32031404953367254,
            -0.10675158913607562,
            0.06402724397938601,
            -0.045708348639099484,
            0.035524608815134716,
            -0.0290386730354654,
            0.024543868940610197,
            -0.021243750540180146,
            0.01871667093508393,
            -0.01671851963595936,
            0.015098181413680897,
            -0.013757103575462731,
            0.012628270948224513,
            -0.011664525313602769,
            0.010831718180021,
            -0.010104514094437885,
            0.009463720022395613,
            -0.00889453036904902,
            0.00838534118242672,
            -0.007926929217098087,
            0.007511871271766723,
            -0.007134125572070907,
            0.006788724784527351,
            -0.0064715474097890545,
        ];

        let expected_result = vec![
            0.0,
            -0.0064715474097890545,
            -0.006154370035050758,
            -0.012971318232383369,
            0.013609794481206961,
            -0.014305563389777363,
            0.015067096563530714,
            -0.015904635655489836,
            0.016830682831577737,
            -0.01786066515679372,
            0.019013850058332067,
            -0.020314631236874423,
            0.021794374861081975,
            -0.02349413761982201,
            0.02546878710742898,
            -0.027793467534985756,
            0.030574175904207905,
            -0.03396596757789036,
            0.0382063806655017,
            -0.04368218677478544,
            0.05107886956603451,
            -0.06177515011522625,
            0.07918437314659119,
            -0.11582214709460205,
            0.29889260319967936,
            0.6406873811927908,
            1.4948186585322873,
            0.8114662143082524,
            -0.23790862808855429,
            0.13618964347509377,
            -0.09511450132249442,
            0.07304034931508355,
            -0.05927203176535595,
            0.04986077667655423,
            -0.043016429386331934,
            0.037811154947013974,
            -0.03371629965597902,
            0.030408608038341743,
            -0.027679294143541935,
            0.025387480397489004,
            -0.02343465367520419,
            0.02174984637358284,
            -0.02028063260757145,
            0.01898744051151552,
            -0.01783983795939171,
            0.01681403638485071,
            -0.015891170679831722,
            0.015056087455685704,
            -0.014296474556947074,
            0.007423079534003944,
            -0.019414642229367163,
        ];

        let mut sig = vec![vec![0., 1., 2., 3.], vec![0.; expected_result.len() - 4]]
            .into_iter()
            .flatten()
            .collect::<Vec<f32>>();

        let mut buff = CircularConvBuffer::new(filter_taps.len(), &filter_taps);

        buff.convolve(&mut sig);

        check_results(&sig, &expected_result)
    }

    #[test]
    fn doc_t() {
        let mut buf = CircularConvBuffer::new(4, &[0., 1., 2., 4.]);
        let input_signal = &mut [1., 0., 0., 0., 0.];
        buf.convolve(input_signal);
        dbg!(&input_signal);
    }

    #[test]
    fn test_cblas_conv() {
        let kernel: [f32; 32] = [
            -0.00116417,
            -0.00139094,
            0.00195951,
            0.00293134,
            -0.00437535,
            -0.00637313,
            0.00902803,
            0.01248116,
            -0.0169409,
            -0.02273977,
            0.03045372,
            0.04118039,
            -0.05729369,
            -0.08500841,
            0.14720004,
            0.45005217,
            0.45005217,
            0.14720004,
            -0.08500841,
            -0.05729369,
            0.04118039,
            0.03045372,
            -0.02273977,
            -0.0169409,
            0.01248116,
            0.00902803,
            -0.00637313,
            -0.00437535,
            0.00293134,
            0.00195951,
            -0.00139094,
            -0.00116417,
        ];

        let mut kernel_reversed: [f32; 32] = [0.0_f32; 32];
        kernel_reversed
            .iter_mut()
            .zip(kernel.iter().rev())
            .for_each(|(kr, k)| *kr = *k);
        // kernel = kernel.into_iter().rev().collect::<Vec<_>>();
        // let mut kernel_slice: [f32; 32] = [0.0_f32; 32];
        // kernel_slice
        // .iter_mut()
        // .zip(kernel.iter())
        // .for_each(|(ks, k)| *ks = *k);

        let sig: &mut [f32] = &mut [
            0.39859362,
            0.41525508,
            0.94807649,
            0.51561058,
            1.22882384,
            0.23256073,
            -0.1342023,
            1.74006643,
            0.19159666,
            -1.40812097,
            1.2478925,
            -0.73142635,
            -0.06287913,
            0.92967597,
            0.15283479,
            -1.1140269,
            0.84175889,
            0.39103447,
            0.98671906,
            0.5805405,
            -0.16588464,
            -0.79760456,
            -1.30265408,
            0.38652751,
            -0.51993827,
            -0.42709767,
            0.14080621,
            -0.9849361,
            0.62268698,
            -0.68781029,
            -1.58141413,
            1.13639306,
            0.69474109,
            -0.3400894,
            -0.04647407,
            -0.89640212,
            0.3764416,
            -0.02898992,
            1.32725035,
            2.67974568,
            0.68779525,
            0.43760831,
            -1.01003035,
            -0.30868521,
            -1.05419557,
            2.22675651,
            0.22789262,
            -0.48480561,
            0.46329582,
            1.07014692,
            -0.40205107,
            -0.42221082,
            1.10952191,
            -1.22632506,
            -0.08106437,
            0.92003661,
            -0.41518023,
            -0.14786155,
            1.65865399,
            -2.01471316,
            0.54427785,
            1.24392378,
            1.05165889,
            -1.04828755,
        ];

        let expected_result = [
            -4.64031751e-04,
            -1.03784711e-03,
            -9.00270770e-04,
            6.31333574e-05,
            -8.16714618e-04,
            -2.54765879e-03,
            5.55948925e-04,
            2.64438161e-03,
            -3.89762700e-03,
            -4.07059136e-03,
            9.25090060e-03,
            6.36314411e-03,
            -2.02351028e-02,
            -1.46096150e-02,
            4.79181119e-02,
            1.49284784e-01,
            3.56493520e-01,
            6.99263472e-01,
            9.23551763e-01,
            7.74168455e-01,
            4.83552462e-01,
            4.81523463e-01,
            6.77110344e-01,
            5.67521062e-01,
            5.88821361e-02,
            -3.44865832e-01,
            -2.38717901e-01,
            1.84915808e-01,
            4.12825452e-01,
            1.96207624e-01,
            -1.83879935e-01,
            -1.96039522e-01,
            3.40831973e-01,
            9.48063627e-01,
            9.30666351e-01,
            1.61849035e-01,
            -7.07922707e-01,
            -9.22764568e-01,
            -4.92119999e-01,
            -1.50921305e-01,
            -2.85332635e-01,
            -4.12673636e-01,
            -1.72582522e-01,
            -8.17550769e-02,
            -5.12248098e-01,
            -7.69204960e-01,
            -1.99605847e-01,
            5.63479634e-01,
            5.32488348e-01,
            -1.48131605e-01,
            -5.98218036e-01,
            -4.42816637e-01,
            1.33373255e-01,
            9.62953413e-01,
            1.76377555e+00,
            1.86476408e+00,
            8.59810067e-01,
            -5.52330988e-01,
            -1.06726057e+00,
            -3.03666286e-01,
            6.84704082e-01,
            8.02141166e-01,
            2.84435069e-01,
            7.68219493e-02,
        ];

        let mut buf: SizedCircularConvBuff<32, 64> = SizedCircularConvBuff::new();

        buf.convolve(sig, &kernel_reversed);
        assert_eq!(sig.len(), expected_result.len());

        check_results(sig, &expected_result);
    }

    /*
    #[test]
    fn test_cblas_conv_small() {
        let kernel: [f32; 32] = [
            -0.00116417,
            -0.00139094,
            0.00195951,
            0.00293134,
            -0.00437535,
            -0.00637313,
            0.00902803,
            0.01248116,
            -0.0169409,
            -0.02273977,
            0.03045372,
            0.04118039,
            -0.05729369,
            -0.08500841,
            0.14720004,
            0.45005217,
            0.45005217,
            0.14720004,
            -0.08500841,
            -0.05729369,
            0.04118039,
            0.03045372,
            -0.02273977,
            -0.0169409,
            0.01248116,
            0.00902803,
            -0.00637313,
            -0.00437535,
            0.00293134,
            0.00195951,
            -0.00139094,
            -0.00116417,
        ];

        let mut kernel_reversed: [f32; 32] = [0.0_f32; 32];
        kernel_reversed
            .iter_mut()
            .zip(kernel.iter().rev())
            .for_each(|(kr, k)| *kr = *k);

        let sig: &mut [f32] = &mut [
            -0.3600947,
            1.25699783,
            0.26974943,
            -0.14809912,
            -1.27555489,
            -0.48406062,
            -0.68576611,
            0.91506295,
        ];

        let expected_result = &[
            0.00041921,
            -0.00096249,
            -0.00276805,
            0.00120475,
            0.00747976,
            -0.0003666,
            -0.01390415,
            0.00098361,
        ];

        let mut buf = CircularConvBuffer32::new();

        buf.convolve(sig, &kernel_reversed);
        assert_eq!(sig.len(), expected_result.len());

        dbg!(&sig);
        check_results(sig, expected_result);
    }
    */
}
