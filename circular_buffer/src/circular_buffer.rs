use std::collections::VecDeque;
use std::ptr;

#[cfg(target_arch = "aarch64")]
use apple_sys::Accelerate::cblas_sdot;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[inline]
#[cfg(target_arch = "aarch64")]
fn dot_prod(buf: &[f32], kernel: &[f32], n: i32) -> f32 {
    unsafe { cblas_sdot(n, buf.as_ptr(), 1, kernel.as_ptr(), 1) }
}

#[cfg(not(target_arch = "aarch64"))]
fn dot_prod(buf: &[f32], kernel: &[f32]) -> f32 {
    buf.iter()
        .zip(kernel.iter())
        .fold(0.0, |acc, (b, k)| acc + (b * k))
}

#[derive(Debug)]
pub struct TiledConv {
    buffer: Vec<f32>,
    k_len: usize,
    i_len: usize,
}

impl TiledConv {
    pub fn new(k_len: usize, i_len: usize) -> Self {
        TiledConv {
            buffer: vec![0.0_f32; k_len + i_len - 1],
            k_len,
            i_len,
        }
    }

    pub fn convolve(&mut self, input: &mut [f32], kernel: &[f32]) {
        Self::fast_copy(input, &mut self.buffer[self.k_len - 1..]);
        for i in 0..self.i_len {
            unsafe {
                input[i] = Self::neon_dot_product(&self.buffer[i..i + self.k_len], kernel);
            }
        }
        for i in 0..self.k_len - 1 {
            self.buffer[i] = self.buffer[self.i_len + i];
        }
    }

    #[inline]
    unsafe fn neon_dot_product(a: &[f32], b: &[f32]) -> f32 {
        assert!(a.len() == b.len());
        let mut sum = vdupq_n_f32(0.0);
        let mut result = 0.0;

        for (chunk_a, chunk_b) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
            let a_vec = vld1q_f32(chunk_a.as_ptr());
            let b_vec = vld1q_f32(chunk_b.as_ptr());
            sum = vmlaq_f32(sum, a_vec, b_vec);
        }

        let a_remain = a.chunks_exact(4).remainder();
        let b_remain = b.chunks_exact(4).remainder();

        result += vaddvq_f32(sum);

        for (aa, bb) in a_remain.iter().zip(b_remain.iter()) {
            result += aa * bb;
        }

        result
    }

    #[inline]
    fn fast_copy(src: &[f32], dst: &mut [f32]) {
        assert!(src.len() <= dst.len());
        unsafe {
            ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), src.len());
        }
    }
}

#[derive(Debug)]
struct Delay<I>
where
    I: Iterator,
{
    iter: Option<I>,
    buffer: VecDeque<I::Item>,
    delay: usize,
}

// impl<I> Delay<I> where I: Iterator

#[derive(Debug)]
pub struct CircularConvBuffer {
    buff: Vec<f32>,
    block_size: usize,
    k_size: usize,
    k_size_i32: i32,
}

impl CircularConvBuffer {
    // const KERNEL_SIZE: usize = SIZE;
    // const KERNEL_SIZE_I32: i32 = K_SIZE as i32;
    // const NUM_CONV_BLOCKS: usize = 1;
    // const BLOCK_SIZE: usize = K_SIZE * Self::NUM_CONV_BLOCKS;
    // const BUF_SIZE: usize = Self::BLOCK_SIZE + Self::KERNEL_SIZE;

    pub fn new(new_k_size: usize) -> Self {
        // assert_eq!((buf_partitions * K_SIZE) % B_SIZE, 0);
        CircularConvBuffer {
            buff: vec![0.0_f32; new_k_size * 2],
            block_size: new_k_size,
            k_size: new_k_size,
            k_size_i32: new_k_size as i32,
        }
    }

    pub fn convolve(&mut self, input: &mut [f32], kernel: &[f32]) {
        // for i in 0..input.len() / kernel.len()
        // copy input[i * k..(i + 1) * k] -> buff[k..k * 2]
        // for j in 0..k
        // input[i+j] = buf[i + 1 .. i + 1 + k] dot kernel
        // buf[k..k * 2] -> buf[0..k]

        for i in 0..(input.len() / self.block_size) {
            self.buff[self.k_size..]
                .iter_mut()
                .zip(input.iter().skip(i * self.block_size).take(self.block_size))
                .for_each(|(b, i)| *b = *i);

            for j in 0..self.block_size {
                input[(i * self.block_size) + j] =
                    dot_prod(&self.buff[j + 1..], kernel, self.k_size_i32);
            }
            for j in 0..self.k_size {
                self.buff[j] = self.buff[j + self.block_size];
            }
        }
    }
}

#[derive(Debug)]
pub struct SizedCircularConvBuff<const K_SIZE: usize, const B_SIZE: usize> {
    buff: [f32; B_SIZE],
    block_size: usize,
}

impl<const K_SIZE: usize, const B_SIZE: usize> SizedCircularConvBuff<K_SIZE, B_SIZE> {
    // const KERNEL_SIZE: usize = SIZE;
    const KERNEL_SIZE_I32: i32 = K_SIZE as i32;
    // const NUM_CONV_BLOCKS: usize = 1;
    // const BLOCK_SIZE: usize = K_SIZE * Self::NUM_CONV_BLOCKS;
    // const BUF_SIZE: usize = Self::BLOCK_SIZE + Self::KERNEL_SIZE;

    pub fn new(buf_partitions: usize) -> Self {
        // assert_eq!((buf_partitions * K_SIZE) % B_SIZE, 0);
        SizedCircularConvBuff {
            buff: [0.0_f32; B_SIZE],
            block_size: K_SIZE * buf_partitions,
        }
    }

    pub fn set_num_partitions(&mut self, buf_partitions: usize) {
        self.block_size = K_SIZE * buf_partitions;
    }

    pub fn convolve(&mut self, input: &mut [f32], kernel: &[f32; K_SIZE]) {
        // for i in 0..input.len() / kernel.len()
        // copy input[i * k..(i + 1) * k] -> buff[k..k * 2]
        // for j in 0..k
        // input[i+j] = buf[i + 1 .. i + 1 + k] dot kernel
        // buf[k..k * 2] -> buf[0..k]

        for i in 0..(input.len() / self.block_size) {
            self.buff[K_SIZE..]
                .iter_mut()
                .zip(input.iter().skip(i * self.block_size).take(self.block_size))
                .for_each(|(b, i)| *b = *i);

            for j in 0..self.block_size {
                unsafe {
                    input[(i * self.block_size) + j] = cblas_sdot(
                        Self::KERNEL_SIZE_I32,
                        self.buff[j + 1..].as_ptr(),
                        1,
                        kernel.as_ptr(),
                        1,
                    );
                }
            }
            for j in 0..K_SIZE {
                self.buff[j] = self.buff[j + self.block_size];
            }
        }
    }
}

// const SIZED_DELAY_32_SIZE: usize = 32;
#[derive(Debug)]
pub struct SizedDelayBuffer<const MAX_DELAY_LEN: usize> {
    data: [f32; MAX_DELAY_LEN],
    num_samples_delay: usize,
    pos: usize,
}

impl<const MAX_DELAY_LEN: usize> SizedDelayBuffer<MAX_DELAY_LEN> {
    pub fn new(delay_len: usize) -> Self {
        assert!(delay_len <= MAX_DELAY_LEN && delay_len > 0);
        SizedDelayBuffer {
            data: [0.0_f32; MAX_DELAY_LEN],
            num_samples_delay: delay_len,
            pos: 0,
        }
    }

    #[inline]
    fn push(&mut self, val: f32) {
        self.data[self.pos] = val;
    }

    #[inline]
    fn decrement_pos(&mut self) {
        self.pos = if self.pos == 0 {
            self.num_samples_delay - 1
        } else {
            self.pos - 1
        };
    }

    #[cold]
    pub fn reset(&mut self) {
        self.data.iter_mut().for_each(|x| *x = 0.0_f32.into());
        self.pos = 0;
    }

    pub fn delay(&mut self, input: &mut [f32]) {
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
    pub fn delay(&mut self, input: &mut [f32]) {
        input.iter_mut().for_each(|v| {
            self.push(*v);
            self.decrement_pos();
            *v = self.data[self.pos];
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

        let new_conv = CircularConvBuffer::new(1);
        assert_eq!(new_conv.buff, vec![0.0, 0.0]);
        assert_eq!(new_conv.block_size, 1);
        assert_eq!(new_conv.k_size, 1);
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
        let mut signal1 = vec![0., 1., 2.];
        let mut signal2 = vec![3., 4., 0.];
        let mut signal3 = vec![0., 0., 0.];
        let mut buf = TiledConv::new(3, 3);
        let kernel = vec![2., 1., 0.];
        let mut k = [0.0_f32; 3];
        k.copy_from_slice(&kernel);

        buf.convolve(&mut signal1, &k);
        buf.convolve(&mut signal2, &k);
        buf.convolve(&mut signal3, &k);
        // dbg!(&signal[..3]);
        // dbg!(&signal[3..6]);
        // dbg!(&signal[6..]);
        let mut result: Vec<f32> = Vec::with_capacity(7);

        signal1.iter().for_each(|a| {
            result.push(*a);
        });

        signal2.iter().for_each(|a| {
            result.push(*a);
        });

        signal3.iter().for_each(|a| {
            result.push(*a);
        });

        dbg!(&result);
        dbg!(&buf);

        assert_eq!(result, vec![0., 0., 1., 4., 7., 10., 8., 0., 0.])
    }

    const ERR_TOL: f32 = 1e-5;

    #[test]
    fn conv_sin_filter() {
        let mut sig: Vec<f32> = (0..30)
            .map(|x| (((x as f32) * std::f32::consts::PI * 2.0) / 44100.0).sin())
            .collect();

        let coefs: [f32; 10] = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let kernel = &coefs.into_iter().rev().collect::<Vec<f32>>();

        let mut buf = TiledConv::new(10, 30);

        buf.convolve(&mut sig, &kernel);

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
    fn test_random_32_48() {
        let mut input: &mut [f32] = &mut [
            0.33110049,
            1.21255977,
            -0.63855535,
            0.87021457,
            0.28245597,
            0.30258054,
            -0.16397546,
            0.91792172,
            -1.03488682,
            -0.28097881,
            0.9075962,
            -0.08794321,
            -1.03246297,
            1.20534872,
            -0.49227481,
            -0.6730221,
            0.07807707,
            1.30280194,
            1.93008999,
            -0.54959872,
            -1.25225005,
            1.48116167,
            1.57475551,
            0.61098639,
            -0.21768917,
            1.85008542,
            -0.53402873,
            -1.03331307,
            0.77831737,
            -0.45047843,
            1.59660967,
            -1.04902077,
        ];

        let kernel = vec![
            -0.51786801,
            0.12806374,
            0.12162219,
            -1.41806002,
            0.44641012,
            -0.20035058,
            0.60430911,
            0.16673836,
            -0.65460348,
            0.85289387,
            -0.44087577,
            1.29083681,
            -0.33657188,
            0.47837313,
            1.50228393,
            2.28960102,
            -0.5757445,
            -0.9422924,
            1.0910025,
            -0.66021472,
            0.34913295,
            -1.21597745,
            -1.50175691,
            -0.07521028,
            0.6329468,
            -1.21769653,
            0.91466479,
            -0.6157067,
            -0.35774266,
            2.58885707,
            1.28951167,
            -0.87187175,
            0.6751307,
            -0.02451999,
            -0.10609072,
            0.38177321,
            -0.57090344,
            -0.41314233,
            0.06315846,
            -0.15274474,
            0.88008929,
            -0.07743066,
            0.06828122,
            1.11188539,
            -0.25436785,
            -0.58250143,
            -2.31532648,
            1.90497062,
        ];

        let expected_result = vec![
            -0.17146635,
            -0.58554395,
            0.52624149,
            -0.85447827,
            -1.6841704,
            1.36578557,
            -1.40390391,
            0.44427496,
            -0.24423406,
            0.3436846,
            -0.44535399,
            1.04624509,
            2.63032095,
            -2.85969501,
            3.05411655,
            2.80455143,
            2.03336768,
            -1.65197954,
            2.04173485,
            2.47249761,
            -2.25375619,
            -2.55896367,
            -0.31586582,
            -3.15774946,
            0.8601969,
            -1.2072917,
            -7.37268089,
            4.46410864,
            -1.12207614,
            1.46065619,
            1.92950953,
            4.64211027,
        ];

        let mut buff = TiledConv::new(kernel.len(), input.len());

        buff.convolve(&mut input, &kernel.into_iter().rev().collect::<Vec<f32>>());

        dbg!(&input);
        check_results(&input, &expected_result)
    }

    #[test]
    fn doc_t() {
        let mut buf = CircularConvBuffer::new(4);
        let input_signal = &mut [1., 0., 0., 0., 0.];
        buf.convolve(input_signal, &[1., 2., 3., 4.]);
        dbg!(&input_signal);
    }

    /*
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

        let mut buf: Cir<32, 64> = SizedCircularConvBuff::new();

        buf.convolve(sig, &kernel_reversed);
        assert_eq!(sig.len(), expected_result.len());

        check_results(sig, &expected_result);
    }
    */
}
