#[allow(dead_code)]
pub trait Env {
    fn consume(&mut self) -> f32;
    fn target_reached(&self) -> bool;
}

#[derive(Debug, Clone)]
pub struct LinearEnvelope {
    current_value: f32,
    target_value: f32,
    num_steps: i32,
    step_size: f32,
}

#[allow(dead_code)]
impl LinearEnvelope {
    pub fn new(start: f32, end: f32, steps: i32) -> Self {
        LinearEnvelope {
            current_value: start,
            target_value: end,
            num_steps: steps,
            step_size: (end - start) / (steps as f32),
        }
    }

    pub fn fade_in(steps: i32) -> Self {
        LinearEnvelope {
            current_value: 0.0,
            target_value: 1.0,
            num_steps: steps,
            step_size: 1.0 / (steps as f32),
        }
    }

    pub fn fade_out(steps: i32) -> Self {
        LinearEnvelope {
            current_value: 1.0,
            target_value: 0.0,
            num_steps: steps,
            step_size: -1.0 / (steps as f32),
        }
    }
}

#[allow(dead_code)]
impl Env for LinearEnvelope {
    fn consume(&mut self) -> f32 {
        assert!(self.num_steps >= 0);
        if self.num_steps > 0 {
            self.current_value += self.step_size;
            self.num_steps -= 1;
            self.current_value
        } else {
            self.target_value
        }
    }

    fn target_reached(&self) -> bool {
        self.current_value == self.target_value
    }
}

#[derive(Debug)]
#[allow(dead_code)]
struct ExponentialEnvelope {
    start_value: f32,
    current_value: f32,
    target_value: f32,
    tot_steps: i32,
    curr_step: i32,
    z: f32,
    delta: f32,
}

#[allow(dead_code)]
impl ExponentialEnvelope {
    fn new(start: f32, end: f32, steps: i32, curve: f32) -> Self {
        ExponentialEnvelope {
            start_value: start,
            current_value: start,
            target_value: end,
            tot_steps: steps,
            curr_step: 0,
            z: curve,
            delta: end - start,
        }
    }
}

impl Env for ExponentialEnvelope {
    fn consume(&mut self) -> f32 {
        assert!(self.curr_step >= 0);
        if self.curr_step <= self.tot_steps {
            self.current_value = self.delta
                * (self.curr_step as f32 / (self.tot_steps - 1) as f32).powf(self.z)
                + self.start_value;
            self.curr_step += 1;
            self.current_value
        } else {
            self.target_value
        }
    }

    fn target_reached(&self) -> bool {
        self.current_value == self.target_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn basic_envelope_test() {
        let mut env = LinearEnvelope::new(1.0, 0.0, 10);
        dbg!(&env);
        let result = (0..10)
            .into_iter()
            .map(|_| env.consume())
            .collect::<Vec<_>>();
        let expected_result: Vec<f32> = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0];

        result
            .into_iter()
            .zip(expected_result)
            .for_each(|(r, e)| assert_approx_eq!(f32, r, e));
        // assert_approx_eq!(env.consume(), 0.0);
    }

    #[test]
    fn create_basic_envelope_test() {
        let mut env = LinearEnvelope::fade_out(10);
        dbg!(&env);
        let result = (0..10)
            .into_iter()
            .map(|_| env.consume())
            .collect::<Vec<_>>();
        let expected_result: Vec<f32> = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0];

        result
            .into_iter()
            .zip(expected_result)
            .for_each(|(r, e)| assert_approx_eq!(f32, r, e));
        // assert_approx_eq!(env.consume(), 0.0);
    }

    #[test]
    fn create_basic_envelope_up() {
        let mut env = LinearEnvelope::fade_in(10);
        let result = (0..10)
            .into_iter()
            .map(|_| env.consume())
            .collect::<Vec<_>>();
        let expected_result = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        result
            .into_iter()
            .zip(expected_result)
            .for_each(|(r, e)| assert_approx_eq!(f32, r, e));
    }

    #[test]
    fn basic_envelope_up() {
        let mut env = LinearEnvelope::new(0.0, 1.0, 10);
        let result = (0..10)
            .into_iter()
            .map(|_| env.consume())
            .collect::<Vec<_>>();
        let expected_result = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        result
            .into_iter()
            .zip(expected_result)
            .for_each(|(r, e)| assert_approx_eq!(f32, r, e));
    }

    #[test]
    fn basic_exp_up() {
        let mut env = ExponentialEnvelope::new(0.0, 1.0, 10, 2.0);
        let result = (0..10)
            .into_iter()
            .map(|_| env.consume())
            .collect::<Vec<_>>();
        let expected_result = vec![
            0., 0.01234568, 0.04938272, 0.11111111, 0.19753086, 0.30864198, 0.44444444, 0.60493827,
            0.79012346, 1.,
        ];

        result
            .into_iter()
            .zip(expected_result)
            .for_each(|(r, e)| assert_approx_eq!(f32, r, e));
    }

    #[test]
    fn basic_exp_down() {
        let mut env = ExponentialEnvelope::new(1.0, 0.0, 10, 2.0);
        let result = (0..10)
            .into_iter()
            .map(|_| env.consume())
            .collect::<Vec<_>>();
        let expected_result = &[
            1., 0.98765432, 0.95061728, 0.88888889, 0.80246914, 0.69135802, 0.55555556, 0.39506173,
            0.20987654, 0.,
        ];

        result
            .into_iter()
            .zip(expected_result)
            .for_each(|(r, e)| assert_approx_eq!(f32, r, *e));
    }
}
