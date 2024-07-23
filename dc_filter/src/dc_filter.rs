pub struct DCFilter {
    xn: f32,
    yn: f32,
    r: f32,
}

impl DCFilter {
    pub fn new() -> Self {
        DCFilter {
            xn: 0.0,
            yn: 0.0,
            r: 0.995,
        }
    }

    pub fn process(&mut self, input: f32) -> f32 {
        let this_output = input - self.xn + (self.r * self.yn);
        self.xn = input;
        self.yn = this_output;
        this_output
    }
}
