# JDSP
Collection of audio processing tools written in Rust for use in audio plug-in development with Nih-Plug.

## Features
- adaa_nl

Nonlinear waveshaper with anti-derivative anti-aliasing

- circular_buffer

Circular buffer implementation with advanced portable SIMD convolution implementation for improved FIR filtering

- dc_filter 
- envelope

Linear envelope generator

- iir_biquad_filter

IIR biquad filter implementation  

- oversampler

2, 4, 8, or 16 times variable FIR halfband polyphase oversampling 

- window

Sinc, Hann, and Kaiser window impelementations

## Installation Instructions
- add to Cargo.toml file
```Rust
[dependencies]
jdsp = { git = "https://github.com/jsquie/jdsp.git", features = ["all"] }
```

## Usage Instructions
- import and use in your own project
```Rust
use jdsp::IIRBiquadFilter;

// declare
struct Filter {
    filters: [IIRBiquadFilter; 2],
}

// initialize
self.filters = [IIRBiquadFilter::default(), IIRBiquadFilter::default()];

// prefilter processing
for (l, r) in left.iter_mut().zip(right.iter_mut()) {
    let param_filter_cutoff: &Smoother<f32> =
        &self.params.filter_cutoff.smoothed;

    // recalculate every coefficient while smoothing
    if param_filter_cutoff.is_smoothing() {
        let next_smoothed = param_filter_cutoff.next();
        self.filters.iter_mut().for_each(|f| {
            f.set_cutoff(next_smoothed);
        })
    }

    self.filters[0].process_sample(l);
    self.filters[1].process_sample(r);
}

```




