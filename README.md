# JDSP
Collection of audio processing tools written in Rust for use in audio plug-in development with Nih-Plug.

## Features
- adaa_nl 
- circular_buffer
- dc_filter
- envelope
- iir_biquad_filter
- oversampler
- window

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




