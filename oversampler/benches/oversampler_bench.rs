use criterion::{criterion_group, criterion_main, Criterion};
use oversampler::Oversample;
use oversampler::OversampleFactor;

fn os_bench(c: &mut Criterion) {
    let mut os_2x = Oversample::new(OversampleFactor::TwoTimes, 64);
    let mut os_4x = Oversample::new(OversampleFactor::FourTimes, 64);
    let mut os_8x = Oversample::new(OversampleFactor::EightTimes, 64);
    let mut os_16x = Oversample::new(OversampleFactor::SixteenTimes, 64);

    os_2x.initialize_oversample_stages();
    os_4x.initialize_oversample_stages();
    os_8x.initialize_oversample_stages();
    os_16x.initialize_oversample_stages();

    let mut sig_2x = vec![vec![1.0], vec![0.0_f32; 63]]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let mut sig_4x = sig_2x.clone();
    let mut sig_8x = sig_2x.clone();
    let mut sig_16x = sig_2x.clone();

    let mut up_sample_output_2x = vec![0.0_f32; 64 * 2];
    let mut up_sample_output_4x = vec![0.0_f32; 64 * 4];
    let mut up_sample_output_8x = vec![0.0_f32; 64 * 8];
    let mut up_sample_output_16x = vec![0.0_f32; 64 * 16];
    let mut output_2x = vec![0.0_f32; 64];
    let mut output_4x = vec![0.0_f32; 64];
    let mut output_8x = vec![0.0_f32; 64];
    let mut output_16x = vec![0.0_f32; 64];

    c.bench_function("os 2x up down", |b| {
        b.iter(|| {
            os_2x.process_up(&mut sig_2x, &mut up_sample_output_2x);
            os_2x.process_down(&mut up_sample_output_2x, &mut output_2x);
        })
    });

    c.bench_function("os 4x up down", |b| {
        b.iter(|| {
            os_4x.process_up(&mut sig_4x, &mut up_sample_output_4x);
            os_4x.process_down(&mut up_sample_output_4x, &mut output_4x);
        })
    });

    c.bench_function("os 8x up down", |b| {
        b.iter(|| {
            os_8x.process_up(&mut sig_8x, &mut up_sample_output_8x);
            os_8x.process_down(&mut up_sample_output_8x, &mut output_8x);
        })
    });

    c.bench_function("os 16x up down", |b| {
        b.iter(|| {
            os_16x.process_up(&mut sig_16x, &mut up_sample_output_16x);
            os_16x.process_down(&mut up_sample_output_16x, &mut output_16x);
        })
    });
}

criterion_group!(benches, os_bench);
criterion_main!(benches);
