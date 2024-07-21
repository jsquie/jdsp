use adaa_nl::adaa::{AntiderivativeOrder, NonlinearProcessor, ProcessorState, ProcessorStyle};
use criterion::{criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

fn generate_signal_data() -> Vec<f32> {
    let mut r = StdRng::seed_from_u64(222); // <- Here we set the seed
    let normal = Normal::new(0.0, 2.0).unwrap();
    (0..480)
        .map(|_| normal.sample(&mut r))
        .collect::<Vec<f32>>()
}

fn adaa_bench(c: &mut Criterion) {
    let mut hard_clip_first_order = NonlinearProcessor::new();
    let mut hard_clip_second_order = NonlinearProcessor::new();
    let mut tanh_first_order = NonlinearProcessor::new();
    let mut tanh_second_order = NonlinearProcessor::new();

    hard_clip_second_order.compare_and_change_state(ProcessorState::State(
        ProcessorStyle::HardClip,
        AntiderivativeOrder::SecondOrder,
    ));

    tanh_first_order.compare_and_change_state(ProcessorState::State(
        ProcessorStyle::Tanh,
        AntiderivativeOrder::FirstOrder,
    ));

    tanh_second_order.compare_and_change_state(ProcessorState::State(
        ProcessorStyle::Tanh,
        AntiderivativeOrder::SecondOrder,
    ));

    let sig = generate_signal_data();

    c.bench_function("hard clip first order", |b| {
        b.iter(|| {
            sig.iter().for_each(|v| {
                hard_clip_first_order.process(*v);
            })
        })
    });

    c.bench_function("hard clip second order", |b| {
        b.iter(|| {
            sig.iter().for_each(|v| {
                hard_clip_second_order.process(*v);
            })
        })
    });

    c.bench_function("tanh clip first order", |b| {
        b.iter(|| {
            sig.iter().for_each(|v| {
                tanh_first_order.process(*v);
            })
        })
    });

    c.bench_function("tanh clip second order", |b| {
        b.iter(|| {
            sig.iter().for_each(|v| {
                tanh_second_order.process(*v);
            })
        })
    });
}

criterion_group!(benches, adaa_bench);
criterion_main!(benches);
