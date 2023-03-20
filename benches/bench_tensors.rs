use criterion::{criterion_group, criterion_main, Criterion};
use rest_tensors::{RIFull,MatrixFull};


fn ao2mo() {
    let ri3fn = RIFull::new([10,10,20],2.0);
    let eigenvector = MatrixFull::new([10,10],1.0);

    ri3fn.ao2mo_v02(&eigenvector).unwrap();
    
    //println!("{:?}", ri3mo.size);
}


fn bench_ao2mo(c: &mut Criterion) {
    c.bench_function("ao2mo for RIFull", |b| b.iter(|| ao2mo()));
}

criterion_group!(benches, bench_ao2mo);
criterion_main!(benches);