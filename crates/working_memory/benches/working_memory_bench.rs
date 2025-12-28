use criterion::{criterion_group, criterion_main, Criterion};
use working_memory::{WorkingMemoryManager, StructuredNote, ContextRecord};
use std::time::{SystemTime, UNIX_EPOCH};
use tempfile::tempdir;

fn benchmark_working_memory(c: &mut Criterion) {
    let temp_dir = tempdir().unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let wm = rt.block_on(async {
        WorkingMemoryManager::new(temp_dir.path()).await.unwrap()
    });
    
    // Benchmark note write
    c.bench_function("write_note", |b| {
        b.iter(|| {
            let note = StructuredNote {
                id: "bench-note".to_string(),
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                domain: "benchmark".to_string(),
                content: "Benchmark content".to_string(),
                tags: vec!["bench".to_string()],
            };
            
            rt.block_on(async {
                wm.write_note(&note).await
            }).unwrap();
        })
    });
    
    // Benchmark note read
    c.bench_function("read_note", |b| {
        b.iter(|| {
            rt.block_on(async {
                wm.read_note("bench-note").await
            }).unwrap();
        })
    });
    
    // Benchmark context store
    c.bench_function("store_context", |b| {
        b.iter(|| {
            let context = ContextRecord {
                id: "bench-context".to_string(),
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                domain: "benchmark".to_string(),
                context: vec![],
            };
            
            rt.block_on(async {
                wm.store_context(&context).await
            }).unwrap();
        })
    });
    
    // Benchmark context retrieve
    c.bench_function("retrieve_context", |b| {
        b.iter(|| {
            rt.block_on(async {
                wm.retrieve_context("bench-context").await
            }).unwrap();
        })
    });
}

criterion_group!(benches, benchmark_working_memory);
criterion_main!(benches);
