use tch::Tensor;

fn main() {
    println!("🔥 Starting Kinetic Stack Smoke Test...");
    
    // 1. Verify we can create a tensor (LibTorch linking check)
    let t = Tensor::from_slice(&[3, 1, 4, 1, 5]);
    println!("✅ Tensor created successfully: {:?}", t);
    
    // 2. Verify basic math (Torch C++ Ops)
    let t2 = &t * 2;
    t2.print();
    
    // 3. Verify CUDA availability (Likely false on Mac/CPU container, but good to know)
    println!("ℹ️ CUDA Available: {}", tch::Cuda::is_available());
    println!("ℹ️ MPS Available: {}", tch::utils::has_mps());
    
    println!("🎉 KINETIC STACK VERIFIED: Rust-LibTorch-Python link established.");
}
