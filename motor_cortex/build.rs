fn main() {
    // Bypass LibTorch version check to allow PyTorch 2.10.0 with tch 0.14
    println!("cargo:rustc-env=LIBTORCH_BYPASS_VERSION_CHECK=1");
    println!("cargo:rerun-if-env-changed=LIBTORCH_BYPASS_VERSION_CHECK");
}
