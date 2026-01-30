fn main() {
    // Generate C bindings
    let bindings = cbindgen::generate(".")
        .expect("Failed to generate bindings");
    
    bindings.write_to_file("src/powerinfer_bindings.h");
    
    // Print cargo link directive
    println!("cargo:rustc-link-lib=powerinfer");
}