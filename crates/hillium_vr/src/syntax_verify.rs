use zenoh::config::Config;

fn main() {
    let config = Config::default();
    // Verification of the new skill pattern: .res().wait()
    let _ = zenoh::open(config).res().wait();
    println!("Zenoh syntax verified!");
}
