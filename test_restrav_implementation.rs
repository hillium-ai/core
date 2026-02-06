use restrav_detector::VisualValidator;
use restrav_detector::ReStraVDetector;
// MockReStraVDetector is not implemented yet, using real one for now

fn main() {
    println!("ReStraV detector implementation test");
    let detector = ReStraVDetector::new();
    println!("Detector created successfully");
}