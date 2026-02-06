use restav_detector::VisualValidator;
use restav_detector::ReStraVDetector;
use restav_detector::MockReStraVDetector;

fn main() {
    println!("ReStraV detector implementation test");
    let detector = MockReStraVDetector::new();
    println!("Mock detector created successfully");
}