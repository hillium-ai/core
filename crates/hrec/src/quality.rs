/// Jitter scorer for measuring temporal consistency in streams
pub struct JitterScorer {
    window_size: usize,
    samples: Vec<u64>,
}

impl JitterScorer {
    pub fn new(window_size: usize) -> Self {
        JitterScorer {
            window_size,
            samples: Vec::new(),
        }
    }

    pub fn add_sample(&mut self, timestamp_us: u64) {
        self.samples.push(timestamp_us);
        if self.samples.len() > self.window_size {
            self.samples.remove(0);
        }
    }

    pub fn jitter_score(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }

        let intervals: Vec<u64> = self.samples
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        let mean: u64 = intervals.iter().sum::<u64>() / intervals.len() as u64;
        let variance: f64 = intervals
            .iter()
            .map(|&i| {
                let diff = (i as i64 - mean as i64) as f64;
                diff * diff
            })
            .sum::<f64>()
            / intervals.len() as f64;

        variance.sqrt()
    }
}

/// Coverage metric for measuring data completeness
pub struct CoverageMetric {
    expected_samples: u64,
    actual_samples: u64,
}

impl CoverageMetric {
    pub fn new(expected_samples: u64, actual_samples: u64) -> Self {
        CoverageMetric {
            expected_samples,
            actual_samples,
        }
    }

    pub fn coverage(&self) -> f64 {
        if self.expected_samples == 0 {
            return 1.0;
        }
        self.actual_samples as f64 / self.expected_samples as f64
    }
}
