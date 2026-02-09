//! Fibonacci Heap implementation

use pyo3::prelude::*;

/// Fibonacci Heap data structure
#[pyclass]
pub struct FibonacciHeap {
    // Simplified implementation for now
    elements: Vec<(f64, String)>,
}

impl FibonacciHeap {
    /// Create a new empty Fibonacci Heap
    pub fn new() -> Self {
        FibonacciHeap {
            elements: Vec::new(),
        }
    }
    
    /// Insert a new element into the heap
    pub fn insert(&mut self, key: f64, value: String) {
        self.elements.push((key, value));
        self.elements.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }
    
    /// Extract the minimum element
    pub fn extract_min(&mut self) -> Option<String> {
        if self.elements.is_empty() {
            None
        } else {
            Some(self.elements.remove(0).1)
        }
    }
    
    /// Get the number of elements in the heap
    pub fn len(&self) -> usize {
        self.elements.len()
    }
    
    /// Check if the heap is empty
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fibonacci_heap() {
        let mut heap = FibonacciHeap::new();
        assert!(heap.is_empty());
        
        heap.insert(5.0, "five".to_string());
        heap.insert(2.0, "two".to_string());
        heap.insert(8.0, "eight".to_string());
        
        assert_eq!(heap.len(), 3);
        assert!(!heap.is_empty());
        
        let min = heap.extract_min();
        assert_eq!(min, Some("two".to_string()));
        
        let min = heap.extract_min();
        assert_eq!(min, Some("five".to_string()));
        
        let min = heap.extract_min();
        assert_eq!(min, Some("eight".to_string()));
        
        assert!(heap.is_empty());
    }
}