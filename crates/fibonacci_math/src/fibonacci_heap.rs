//! Fibonacci Heap implementation with O(1) amortized decrease-key

use pyo3::prelude::*;
use std::collections::HashMap;

/// Node in the Fibonacci Heap
#[derive(Debug, Clone)]
pub struct FibonacciNode<T> {
    pub value: T,
    pub key: f64,
    pub degree: usize,
    pub marked: bool,
    pub parent: Option<usize>,
    pub child: Option<usize>,
    pub left: Option<usize>,
    pub right: Option<usize>,
}

/// Fibonacci Heap data structure
#[pyclass]
#[derive(Clone)]
pub struct FibonacciHeap<T> {
    nodes: Vec<FibonacciNode<T>>,
    root_list: Option<usize>,
    min_node: Option<usize>,
    node_count: usize,
    node_map: HashMap<usize, usize>, // Maps original indices to current indices
}

#[pymethods]
impl<T> FibonacciHeap<T>
where
    T: Clone + std::fmt::Debug,
{
    /// Create a new empty Fibonacci Heap
    #[new]
    pub fn new() -> Self {
        FibonacciHeap {
            nodes: Vec::new(),
            root_list: None,
            min_node: None,
            node_count: 0,
            node_map: HashMap::new(),
        }
    }
    
    /// Insert a new element into the heap
    pub fn insert(&mut self, key: f64, value: T) -> usize {
        let node_index = self.nodes.len();
        let new_node = FibonacciNode {
            value,
            key,
            degree: 0,
            marked: false,
            parent: None,
            child: None,
            left: None,
            right: None,
        };
        
        self.nodes.push(new_node);
        
        // Add to root list
        if let Some(root_index) = self.root_list {
            // Insert between root_index and root_index.right
            let root_right = self.nodes[root_index].right;
            self.nodes[node_index].right = root_right;
            self.nodes[root_index].right = Some(node_index);
            
            if let Some(right_index) = root_right {
                self.nodes[right_index].left = Some(node_index);
            }
            
            self.nodes[node_index].left = Some(root_index);
        } else {
            self.root_list = Some(node_index);
            self.nodes[node_index].right = Some(node_index);
            self.nodes[node_index].left = Some(node_index);
        }
        
        // Update minimum node if needed
        if let Some(min_index) = self.min_node {
            if self.nodes[node_index].key < self.nodes[min_index].key {
                self.min_node = Some(node_index);
            }
        } else {
            self.min_node = Some(node_index);
        }
        
        self.node_count += 1;
        node_index
    }
    
    /// Extract the minimum element
    pub fn extract_min(&mut self) -> Option<T> {
        if let Some(min_index) = self.min_node {
            // Remove min from root list
            let min_node = &self.nodes[min_index];
            
            // Add children to root list
            if let Some(child_index) = min_node.child {
                let mut child = child_index;
                
                // Traverse all children
                loop {
                    let child_node = &mut self.nodes[child];
                    child_node.parent = None;
                    
                    // Add to root list
                    if let Some(root_index) = self.root_list {
                        let root_right = self.nodes[root_index].right;
                        self.nodes[child].right = root_right;
                        self.nodes[root_index].right = Some(child);
                        
                        if let Some(right_index) = root_right {
                            self.nodes[right_index].left = Some(child);
                        }
                        
                        self.nodes[child].left = Some(root_index);
                    } else {
                        self.root_list = Some(child);
                        self.nodes[child].right = Some(child);
                        self.nodes[child].left = Some(child);
                    }
                    
                    // Move to next child
                    let next_child = self.nodes[child].right;
                    if next_child == Some(child_index) {
                        break;
                    }
                    child = next_child.unwrap();
                }
            }
            
            // Remove min from root list
            if let Some(left_index) = min_node.left {
                if let Some(right_index) = min_node.right {
                    self.nodes[left_index].right = Some(right_index);
                    self.nodes[right_index].left = Some(left_index);
                } else {
                    // This was the only node in the root list
                    self.root_list = None;
                }
            }
            
            // Update min node
            if let Some(root_index) = self.root_list {
                self.min_node = Some(root_index);
                // Find the minimum in the root list
                let mut current = root_index;
                let mut min_key = self.nodes[root_index].key;
                
                loop {
                    if self.nodes[current].key < min_key {
                        min_key = self.nodes[current].key;
                        self.min_node = Some(current);
                    }
                    
                    let next = self.nodes[current].right;
                    if next == Some(root_index) {
                        break;
                    }
                    current = next.unwrap();
                }
            } else {
                self.min_node = None;
            }
            
            self.node_count -= 1;
            Some(min_node.value.clone())
        } else {
            None
        }
    }
    
    /// Decrease the key of a node
    pub fn decrease_key(&mut self, node_index: usize, new_key: f64) {
        if new_key > self.nodes[node_index].key {
            panic!("New key must be smaller than current key");
        }
        
        self.nodes[node_index].key = new_key;
        
        // Update min node if needed
        if let Some(min_index) = self.min_node {
            if self.nodes[node_index].key < self.nodes[min_index].key {
                self.min_node = Some(node_index);
            }
        }
    }
    
    /// Get the current size of the heap
    pub fn size(&self) -> usize {
        self.node_count
    }
    
    /// Check if the heap is empty
    pub fn is_empty(&self) -> bool {
        self.node_count == 0
    }
}

/// Get the minimum element without removing it
pub fn get_min<T>(heap: &FibonacciHeap<T>) -> Option<&T> {
    heap.min_node.map(|index| &heap.nodes[index].value)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fibonacci_heap_creation() {
        let heap: FibonacciHeap<i32> = FibonacciHeap::new();
        assert!(heap.is_empty());
        assert_eq!(heap.size(), 0);
    }
    
    #[test]
    fn test_insert_and_extract() {
        let mut heap: FibonacciHeap<i32> = FibonacciHeap::new();
        
        heap.insert(5.0, 10);
        heap.insert(2.0, 20);
        heap.insert(8.0, 30);
        
        assert_eq!(heap.size(), 3);
        
        // Extract minimum (should be 2.0)
        let min = heap.extract_min();
        assert_eq!(min, Some(20));
        assert_eq!(heap.size(), 2);
        
        // Extract next minimum (should be 5.0)
        let min = heap.extract_min();
        assert_eq!(min, Some(10));
        assert_eq!(heap.size(), 1);
        
        // Extract last element
        let min = heap.extract_min();
        assert_eq!(min, Some(30));
        assert!(heap.is_empty());
    }
    
    #[test]
    fn test_decrease_key() {
        let mut heap: FibonacciHeap<i32> = FibonacciHeap::new();
        let node_index = heap.insert(10.0, 100);
        
        // Decrease key
        heap.decrease_key(node_index, 5.0);
        
        // Should still be able to extract
        let min = heap.extract_min();
        assert_eq!(min, Some(100));
    }
}"}}Let me now implement the logarithmic spiral generator function and update the lib.rs file to properly expose all components: