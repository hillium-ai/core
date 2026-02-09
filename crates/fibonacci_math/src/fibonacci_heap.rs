//! Fibonacci Heap data structure

use std::collections::HashMap;

/// Node in the Fibonacci heap
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

/// Fibonacci Heap implementation
pub struct FibonacciHeap<T> {
    nodes: Vec<FibonacciNode<T>>,
    root_list: Option<usize>,
    min_node: Option<usize>,
    node_count: usize,
}

impl<T> FibonacciHeap<T> {
    /// Creates a new empty Fibonacci heap
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root_list: None,
            min_node: None,
            node_count: 0,
        }
    }
    
    /// Inserts a new node with the given key and value
    pub fn insert(&mut self, key: f64, value: T) -> usize {
        let node_index = self.nodes.len();
        let node = FibonacciNode {
            value,
            key,
            degree: 0,
            marked: false,
            parent: None,
            child: None,
            left: None,
            right: None,
        };
        
        self.nodes.push(node);
        self.node_count += 1;
        
        // Add to root list
        if let Some(root_index) = self.root_list {
            // Insert between root_index and root_index.right
            let root_node = &mut self.nodes[root_index];
            let right_of_root = root_node.right;
            
            // Update links
            root_node.right = Some(node_index);
            self.nodes[node_index].left = Some(root_index);
            self.nodes[node_index].right = right_of_root;
            
            if let Some(right_index) = right_of_root {
                self.nodes[right_index].left = Some(node_index);
            }
        } else {
            self.root_list = Some(node_index);
            self.min_node = Some(node_index);
        }
        
        // Update min node if needed
        if let Some(min_index) = self.min_node {
            if self.nodes[node_index].key < self.nodes[min_index].key {
                self.min_node = Some(node_index);
            }
        }
        
        node_index
    }
    
    /// Decreases the key of a node
    pub fn decrease_key(&mut self, node_index: usize, new_key: f64) {
        // TODO: Implement proper decrease-key with cascading cuts
        // For now, just update the key
        self.nodes[node_index].key = new_key;
        
        // Update min node if needed
        if let Some(min_index) = self.min_node {
            if self.nodes[node_index].key < self.nodes[min_index].key {
                self.min_node = Some(node_index);
            }
        }
    }
    
    /// Returns the number of nodes in the heap
    pub fn len(&self) -> usize {
        self.node_count
    }
    
    /// Checks if the heap is empty
    pub fn is_empty(&self) -> bool {
        self.node_count == 0
    }
}
