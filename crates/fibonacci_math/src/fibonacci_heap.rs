// Fibonacci Heap implementation for Rust

use std::cmp::Ordering;
use std::collections::HashMap;

/// Node in the Fibonacci Heap
#[derive(Debug, Clone)]
struct FibonacciNode<T> {
    key: T,
    degree: usize,
    marked: bool,
    parent: Option<usize>,
    child: Option<usize>,
    left: Option<usize>,
    right: Option<usize>,
}

/// Fibonacci Heap data structure
#[derive(Debug, Clone)]
pub struct FibonacciHeap<T: Ord + Clone> {
    nodes: Vec<FibonacciNode<T>>,
    root_list: Option<usize>,
    min_node: Option<usize>,
    node_count: usize,
    node_map: HashMap<usize, usize>, // Maps node indices to their actual positions
}

impl<T: Ord + Clone> FibonacciHeap<T> {
    /// Creates a new empty Fibonacci Heap
    pub fn new() -> Self {
        FibonacciHeap {
            nodes: Vec::new(),
            root_list: None,
            min_node: None,
            node_count: 0,
            node_map: HashMap::new(),
        }
    }

    /// Inserts a new key into the heap
    pub fn insert(&mut self, key: T) -> usize {
        let node_index = self.nodes.len();
        let new_node = FibonacciNode {
            key,
            degree: 0,
            marked: false,
            parent: None,
            child: None,
            left: None,
            right: None,
        };

        self.nodes.push(new_node);
        self.node_count += 1;

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

        node_index
    }

    /// Extracts the minimum element from the heap
    pub fn extract_min(&mut self) -> Option<T> {
        if let Some(min_index) = self.min_node {
            // Move all children of min to root list
            if let Some(child_index) = self.nodes[min_index].child {
                let mut current_child = child_index;
                let mut child_count = 0;
                
                // Count children
                let mut temp = current_child;
                loop {
                    child_count += 1;
                    temp = self.nodes[temp].right.unwrap_or(current_child);
                    if temp == current_child {
                        break;
                    }
                }
                
                // Move all children to root list
                let mut child = child_index;
                for _ in 0..child_count {
                    let next_child = self.nodes[child].right.unwrap();
                    
                    // Remove from child list
                    if let Some(child_left) = self.nodes[child].left {
                        self.nodes[child_left].right = self.nodes[child].right;
                    }
                    if let Some(child_right) = self.nodes[child].right {
                        self.nodes[child_right].left = self.nodes[child].left;
                    }
                    
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
                    
                    child = next_child;
                }
            }
            
            // Remove min node from root list
            if let Some(min_left) = self.nodes[min_index].left {
                self.nodes[min_left].right = self.nodes[min_index].right;
            }
            if let Some(min_right) = self.nodes[min_index].right {
                self.nodes[min_right].left = self.nodes[min_index].left;
            }
            
            // If min was the only node in root list
            if self.root_list == Some(min_index) {
                self.root_list = self.nodes[min_index].right;
            }
            
            // Clear min node
            self.min_node = None;
            
            // Consolidate
            self.consolidate();
            
            // Return the key
            let key = self.nodes[min_index].key.clone();
            self.nodes.remove(min_index);
            self.node_count -= 1;
            
            Some(key)
        } else {
            None
        }
    }

    /// Decreases the key of a node
    pub fn decrease_key(&mut self, node_index: usize, new_key: T) {
        if node_index >= self.nodes.len() {
            return;
        }
        
        if new_key > self.nodes[node_index].key {
            return; // Cannot increase key
        }
        
        self.nodes[node_index].key = new_key;
        
        // If node is not in root list and key is smaller than parent's key
        if let Some(parent_index) = self.nodes[node_index].parent {
            if self.nodes[node_index].key < self.nodes[parent_index].key {
                // Cut node from parent
                self.cut(node_index, parent_index);
                // Cascading cut
                self.cascading_cut(parent_index);
            }
        }
        
        // Update minimum if needed
        if let Some(min_index) = self.min_node {
            if self.nodes[node_index].key < self.nodes[min_index].key {
                self.min_node = Some(node_index);
            }
        }
    }

    /// Helper function to cut a node from its parent
    fn cut(&mut self, node_index: usize, parent_index: usize) {
        // Remove node from parent's child list
        if let Some(child_index) = self.nodes[parent_index].child {
            if child_index == node_index {
                // If this was the only child
                if self.nodes[node_index].left == Some(node_index) && self.nodes[node_index].right == Some(node_index) {
                    self.nodes[parent_index].child = None;
                } else {
                    // Find the next child
                    let next_child = self.nodes[node_index].right;
                    self.nodes[parent_index].child = next_child;
                    
                    // Update links
                    if let Some(next_child_index) = next_child {
                        self.nodes[next_child_index].left = self.nodes[node_index].left;
                    }
                }
            }
            
            // Remove node from its own links
            if let Some(left_index) = self.nodes[node_index].left {
                self.nodes[left_index].right = self.nodes[node_index].right;
            }
            if let Some(right_index) = self.nodes[node_index].right {
                self.nodes[right_index].left = self.nodes[node_index].left;
            }
            
            // Add to root list
            if let Some(root_index) = self.root_list {
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
            
            // Clear parent
            self.nodes[node_index].parent = None;
            self.nodes[node_index].marked = false;
        }
    }

    /// Helper function for cascading cut
    fn cascading_cut(&mut self, node_index: usize) {
        if let Some(parent_index) = self.nodes[node_index].parent {
            if !self.nodes[node_index].marked {
                self.nodes[node_index].marked = true;
            } else {
                // Cut node from parent
                self.cut(node_index, parent_index);
                // Continue cascading
                self.cascading_cut(parent_index);
            }
        }
    }

    /// Consolidate the heap
    fn consolidate(&mut self) {
        if self.root_list.is_none() {
            return;
        }
        
        // Create array to hold roots of same degree
        let mut degree_array = vec![None; self.node_count + 1];
        
        // Process all nodes in root list
        let mut current = self.root_list;
        let mut count = 0;
        
        // Count nodes in root list
        if let Some(start_index) = current {
            let mut temp = start_index;
            loop {
                count += 1;
                temp = self.nodes[temp].right.unwrap_or(start_index);
                if temp == start_index {
                    break;
                }
            }
        }
        
        // Process each node
        let mut node_count = 0;
        let mut current = self.root_list;
        
        while node_count < count && current.is_some() {
            let mut x = current.unwrap();
            let mut d = self.nodes[x].degree;
            
            // While there's another node with the same degree
            while let Some(y) = degree_array[d] {
                // Make sure x is the smaller key
                if self.nodes[x].key > self.nodes[y].key { // Type annotation needed for comparison
                    std::mem::swap(&mut x, &mut y);
                }
                
                // Make y a child of x
                self.make_child(x, y);
                
                // Clear degree of y
                degree_array[d] = None;
                d += 1;
            }
            
            degree_array[d] = Some(x);
            
            // Move to next node
            current = self.nodes[x].right;
            node_count += 1;
        }
        
        // Update root list
        self.root_list = None;
        self.min_node = None;
        
        // Rebuild root list
        for i in 0..degree_array.len() {
            if let Some(node_index) = degree_array[i] {
                if self.root_list.is_none() {
                    self.root_list = Some(node_index);
                    self.nodes[node_index].right = Some(node_index);
                    self.nodes[node_index].left = Some(node_index);
                } else {
                    let root_right = self.nodes[self.root_list.unwrap()].right;
                    self.nodes[node_index].right = root_right;
                    self.nodes[self.root_list.unwrap()].right = Some(node_index);
                    if let Some(right_index) = root_right {
                        self.nodes[right_index].left = Some(node_index);
                    }
                    self.nodes[node_index].left = Some(self.root_list.unwrap());
                }
                
                // Update minimum
                if let Some(min_index) = self.min_node {
                    if self.nodes[node_index].key < self.nodes[min_index].key {
                        self.min_node = Some(node_index);
                    }
                } else {
                    self.min_node = Some(node_index);
                }
            }
        }
    }

    /// Make node y a child of node x
    fn make_child(&mut self, x: usize, y: usize) {
        // Remove y from root list
        if let Some(y_left) = self.nodes[y].left {
            self.nodes[y_left].right = self.nodes[y].right;
        }
        if let Some(y_right) = self.nodes[y].right {
            self.nodes[y_right].left = self.nodes[y].left;
        }
        
        // If y was the only node in root list
        if self.root_list == Some(y) {
            self.root_list = self.nodes[y].right;
        }
        
        // Make y a child of x
        self.nodes[y].parent = Some(x);
        self.nodes[y].marked = false;
        
        // Add y to children of x
        if let Some(child_index) = self.nodes[x].child {
            let child_right = self.nodes[child_index].right;
            self.nodes[y].right = child_right;
            self.nodes[child_index].right = Some(y);
            if let Some(right_index) = child_right {
                self.nodes[right_index].left = Some(y);
            }
            self.nodes[y].left = Some(child_index);
        } else {
            self.nodes[x].child = Some(y);
            self.nodes[y].right = Some(y);
            self.nodes[y].left = Some(y);
        }
        
        // Increase degree of x
        self.nodes[x].degree += 1;
    }

    /// Returns the number of elements in the heap
    pub fn len(&self) -> usize {
        self.node_count
    }

    /// Returns true if the heap is empty
    pub fn is_empty(&self) -> bool {
        self.node_count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_extract_min() {
        let mut heap = FibonacciHeap::new();
        heap.insert(5);
        heap.insert(3);
        heap.insert(7);
        
        assert_eq!(heap.extract_min(), Some(3));
        assert_eq!(heap.extract_min(), Some(5));
        assert_eq!(heap.extract_min(), Some(7));
        assert_eq!(heap.extract_min(), None);
    }

    #[test]
    fn test_decrease_key() {
        let mut heap = FibonacciHeap::new();
        let node_index = heap.insert(10);
        heap.decrease_key(node_index, 2);
        
        assert_eq!(heap.extract_min(), Some(2));
    }

    #[test]
    fn test_heap_properties() {
        let mut heap = FibonacciHeap::new();
        heap.insert(5);
        heap.insert(3);
        heap.insert(7);
        heap.insert(1);
        
        assert_eq!(heap.extract_min(), Some(1));
        assert_eq!(heap.extract_min(), Some(3));
        assert_eq!(heap.extract_min(), Some(5));
        assert_eq!(heap.extract_min(), Some(7));
        assert_eq!(heap.extract_min(), None);
    }
}