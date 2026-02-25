use pyo3::prelude::*;


/// Node in the Fibonacci Heap
#[derive(Debug, Clone)]
struct FibonacciNode {
    key: f64,
    #[allow(dead_code)]
    degree: usize,
    marked: bool,
    parent: Option<usize>,
    child: Option<usize>,
    left: usize,  // Circular doubly-linked list pointers
    right: usize,
}

/// Fibonacci Heap data structure
#[pyclass]
#[derive(Debug, Clone)]
pub struct FibonacciHeap {
    nodes: Vec<FibonacciNode>,
    min_root: Option<usize>,
    node_count: usize,
    root_list_head: Option<usize>,
}

#[pymethods]
impl FibonacciHeap {
    /// Create a new empty Fibonacci Heap
    #[new]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            min_root: None,
            node_count: 0,
            root_list_head: None,
        }
    }

    /// Insert a new key into the heap
    pub fn insert(&mut self, key: f64) -> usize {
        let node_idx = self.nodes.len();
        
        let new_node = FibonacciNode {
            key,
            degree: 0,
            marked: false,
            parent: None,
            child: None,
            left: node_idx,  // Points to itself initially
            right: node_idx,
        };
        
        self.nodes.push(new_node);
        
        // Add to root list
        if let Some(root_head) = self.root_list_head {
            // Insert between root_head and root_head.right
            let right_of_head = self.nodes[root_head].right;
            self.nodes[node_idx].right = right_of_head;
            self.nodes[right_of_head].left = node_idx;
            self.nodes[root_head].right = node_idx;
            self.nodes[node_idx].left = root_head;
        } else {
            // First node
            self.root_list_head = Some(node_idx);
        }
        
        // Update min_root if needed
        if let Some(min_idx) = self.min_root {
            if key < self.nodes[min_idx].key {
                self.min_root = Some(node_idx);
            }
        } else {
            self.min_root = Some(node_idx);
        }
        
        self.node_count += 1;
        node_idx
    }

    /// Get the minimum key without removing it
    pub fn find_min(&self) -> Option<f64> {
        self.min_root.map(|idx| self.nodes[idx].key)
    }

    /// Extract the minimum key
    pub fn extract_min(&mut self) -> Option<f64> {
        if let Some(min_idx) = self.min_root {
            let min_key = self.nodes[min_idx].key;
            
            // Move all children of min to root list
            if let Some(child_idx) = self.nodes[min_idx].child {
                let mut current_child = child_idx;
                loop {
                    // Remove from child's parent
                    self.nodes[current_child].parent = None;
                    
                    // Add to root list
                    if let Some(root_head) = self.root_list_head {
                        let right_of_head = self.nodes[root_head].right;
                        self.nodes[current_child].right = right_of_head;
                        self.nodes[right_of_head].left = current_child;
                        self.nodes[root_head].right = current_child;
                        self.nodes[current_child].left = root_head;
                    } else {
                        self.root_list_head = Some(current_child);
                        self.nodes[current_child].left = current_child;
                        self.nodes[current_child].right = current_child;
                    }
                    
                    // Move to next child (circular)
                    let next_child = self.nodes[current_child].right;
                    if next_child == child_idx {
                        break;
                    }
                    current_child = next_child;
                }
            }
            
            // Remove min node from root list
            if self.nodes[min_idx].left != min_idx {
                // Not the only node in root list
                let left_of_min = self.nodes[min_idx].left;
                let right_of_min = self.nodes[min_idx].right;
                self.nodes[left_of_min].right = right_of_min;
                self.nodes[right_of_min].left = left_of_min;
            } else {
                // Only node in root list
                self.root_list_head = None;
            }
            
            // Clear min_root
            self.min_root = None;
            
            // Consolidate the heap
            self.consolidate();
            
            self.node_count -= 1;
            Some(min_key)
        } else {
            None
        }
    }

    /// Decrease key of a node
    pub fn decrease_key(&mut self, node_id: usize, new_key: f64) {
        if new_key > self.nodes[node_id].key {
            return; // Cannot increase key
        }
        
        self.nodes[node_id].key = new_key;
        
        // If the new key is smaller than its parent's key, cut it
        if let Some(parent_idx) = self.nodes[node_id].parent {
            if new_key < self.nodes[parent_idx].key {
                self.cut(node_id, parent_idx);
                self.cascading_cut(parent_idx);
            }
        }
        
        // Update min_root if needed
        if let Some(min_idx) = self.min_root {
            if new_key < self.nodes[min_idx].key {
                self.min_root = Some(node_id);
            }
        } else {
            self.min_root = Some(node_id);
        }
    }

    /// Helper function for decrease_key
    fn cut(&mut self, node_id: usize, parent_idx: usize) {
        // Remove node from parent's child list
        if let Some(child_idx) = self.nodes[parent_idx].child {
            if child_idx == node_id {
                // If it was the only child
                if self.nodes[node_id].right == node_id {
                    self.nodes[parent_idx].child = None;
                } else {
                    // Remove from circular list
                    let right_of_node = self.nodes[node_id].right;
                    let left_of_node = self.nodes[node_id].left;
                    self.nodes[parent_idx].child = Some(right_of_node);
                    self.nodes[right_of_node].left = left_of_node;
                    self.nodes[left_of_node].right = right_of_node;
                }
            } else {
                // Remove from circular list
                let right_of_node = self.nodes[node_id].right;
                let left_of_node = self.nodes[node_id].left;
                let left_id = left_of_node;
                self.nodes[right_of_node].left = left_id;
                self.nodes[left_id].right = right_of_node;
            }
        }
        
        // Mark as unmarked
        self.nodes[node_id].marked = false;
        
        // Add to root list
        if let Some(root_head) = self.root_list_head {
            let right_of_head = self.nodes[root_head].right;
            self.nodes[node_id].right = right_of_head;
            self.nodes[right_of_head].left = node_id;
            self.nodes[root_head].right = node_id;
            self.nodes[node_id].left = root_head;
        } else {
            self.root_list_head = Some(node_id);
            self.nodes[node_id].left = node_id;
            self.nodes[node_id].right = node_id;
        }
        
        // Clear parent
        self.nodes[node_id].parent = None;
    }

    /// Helper function for decrease_key
    fn cascading_cut(&mut self, node_idx: usize) {
        if let Some(parent_idx) = self.nodes[node_idx].parent {
            if !self.nodes[node_idx].marked {
                self.nodes[node_idx].marked = true;
            } else {
                self.cut(node_idx, parent_idx);
                self.cascading_cut(parent_idx);
            }
        }
    }

    /// Consolidate the heap
    fn consolidate(&mut self) {
        // In a full implementation, this would consolidate trees of the same degree
        // For now, we'll just update min_root if needed
        if let Some(root_head) = self.root_list_head {
            let mut current = root_head;
            let mut min_key = self.nodes[root_head].key;
            let mut min_idx = root_head;
            
            loop {
                if self.nodes[current].key < min_key {
                    min_key = self.nodes[current].key;
                    min_idx = current;
                }
                
                current = self.nodes[current].right;
                if current == root_head {
                    break;
                }
            }
            
            self.min_root = Some(min_idx);
        }
    }

    /// Get the number of nodes in the heap
    pub fn len(&self) -> usize {
        self.node_count
    }

    /// Check if the heap is empty
    pub fn is_empty(&self) -> bool {
        self.node_count == 0
    }
}
