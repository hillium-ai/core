use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use tokio::sync::Mutex;

/// Structured note representing a piece of contextual information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredNote {
    pub id: String,
    pub timestamp: u64,
    pub domain: String,
    pub content: String,
    pub tags: Vec<String>,
}

/// Context record for conversation context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextRecord {
    pub id: String,
    pub timestamp: u64,
    pub domain: String,
    pub context: Vec<StructuredNote>,
}

/// Working Memory Manager using Sled DB
pub type WorkingMemory = WorkingMemoryManager;

#[derive(Debug)]
pub struct WorkingMemoryManager {
    db: Arc<Mutex<sled::Db>>,
}

impl WorkingMemoryManager {
    /// Create a new WorkingMemoryManager instance
    pub async fn new<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let db = sled::open(path)?;
        Ok(Self {
            db: Arc::new(Mutex::new(db)),
        })
    }

    /// Open an existing WorkingMemoryManager instance
    pub async fn open<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        Self::new(path).await
    }

    /// Store a context record
    pub async fn store_context(&self, record: &ContextRecord) -> anyhow::Result<()> {
        let db = self.db.lock().await;
        let key = format!("ctx:{}", record.id);
        let value = bincode::serialize(record)?;
        db.insert(key, value)?;
        db.flush_async().await?;
        Ok(())
    }

    /// Retrieve a context record by ID
    pub async fn retrieve_context(&self, id: &str) -> anyhow::Result<Option<ContextRecord>> {
        let db = self.db.lock().await;
        let key = format!("ctx:{}", id);
        match db.get(&key)? {
            Some(value) => {
                let record: ContextRecord = bincode::deserialize(&value)?;
                Ok(Some(record))
            }
            None => Ok(None),
        }
    }

    /// Update a context record
    pub async fn update_context(&self, record: &ContextRecord) -> anyhow::Result<()> {
        self.store_context(record).await
    }

    /// Store a structured note
    pub async fn write_note(&self, note: &StructuredNote) -> anyhow::Result<()> {
        let db = self.db.lock().await;
        let tree = db.open_tree("notes")?;
        let key = format!("note:{}", note.id);
        let value = bincode::serialize(note)?;
        tree.insert(key, value)?;
        db.flush_async().await?;
        Ok(())
    }

    /// Read a structured note by ID
    pub async fn read_note(&self, id: &str) -> anyhow::Result<Option<StructuredNote>> {
        let db = self.db.lock().await;
        let tree = db.open_tree("notes")?;
        let key = format!("note:{}", id);
        match tree.get(&key)? {
            Some(value) => {
                let note: StructuredNote = bincode::deserialize(&value)?;
                Ok(Some(note))
            }
            None => Ok(None),
        }
    }

    /// Scan notes by domain
    pub async fn scan_by_domain(&self, domain: &str) -> anyhow::Result<Vec<StructuredNote>> {
        let db = self.db.lock().await;
        let mut results = Vec::new();
        
        let tree = db.open_tree("notes")?;
        for result in tree.iter() {
            let (_, value) = result?;
            if let Ok(note) = bincode::deserialize::<StructuredNote>(&value) {
                if note.domain == domain {
                    results.push(note);
                }
            }
        }
        
        Ok(results)
    }

    /// Garbage collect old notes (older than timestamp)
    pub async fn gc_old_notes(&self, max_age_seconds: u64) -> anyhow::Result<usize> {
        let db = self.db.lock().await;
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0))
            .as_secs();
        let cutoff = current_time.saturating_sub(max_age_seconds);
        
        let mut deleted_count = 0;
        let mut to_delete = Vec::new();
        
        let tree = db.open_tree("notes")?;
        for result in tree.iter() {
            let (key, value) = result?;
            if let Ok(note) = bincode::deserialize::<StructuredNote>(&value) {
                if note.timestamp < cutoff {
                    to_delete.push(key);
                    deleted_count += 1;
                }
            }
        }
        
        for key in to_delete {
            tree.remove(key)?;
        }
        
        db.flush_async().await?;
        Ok(deleted_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_working_memory_manager() -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let wm = WorkingMemoryManager::new(temp_dir.path()).await?;
        
        // Test storing and retrieving a note
        let note = StructuredNote {
            id: "test-note-1".to_string(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::from_secs(0))
                .as_secs(),
            domain: "test".to_string(),
            content: "Test content".to_string(),
            tags: vec!["test".to_string(), "example".to_string()],
        };
        
        wm.write_note(&note).await?;
        let retrieved = wm.read_note("test-note-1").await?;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "Test content");
        
        // Test storing and retrieving a context record
        let context = ContextRecord {
            id: "test-context-1".to_string(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::from_secs(0))
                .as_secs(),
            domain: "test".to_string(),
            context: vec![note.clone()],
        };
        
        wm.store_context(&context).await?;
        let retrieved_context = wm.retrieve_context("test-context-1").await?;
        assert!(retrieved_context.is_some());
        assert_eq!(retrieved_context.unwrap().context[0].content, "Test content");
        
        // Test scanning by domain
        let notes = wm.scan_by_domain("test").await?;
        assert_eq!(notes.len(), 1);
        
        // Test garbage collection
        let old_note = StructuredNote {
            id: "old-note-1".to_string(),
            timestamp: 1_000_000, // Very old timestamp
            domain: "test".to_string(),
            content: "Old content".to_string(),
            tags: vec![],
        };
        wm.write_note(&old_note).await?;
        let deleted = wm.gc_old_notes(100).await?;
        assert_eq!(deleted, 1);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_access() -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let wm = WorkingMemoryManager::new(temp_dir.path()).await?;
        
        // Test concurrent writes
        let mut handles = vec![];
        for i in 0..10 {
            let wm_clone = wm.clone();
            let handle = tokio::spawn(async move {
                let note = StructuredNote {
                    id: format!("concurrent-{}", i),
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or(Duration::from_secs(0))
                    .as_secs(),
                    domain: "concurrent".to_string(),
                    content: format!("Content {}", i),
                    tags: vec![],
                };
                wm_clone.write_note(&note).await
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.await??;
        }
        
        // Verify all notes were written
        let notes = wm.scan_by_domain("concurrent").await?;
        assert_eq!(notes.len(), 10);
        
        Ok(())
    }
}

impl Clone for WorkingMemoryManager {
    fn clone(&self) -> Self {
        Self {
            db: Arc::clone(&self.db),
        }
    }
}
