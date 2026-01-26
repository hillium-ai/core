import sys
from pathlib import Path
sys.path.append("/Users/jsaldana/GitLocalRepo/Hillium-Levitate/src")
from levitate.core.semantic_index import get_semantic_index

def check_symlinks():
    root = Path.cwd()
    idx = get_semantic_index(root)
    
    if not idx.collection:
        print("❌ No index found.")
        return

    print(f"📊 Total Chunks: {idx.collection.count()}")
    
    # Dump paths
    res = idx.collection.get() # Get all metadata
    paths = [m['path'] for m in res['metadatas']]
    
    # Check for MD files and external paths
    md_files = [p for p in paths if p.endswith('.md')]
    
    print("\n📄 Markdown Files Indexed:")
    if md_files:
        for p in md_files:
            print(f" - {p}")
    else:
        print("⚠️ No Markdown files found in index!")

    # Check for 'private_docs' keyword
    symlinked = [p for p in paths if "private_docs" in p or "work-packages" in p]
    print("\n🔗 Symlinked Docs Candidates:")
    if symlinked:
        for p in symlinked:
            print(f" - {p}")
    else:
        print("⚠️ No paths containing 'private_docs' or 'work-packages' found.")

if __name__ == "__main__":
    check_symlinks()
