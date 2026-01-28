# Examples

This folder contains example scripts showing how to use the centralized data loader.

## Available Examples

### `example_usage.py`
Demonstrates how to:
- Load the full dataset
- Get filtered data by category (agriculture, energy, etc.)
- Perform custom analysis
- Use the caching system to avoid repeated downloads

## Running Examples

```bash
# Make sure you're in the project root
cd /path/to/ome-hackathon-season-14

# Run an example
python examples/example_usage.py
```

## Creating Your Own Analysis

Use these examples as a template for your own analysis scripts:

```python
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main.main import get_agriculture_data

# Your analysis code here
df = get_agriculture_data()
# ... perform your analysis
```
