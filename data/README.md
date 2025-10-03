# Data Folder

This folder is intended to store the dataset used by the project.

The main dataset (`nashville_freeway_anomaly.csv`) is not included in this repository.  
It can be downloaded from the official source: [Dataset Website](https://acoursey3.github.io/ft-aed/).

Use the `download_data()` function in `data_utils.py` to download the dataset automatically.

```python
from src.data_utils import download_data

download_data()
