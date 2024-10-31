from .ssl4eo.ssl4eo import SSL4EODataset

def get_ssl4eo_metadata():
    dataset = SSL4EODataset()
    return dataset.metadata
