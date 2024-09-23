import os
import torch
import torch.nn as nn
from pymilvus import MilvusClient

from typing import List, Any

# Set up a Milvus client
client_instance = [None]


def create_client() -> MilvusClient:
    if client_instance[0] is not None:
        return client_instance[0]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    uri = os.path.join(current_dir, "example.db")
    client = MilvusClient(uri=uri)
    client_instance[0] = client  # type: ignore
    return client


def to_numpy(tensor):
    return tensor.detach().flatten(start_dim=1).cpu().numpy()


def search_similar_images(img_tensor: torch.Tensor, model: nn.Module, collection_name: str) -> List[str]:
    """
    Extract image features using ResNet34 model and store them in Milvus.
    """
    client = create_client()
    output = model(img_tensor)
    output = to_numpy(output)
    results: Any = client.search(collection_name, data=output,
                                 output_fields=["filename",],
                                 limit=3, search_params={"metric_type": "COSINE"},)
    return results
