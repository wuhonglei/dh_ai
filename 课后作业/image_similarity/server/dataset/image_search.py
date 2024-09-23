import os
import torch
from pymilvus import MilvusClient

from typing import List, Any


def search_similar_images(img_tensor: torch.Tensor, model: torch.nn.Module) -> List[str]:
    """
    Extract image features using ResNet34 model and store them in Milvus.
    """

    # Set up a Milvus client
    current_dir = os.path.dirname(os.path.abspath(__file__))
    uri = os.path.join(current_dir, "example.db")
    client = MilvusClient(uri=uri)
    output = model(img_tensor)
    print('output.shape', output.shape)
    output = output.detach().flatten(start_dim=1).cpu().numpy()
    results: Any = client.search("image_embeddings", data=output,
                                 output_fields=["filename"], limit=10, search_params={"metric_type": "COSINE"},)
    return results
