# ============================================================
# Indexer() for creating a searchable index of text embeddings
# ============================================================

# Import required packages for Indexer()
import os
import pickle
from typing import List, Tuple
import faiss
import numpy as np
from tqdm import tqdm

class Indexer(object):
    """
    Manage a searchable index of text embeddings using the `faiss` library.
    It does the following:
    - Adding embeddings to the index.
    - Searching for nearest neighbors (similar embeddings) to a given query.
    - Saving and loading the index to/from disk.
    """

    def __init__(self, vector_sz, device='cuda'):
        """
        Create a faiss index using IndexFlatIP (for inner product similarity).
        Initializes an empty list `index_id_to_db_id` to store the mapping
        between internal index IDs and external database IDs.
        """
        self.index = faiss.IndexFlatIP(vector_sz)
        self.device = 'cuda'  # Force GPU use in Colab
        self.index_id_to_db_id = []

    def index_data(self, ids, embeddings):
        """
        Add embeddings to the index by updating the index and mapping IDs.
        It trains the index using the provided embeddings and print the total
          number of indexed data points.
        Args:
            ids (int): A list of IDs corresponding to the embeddings.
            embeddings (np.array): A NumPy array of embedding vectors.
        """
        self._update_id_mapping(ids)
        embeddings = embeddings.astype('float32')
        if not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)
        print(f'Total data indexed {self.index.ntotal}')

    def search_knn(self, query_vectors: np.array, top_docs: int,
                   index_batch_size: int = 8) -> List[Tuple[List[object], List[float]]]:
        """
        Search for the nearest neighbors to the given query vectors.
        Args:
            query_vectors (np.array): A NumPy array of query embedding vectors.
            top_docs (int): The number of nearest neighbors to retrieve.
            index_batch_size (int): The batch size for indexing.

        Returns:
            result (tuples): A list of tuples containing the IDs and scores of
              the nearest neighbors.
        """
        query_vectors = query_vectors.astype('float32')
        result = []
        nbatch = (len(query_vectors)-1) // index_batch_size + 1
        for k in tqdm(range(nbatch)):
            start_idx = k*index_batch_size
            end_idx = min((k+1)*index_batch_size, len(query_vectors))
            q = query_vectors[start_idx: end_idx]
            scores, indexes = self.index.search(q, top_docs)
            # convert to external ids
            db_ids = [[str(self.index_id_to_db_id[i]) for i in query_top_idxs]
                      for query_top_idxs in indexes]
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])
        return result

    def serialize(self, dir_path):
        """
        Saves the index and metadata to disk.
        Args:
            dir_path (str): The directory path to save the index and metadata.
        """
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'index_meta.faiss')
        print(f'Serializing index to {index_file}, meta data to {meta_file}')
        save_index = self.index
        faiss.write_index(save_index, index_file)
        with open(meta_file, mode='wb') as f:
            pickle.dump(self.index_id_to_db_id, f)

    def deserialize_from(self, dir_path):
        """
        Loads the index and metadata from disk.
        Args:
            dir_path (str): The directory path to load the index and metadata.
        """
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'index_meta.faiss')
        print(f'Loading index from {index_file}, meta data from {meta_file}')

        self.index = faiss.read_index(index_file)
        print(f'Loaded index of type {type(self.index)} and size {self.index.ntotal}')

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert len(self.index_id_to_db_id) == self.index.ntotal, \
            'Deserialized index_id_to_db_id should match faiss index size'

    def _update_id_mapping(self, db_ids: List):
        """
        Update the mapping between internal index IDs and external database IDs.
        Args:
            db_ids (List): A list of external database IDs.
        """
        self.index_id_to_db_id.extend(db_ids)

    def reset(self):
        """
        Reset the index and clear the mapping.
        """
        self.index.reset()
        self.index_id_to_db_id = []
        print(f'Index reset, total data indexed {self.index.ntotal}')
