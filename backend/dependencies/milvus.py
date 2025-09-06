from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient, connections

from backend.settings import settings


class Milvus:
    def __init__(self):
        self.client = MilvusClient(
            uri=settings.MILVUS_URI,
            token=settings.MILVUS_TOKEN,
        )
        self.collection_name = settings.MILVUS_COLLECTION

        if not self.client.has_collection(collection_name=self.collection_name):
            self.create_collection()
            self.create_index()

        self.get_info()

    def get_info(self):
        print(
            f"Vector store stats: {self.client.get_collection_stats(self.collection_name)}"
        )
        print(f"Vector store indexes: {self.client.list_indexes(self.collection_name)}")

    def create_collection(self):
        self.client.create_collection(
            collection_name=self.collection_name,
            vector_field_name="embedding",
            dimension=512,
            metric_type="L2",  # Euclidean
            schema=CollectionSchema(
                fields=[
                    FieldSchema(
                        name="id",
                        dtype=DataType.INT64,
                        is_primary=True,
                        auto_id=True,
                    ),
                    FieldSchema(
                        name="image_id",
                        dtype=DataType.INT64,
                    ),
                    FieldSchema(
                        name="face_id",
                        dtype=DataType.INT64,
                    ),
                    FieldSchema(
                        name="name",
                        dtype=DataType.VARCHAR,
                        max_length=255,
                    ),
                    FieldSchema(
                        name="embedding",
                        description="Face embeddings",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=512,
                    ),
                    FieldSchema(
                        name="file_path",
                        dtype=DataType.VARCHAR,
                        max_length=512,
                        description=("Storage path or URL of the image"),
                    ),
                    FieldSchema(
                        name="timestamp",
                        description="Timestamp stored as Unix epoch time in milliseconds",
                        dtype=DataType.INT64,
                    ),
                ],
                description="Face embeddings of customer",
            ),
        )

    def create_index(self):
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            index_name="embedding_index",
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="L2",  # Euclidean
        )
        index_params.add_index(
            index_name="name_index",
            field_name="name",
        )
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params,
        )
        self.client.load_collection(collection_name=self.collection_name)

    def insert_data(self, data):
        count = self.client.insert(collection_name=self.collection_name, data=data)
        return count

    def search_data(self, query_embedding):
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="embedding",
            search_params={
                "params": {
                    "radius": settings.MILVUS_RADIUS,
                    "range_filter": settings.MILVUS_RANGE_FILTER,
                }
            },
            limit=settings.TOP_K,
            output_fields=["name", "file_path"],
        )
        return results


milvus = Milvus()
