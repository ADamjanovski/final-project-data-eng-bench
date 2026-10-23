import json

import boto3
from botocore.exceptions import ClientError

from src.config import RagConfig

NON_FILTERABLE_METADATA = ["trigger", "action", "retrieval_text"]
VECTOR_BATCH_SIZE = 500


def build_clients(region: str):
    return (
        boto3.client("bedrock-runtime", region_name=region),
        boto3.client("s3vectors", region_name=region),
    )


def ensure_store(
    s3vectors,
    bucket: str,
    index: str,
    config: RagConfig,
) -> None:
    try:
        s3vectors.get_vector_bucket(vectorBucketName=bucket)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "NotFoundException":
            raise
        s3vectors.create_vector_bucket(vectorBucketName=bucket)

    try:
        response = s3vectors.get_index(
            vectorBucketName=bucket,
            indexName=index,
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "NotFoundException":
            raise
        s3vectors.create_index(
            vectorBucketName=bucket,
            indexName=index,
            dataType="float32",
            dimension=config.dimension,
            distanceMetric="cosine",
            metadataConfiguration={
                "nonFilterableMetadataKeys": NON_FILTERABLE_METADATA,
            },
        )
        return

    stored = response["index"]
    if (
        stored["dataType"] != "float32"
        or stored["dimension"] != config.dimension
        or stored["distanceMetric"] != "cosine"
    ):
        raise ValueError(
            "the existing index does not match the RAG configuration"
        )


def embed_texts(
    bedrock,
    texts: list[str],
    config: RagConfig,
) -> list[list[float]]:
    embeddings = []
    for text in texts:
        response = bedrock.invoke_model(
            modelId=config.embedding_model,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(
                {
                    "inputText": text,
                    "dimensions": config.dimension,
                    "normalize": True,
                }
            ),
        )
        vector = json.loads(response["body"].read())["embedding"]
        if len(vector) != config.dimension:
            raise RuntimeError("the embedding model returned the wrong dimension")
        embeddings.append(vector)
    return embeddings


def store_lesson_vectors(
    s3vectors,
    bucket: str,
    index: str,
    lessons: list[dict],
    embeddings: list[list[float]],
    *,
    update_step: int = 0,
) -> None:
    if len(lessons) != len(embeddings):
        raise ValueError("each lesson must have one embedding")
    for start in range(0, len(lessons), VECTOR_BATCH_SIZE):
        batch = lessons[start : start + VECTOR_BATCH_SIZE]
        vectors = embeddings[start : start + VECTOR_BATCH_SIZE]
        s3vectors.put_vectors(
            vectorBucketName=bucket,
            indexName=index,
            vectors=[
                {
                    "key": lesson["id"],
                    "data": {"float32": vector},
                    "metadata": _lesson_metadata(lesson, update_step),
                }
                for lesson, vector in zip(batch, vectors, strict=True)
            ],
        )


def save_retrieval_lessons(
    bedrock,
    s3vectors,
    bucket: str,
    index: str,
    lessons: list[dict],
    config: RagConfig,
    *,
    update_step: int = 0,
) -> None:
    for lesson in lessons:
        _validate_lesson(lesson)
        if not isinstance(lesson.get("retrieval_text"), str):
            raise ValueError(f"lesson {lesson['id']} has no retrieval text")
    embeddings = embed_texts(
        bedrock,
        [lesson["retrieval_text"] for lesson in lessons],
        config,
    )
    store_lesson_vectors(
        s3vectors,
        bucket,
        index,
        lessons,
        embeddings,
        update_step=update_step,
    )


def load_lessons(s3vectors, bucket: str, index: str) -> list[dict]:
    lessons = []
    next_token = None
    while True:
        request = {
            "vectorBucketName": bucket,
            "indexName": index,
            "maxResults": 1000,
            "returnMetadata": True,
        }
        if next_token:
            request["nextToken"] = next_token
        response = s3vectors.list_vectors(**request)
        lessons.extend(
            {"id": vector["key"], **vector.get("metadata", {})}
            for vector in response["vectors"]
        )
        next_token = response.get("nextToken")
        if not next_token:
            return sorted(lessons, key=lambda lesson: lesson["id"])


def delete_lessons(
    s3vectors,
    bucket: str,
    index: str,
    lesson_ids: list[str],
) -> None:
    for start in range(0, len(lesson_ids), 500):
        keys = lesson_ids[start : start + 500]
        if keys:
            s3vectors.delete_vectors(
                vectorBucketName=bucket,
                indexName=index,
                keys=keys,
            )


def sync_lessons(
    bedrock,
    s3vectors,
    bucket: str,
    index: str,
    lessons: list[dict],
    config: RagConfig,
    *,
    update_step: int,
) -> None:
    stored = load_lessons(s3vectors, bucket, index)
    _write_lesson_changes(
        bedrock,
        s3vectors,
        bucket,
        index,
        stored,
        lessons,
        config,
        update_step,
    )


def _write_lesson_changes(
    bedrock,
    s3vectors,
    bucket: str,
    index: str,
    stored: list[dict],
    lessons: list[dict],
    config: RagConfig,
    update_step: int,
) -> None:
    stored_by_id = {lesson["id"]: lesson for lesson in stored}
    lessons_by_id = {lesson["id"]: lesson for lesson in lessons}
    removed = sorted(stored_by_id.keys() - lessons_by_id.keys())
    changed = [
        lesson
        for lesson in lessons
        if lesson["id"] not in stored_by_id
        or any(
            lesson.get(field) != stored_by_id[lesson["id"]].get(field)
            for field in (
                "trigger",
                "action",
                "retrieval_text",
                "lesson_type",
                "source_tasks",
            )
        )
    ]
    delete_lessons(s3vectors, bucket, index, removed)
    save_retrieval_lessons(
        bedrock,
        s3vectors,
        bucket,
        index,
        changed,
        config,
        update_step=update_step,
    )


def retrieve_lessons(
    bedrock,
    s3vectors,
    bucket: str,
    index: str,
    query: str,
    config: RagConfig,
    *,
    top_k: int | None = None,
) -> list[dict]:
    vector = embed_texts(bedrock, [query], config)[0]
    response = s3vectors.query_vectors(
        vectorBucketName=bucket,
        indexName=index,
        queryVector={"float32": vector},
        topK=top_k or config.top_k,
        returnDistance=True,
        returnMetadata=True,
    )
    return [
        {
            "id": item["key"],
            "distance": item.get("distance"),
            **item.get("metadata", {}),
        }
        for item in response["vectors"]
    ]


def _validate_lesson(lesson: dict) -> None:
    required = {"id", "trigger", "action", "lesson_type", "source_tasks"}
    missing = required - lesson.keys()
    if missing:
        raise ValueError(f"lesson is missing: {', '.join(sorted(missing))}")


def _lesson_metadata(lesson: dict, update_step: int) -> dict:
    metadata = {
        "trigger": lesson["trigger"],
        "action": lesson["action"],
        "lesson_type": lesson["lesson_type"],
        "source_tasks": lesson["source_tasks"],
        "update_step": update_step,
    }
    if lesson.get("retrieval_text"):
        metadata["retrieval_text"] = lesson["retrieval_text"]
    return metadata
