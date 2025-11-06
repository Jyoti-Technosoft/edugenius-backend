

import uuid
import json,os
import random
from collections import OrderedDict
from datetime import datetime
from typing import  Dict, Any,Optional
from drive_uploader import load_env
import numpy as np

from qdrant_client import QdrantClient, models
load_env()  # make sure env is loaded before using
# Configuration - set these in your environment for Qdrant Cloud
QDRANT_URL =os.environ.get("QDRANT_URL")  # change to your cluster URL
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
VECTOR_DIM = 384
DISTANCE = models.Distance.COSINE
TIMEOUT = 60.0

COLLECTION_MCQ = "mcq_collection"
COLLECTION_QUESTIONS = "questions_collection"
COLLECTION_TEST_SESSIONS = "test_sessions_collection"
COLLECTION_SUBMITTED = "submitted_tests_collection"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=TIMEOUT)

def ensure_collections():
    existing = [c.name for c in client.get_collections().collections]

    vector_params = models.VectorParams(size=VECTOR_DIM, distance=DISTANCE)

    # ---- CREATE COLLECTIONS (new API) ----
    if COLLECTION_MCQ not in existing:
        client.create_collection(
            collection_name=COLLECTION_MCQ,
            vectors_config=vector_params
        )

    if COLLECTION_QUESTIONS not in existing:
        client.create_collection(
            collection_name=COLLECTION_QUESTIONS,
            vectors_config=vector_params
        )

    if COLLECTION_TEST_SESSIONS not in existing:
        client.create_collection(
            collection_name=COLLECTION_TEST_SESSIONS,
            vectors_config=vector_params
        )

    if COLLECTION_SUBMITTED not in existing:
        client.create_collection(
            collection_name=COLLECTION_SUBMITTED,
            vectors_config=vector_params
        )

    # ---- ADD PAYLOAD INDEXES (same as before) ----
    def _safe_index(col, field):
        try:
            client.create_payload_index(
                collection_name=col,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD
            )
        except Exception:
            pass

    _safe_index(COLLECTION_MCQ, "generatedQAId")
    _safe_index(COLLECTION_MCQ, "userId")

    _safe_index(COLLECTION_QUESTIONS, "generatedQAId")
    _safe_index(COLLECTION_QUESTIONS, "userId")
    _safe_index(COLLECTION_QUESTIONS, "questionId")

    _safe_index(COLLECTION_TEST_SESSIONS, "userId")
    _safe_index(COLLECTION_TEST_SESSIONS, "testId")

    _safe_index(COLLECTION_SUBMITTED, "userId")
    _safe_index(COLLECTION_SUBMITTED, "testId")


ensure_collections()

def embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    return [np.random.rand(VECTOR_DIM).tolist() for _ in texts]

def _to_payload_for_bank(meta: Dict[str, Any]):
    p = dict(meta)
    for k, v in list(p.items()):
        if isinstance(v, (dict, list)):
            p[k] = json.dumps(v)
    return p

def _to_payload_for_question(meta: Dict[str, Any]):
    p = dict(meta)
    if "options" in p and not isinstance(p["options"], str):
        p["options"] = json.dumps(p["options"])
    return p

def store_mcqs(userId, title, description, mcqs, pdf_file, createdAt):
    userIdClean = str(userId).strip().lower()
    generatedQAId = str(uuid.uuid4())

    metadata_for_bank = {
        "userId": userIdClean,
        "title": title,
        "generatedQAId": generatedQAId,
        "description": description,
        "file_name": pdf_file,
        "createdAt": createdAt
    }

    bank_vector = embed(f"{title} {description}")[0]
    bank_point = models.PointStruct(id=generatedQAId, vector=bank_vector, payload=_to_payload_for_bank(metadata_for_bank))
    client.upsert(collection_name=COLLECTION_MCQ, points=[bank_point])

    question_points = []
    for i, mcq in enumerate(mcqs):
        questionId = str(uuid.uuid4())
        mcq['generatedQAId'] = generatedQAId
        mcq['questionId'] = questionId
        mcq['userId'] = userIdClean
        mcq['documentIndex'] = i

        options_json = json.dumps(mcq.get("options", {}))
        question_text = mcq.get("question", "") or ""
        q_vec = embed(question_text)[0]

        q_meta = OrderedDict([
            ("questionId", questionId),
            ("generatedQAId", generatedQAId),
            ("userId", userIdClean),
            ("question", mcq.get("question", "")),
            ("noise", mcq.get("noise", "")),
            ("image", mcq.get("image")),
            ("passage", mcq.get("passage") or ""),
            ("options", options_json),
            ("answer", mcq.get("answer", "")),
            ("documentIndex", i)
        ])

        point = models.PointStruct(id=questionId, vector=q_vec, payload=_to_payload_for_question(q_meta))
        question_points.append(point)

        if len(question_points) >= 256:
            client.upsert(collection_name=COLLECTION_QUESTIONS, points=question_points)
            question_points = []

    if question_points:
        client.upsert(collection_name=COLLECTION_QUESTIONS, points=question_points)

    return generatedQAId

def fetch_mcqs(userId: str = None, generatedQAId: str = None):
    results = []

    # 🟢 Fetch by generatedQAId
    if generatedQAId:
        filt = models.Filter(
            must=[
                models.FieldCondition(key="generatedQAId", match=models.MatchValue(value=generatedQAId))
            ]
        )
        dummy_vector = [0.0] * VECTOR_DIM

        # Fetch parent bank (metadata)
        bank_hits = client.search(
            collection_name=COLLECTION_MCQ,
            query_vector=dummy_vector,
            query_filter=filt,
            limit=1,
            with_payload=True
        )
        if not bank_hits:
            return []

        bank = bank_hits[0].payload
        if not bank:
            return []

        # Fetch all questions belonging to this generatedQAId
        hits = client.scroll(
            collection_name=COLLECTION_QUESTIONS,
            scroll_filter=filt,
            limit=1000,
            with_payload=True,
        )[0]  # first element = list of points

        mcq_list = []
        for h in hits:
            payload = h.payload if hasattr(h, "payload") else h.get("payload", {})
            if payload and "options" in payload and isinstance(payload["options"], str):
                try:
                    payload["options"] = json.loads(payload["options"])
                except Exception:
                    pass

            ordered_mcq = OrderedDict([
                ("questionId", payload.get("questionId")),
                ("generatedQAId", payload.get("generatedQAId")),
                ("userId", payload.get("userId")),
                ("question", payload.get("question")),
                ("options", payload.get("options")),
                ("answer", payload.get("answer")),
                ("passage", payload.get("passage") or ""),
                ("noise", payload.get("noise")),
                ("image", payload.get("image")),
                ("documentIndex", payload.get("documentIndex")),
            ])
            mcq_list.append(ordered_mcq)

        mcq_list = sorted(mcq_list, key=lambda x: int(x["documentIndex"]) if x.get("documentIndex") else float("inf"))
        bank["mcqs"] = mcq_list

        results.append({
            "id": generatedQAId,
            "document": bank.get("title", ""),
            "metadata": bank
        })
        return results

    # 🟢 Fetch all MCQ banks by userId
    elif userId:
        userIdClean = str(userId).strip().lower()
        filt = models.Filter(
            must=[
                models.FieldCondition(key="userId", match=models.MatchValue(value=userIdClean))
            ]
        )
        dummy_vector = [0.0] * VECTOR_DIM
        banks = client.search(
            collection_name=COLLECTION_MCQ,
            query_vector=dummy_vector,
            query_filter=filt,
            limit=1000,
            with_payload=True,
        )
        for b in banks:
            payload = b.payload if hasattr(b, "payload") else b.get("payload", {})
            gen = payload.get("generatedQAId")
            if gen:
                fetched = fetch_mcqs(generatedQAId=gen)
                if fetched:
                    results.extend(fetched)
        return results

    return []

def fetch_random_mcqs(generatedQAId: str, num_questions: int = None):
    records = fetch_mcqs(generatedQAId=generatedQAId)
    if not records:
        return []
    record = records[0]
    original_mcqs = record.get("metadata", {}).get("mcqs", [])
    if not original_mcqs:
        return []
    selected = original_mcqs
    if num_questions and num_questions < len(original_mcqs):
        selected = random.sample(original_mcqs, num_questions)
    formatted = []
    for mcq in selected:
        mcq_copy = mcq.copy()
        options = mcq_copy.get("options", {})
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except Exception:
                options = {}
        items = list(options.items())
        random.shuffle(items)
        mcq_copy["options"] = OrderedDict(items)
        formatted.append(mcq_copy)
    record["metadata"]["mcqs"] = formatted
    return [record]

def store_test_session(userId, testId, testTitle, totalTime, createdAt, mcqs_data):
    try:
        userIdClean = str(userId).strip().lower()
        payload = {"userId": userIdClean, "testTitle": testTitle, "totalTime": totalTime, "createdAt": createdAt, "questions": json.dumps(mcqs_data), "testId": testId}
        vec = embed(f"test_session:{testId}")[0]
        p = models.PointStruct(id=testId, vector=vec, payload=payload)
        client.upsert(collection_name=COLLECTION_TEST_SESSIONS, points=[p])
        return True
    except Exception as e:
        print("store_test_session error:", e)
        return False

def _extract_payload(point):
    """Safely extract payload from a qdrant point that may be either an object or a dict."""
    if not point:
        return {}
    # record-like object with .payload
    payload = getattr(point, "payload", None)
    if payload is not None:
        return payload or {}
    # dict-like
    if isinstance(point, dict):
        return point.get("payload", {}) or {}
    return {}

def _extract_id(point):
    """Safely extract id from qdrant point (object or dict)."""
    if not point:
        return None
    pid = getattr(point, "id", None)
    if pid is not None:
        return pid
    if isinstance(point, dict):
        return point.get("id")
    return None

def _normalize_scroll_result(res):
    """
    client.scroll may return a list or a tuple where first element is points list.
    Normalize to a flat list of points/dicts.
    """
    if res is None:
        return []
    if isinstance(res, (list, tuple)) and len(res) > 0:
        # common qdrant pattern: ([points], next_page)
        if isinstance(res[0], list):
            return res[0]
        # sometimes it's directly list of points
        return list(res)
    # fallback single point
    return [res]

def _ensure_json_field(payload, key):
    """If payload[key] is a JSON-encoded string, decode it; otherwise leave as-is."""
    if key in payload and isinstance(payload[key], str):
        try:
            payload[key] = json.loads(payload[key])
        except Exception:
            # keep original string if it fails
            payload[key] = payload[key]
    return payload

# ---------- Functions (rewritten) ----------

def fetch_test_by_testId(testId):
    try:
        result = client.retrieve(
            collection_name=COLLECTION_TEST_SESSIONS,
            ids=[testId],  # Must be a list
            with_payload=True
        )

        if not result:
            return None

        p = result[0]  # Retrieve the first (and only) result
        payload = _extract_payload(p)
        payload = _ensure_json_field(payload, "questions")
        payload["testId"] = payload.get("testId") or _extract_id(p)
        return payload

    except Exception as e:
        print("fetch_test_by_testId error:", e)
        return None


def test_sessions_by_userId(userId):
    try:
        userIdClean = str(userId).strip().lower()
        filt = models.Filter(must=[models.FieldCondition(key="userId", match=models.MatchValue(value=userIdClean))])
        # Using dummy vector because search requires a vector query in your client usage
        dummy_vector = [0.0] * VECTOR_DIM
        hits = client.search(
            collection_name=COLLECTION_TEST_SESSIONS,
            query_vector=dummy_vector,
            query_filter=filt,
            limit=1000,
            with_payload=True
        )
        sessions = []
        for h in hits:
            payload = _extract_payload(h)
            payload = _ensure_json_field(payload, "questions")
            payload["testId"] = payload.get("testId") or _extract_id(h)
            sessions.append(payload)
        return sessions
    except Exception as e:
        print("test_sessions_by_userId error:", e)
        return None

def store_submitted_test(userId, testId, testTitle, timeSpent, totalTime, submittedAt, detailed_results, score, total_questions):
    try:
        userIdClean = str(userId).strip().lower()
        payload = {
            "userId": userIdClean,
            "testId": testId,
            "testTitle": testTitle,
            "timeSpent": timeSpent,
            "totalTime": totalTime,
            "submittedAt": submittedAt,
            "score": score,
            "total_questions": total_questions,
            "detailed_results": json.dumps(detailed_results)
        }
        vec = embed(f"submitted_test:{testId}")[0]
        p = models.PointStruct(id=testId, vector=vec, payload=payload)
        client.upsert(collection_name=COLLECTION_SUBMITTED, points=[p])
        return True
    except Exception as e:
        print("store_submitted_test error:", e)
        return False

def submitted_tests_by_userId(userId):
    try:
        userIdClean = str(userId).strip().lower()
        filt = models.Filter(must=[models.FieldCondition(key="userId", match=models.MatchValue(value=userIdClean))])
        dummy_vector = [0.0] * VECTOR_DIM
        hits = client.search(
            collection_name=COLLECTION_SUBMITTED,
            query_vector=dummy_vector,
            query_filter=filt,
            limit=1000,
            with_payload=True
        )
        sessions = []
        for h in hits:
            payload = _extract_payload(h)
            if "detailed_results" in payload and isinstance(payload["detailed_results"], str):
                try:
                    payload["detailed_results"] = json.loads(payload["detailed_results"])
                except Exception:
                    payload["detailed_results"] = payload["detailed_results"]
            payload["testId"] = payload.get("testId") or _extract_id(h)
            sessions.append(payload)
        return sessions
    except Exception as e:
        print("submitted_tests_by_userId error:", e)
        return None

def delete_single_question(questionId):
    try:
        client.delete(collection_name=COLLECTION_QUESTIONS, points=[questionId])
        return True
    except Exception as e:
        print("delete_single_question error:", e)
        return False

def update_single_question(questionId, updated_data):
    try:
        existing = client.retrieve(collection_name=COLLECTION_QUESTIONS, ids=[questionId], with_payload=True)
        if not existing:
            return False
        existing_payload = _extract_payload(existing)
        # preserve important ids
        updated_data['questionId'] = questionId
        updated_data['generatedQAId'] = existing_payload.get("generatedQAId")
        updated_data['userId'] = existing_payload.get("userId")
        # normalize options
        if 'options' in updated_data:
            if isinstance(updated_data['options'], (dict, list)):
                updated_data['options'] = json.dumps(updated_data['options'])
            elif updated_data['options'] is None:
                updated_data['options'] = json.dumps({})
        payload = _to_payload_for_question(updated_data)
        vec = embed(updated_data.get("question", ""))[0]
        p = models.PointStruct(id=questionId, vector=vec, payload=payload)
        client.upsert(collection_name=COLLECTION_QUESTIONS, points=[p])
        return True
    except Exception as e:
        print("update_single_question error:", e)
        return False

def add_single_question(generatedQAId, question_data):
    try:
        bank = client.get_point(collection_name=COLLECTION_MCQ, point_id=generatedQAId, with_payload=True)
        if not bank:
            print(f"Question bank {generatedQAId} not found")
            return False
        bank_payload = _extract_payload(bank)
        questionId = str(uuid.uuid4())
        question_data['questionId'] = questionId
        question_data['generatedQAId'] = generatedQAId
        question_data['userId'] = bank_payload.get("userId") or "unknown"
        question_text = question_data.get("question", "") or ""
        if 'options' in question_data and isinstance(question_data['options'], (dict, list)):
            question_data['options'] = json.dumps(question_data['options'])
        filtered_metadata = {k: v for k, v in question_data.items() if v is not None}
        payload = _to_payload_for_question(filtered_metadata)
        vec = embed(question_text)[0]
        p = models.PointStruct(id=questionId, vector=vec, payload=payload)
        client.upsert(collection_name=COLLECTION_QUESTIONS, points=[p])
        return True
    except Exception as e:
        print("add_single_question error:", e)
        return False

def store_mcqs_for_manual_creation(userId, title, description, mcqs):
    try:
        generatedQAId = str(uuid.uuid4())
        userIdClean = str(userId).strip().lower()
        createdAt = datetime.now().isoformat()
        pdf_file = "MANUAL_CREATION"
        metadata_for_bank = {
            "userId": userIdClean,
            "title": title,
            "generatedQAId": generatedQAId,
            "description": description,
            "file_name": pdf_file,
            "createdAt": createdAt
        }
        bank_vector = embed(f"{title} {description}")[0]
        bank_point = models.PointStruct(id=generatedQAId, vector=bank_vector, payload=_to_payload_for_bank(metadata_for_bank))
        client.upsert(collection_name=COLLECTION_MCQ, points=[bank_point])

        points = []
        for mcq in mcqs:
            questionId = str(uuid.uuid4())
            mcq['generatedQAId'] = generatedQAId
            mcq['questionId'] = questionId
            mcq['userId'] = userIdClean
            options_data = mcq.get("options", "{}")
            question_text = mcq.get("question", "") or ""
            q_meta = OrderedDict([
                ("questionId", questionId),
                ("generatedQAId", generatedQAId),
                ("userId", userIdClean),
                ("question", mcq.get("question") or ""),
                ("noise", mcq.get("noise") or None),
                ("image", mcq.get("image") or None),
                ("passage", mcq.get("passage") or ""),
                ("options", options_data),
                ("answer", mcq.get("answer") or ""),
                ("documentIndex", mcq.get("documentIndex"))
            ])
            payload = _to_payload_for_question(q_meta)
            vec = embed(question_text)[0]
            p = models.PointStruct(id=questionId, vector=vec, payload=payload)
            points.append(p)

        if points:
            client.upsert(collection_name=COLLECTION_QUESTIONS, points=points)
        return generatedQAId
    except Exception as e:
        print("store_mcqs_for_manual_creation error:", e)
        return None

def delete_mcq_bank(generatedQAId):
    try:
        bank = client.retrieve(collection_name=COLLECTION_MCQ, ids=[generatedQAId], with_payload=True)
        if not bank:
            print("bank not found")
            return False
        filt = models.Filter(must=[models.FieldCondition(key="generatedQAId", match=models.MatchValue(value=generatedQAId))])
        hits = client.scroll(collection_name=COLLECTION_QUESTIONS, limit=1000, scroll_filter=filt)
        hits = _normalize_scroll_result(hits)
        ids_to_delete = []
        for h in hits:
            # h may be dict or object; normalize id
            if isinstance(h, dict):
                pid = h.get("id") or h.get("point_id") or h.get("pointId")
            else:
                pid = getattr(h, "id", None) or getattr(h, "point_id", None)
            if pid:
                ids_to_delete.append(pid)
        if ids_to_delete:
            client.delete(collection_name=COLLECTION_QUESTIONS, points=ids_to_delete)
        client.delete(collection_name=COLLECTION_MCQ, points=[generatedQAId])
        return True
    except Exception as e:
        print("delete_mcq_bank error:", e)
        return False

def delete_test_session_by_id(testId):
    try:
        client.delete(
            collection_name=COLLECTION_SUBMITTED,
            points_selector=models.PointIdsList(
                points=[testId]
            )
        )
        return True
    except Exception as e:
        print("delete_test_session_by_id error:", e)
        return False

def delete_submitted_test_by_id(testId):
    try:
        client.delete(
            collection_name=COLLECTION_SUBMITTED,
            points_selector=models.PointIdsList(
                points=[testId]
            )
        )
        return True
    except Exception as e:
        print("delete_submitted_test_by_id error:", e)
        return False

def update_test_session(testId, updated_metadata):
    try:
        existing = client.retrieve(collection_name=COLLECTION_TEST_SESSIONS, ids=[testId], with_payload=True)
        if not existing:
            return False
        payload = _extract_payload(existing)
        if 'questions' in updated_metadata and isinstance(updated_metadata['questions'], (list, dict)):
            updated_metadata['questions'] = json.dumps(updated_metadata['questions'])
        payload.update(updated_metadata)
        vec = embed(payload.get("testTitle", f"test_session:{testId}"))[0]
        p = models.PointStruct(id=testId, vector=vec, payload=payload)
        client.upsert(collection_name=COLLECTION_TEST_SESSIONS, points=[p])
        return True
    except Exception as e:
        print("update_test_session error:", e)
        return False

def update_question_bank_metadata(generatedQAId: str, title: Optional[str] = None, description: Optional[str] = None):
    """
    Safely update only title and/or description for a question bank in Qdrant.
    This *does not* overwrite or remove existing mcqs or other metadata fields.
    Returns a dict: {"success": bool, "title_updated": bool, "description_updated": bool, "error": str (optional)}
    """
    result = {"success": False, "title_updated": False, "description_updated": False}
    try:
        # Retrieve the bank point (client.retrieve returns a list)
        bank_points = client.retrieve(collection_name=COLLECTION_MCQ, ids=[generatedQAId], with_payload=True)
        if not bank_points or len(bank_points) == 0:
            return result

        bank_point = bank_points[0]
        payload = _extract_payload(bank_point)
        if not payload:
            return result

        updated = False
        if title is not None and payload.get("title") != title:
            payload["title"] = title
            result["title_updated"] = True
            updated = True
        if description is not None and payload.get("description") != description:
            payload["description"] = description
            result["description_updated"] = True
            updated = True

        if not updated:
            result["success"] = True
            return result

        # Recompute the embedding for the bank (optional but keeps vectors consistent)
        new_vec = embed(f"{payload.get('title','')} {payload.get('description','')}")[0]

        # Upsert the point back with same id, new vector and merged payload
        p = models.PointStruct(id=generatedQAId, vector=new_vec, payload=payload)
        client.upsert(collection_name=COLLECTION_MCQ, points=[p], wait=True)

        result["success"] = True
        return result
    except Exception as e:
        print("update_question_bank_metadata error:", e)
        result["error"] = str(e)
        return result
