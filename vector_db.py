import uuid,re
import json,os
import random
from collections import OrderedDict
from datetime import datetime
import collections
from drive_uploader import load_env
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from qdrant_client import QdrantClient, models
load_env()  # make sure env is loaded before using
# Configuration - set these in your environment for Qdrant Cloud
QDRANT_URL =os.environ.get("QDRANT_URL")  # change to your cluster URL
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
VECTOR_DIM = 384
DISTANCE = models.Distance.COSINE
TIMEOUT = 100.0

COLLECTION_MCQ = "mcq_collection"
COLLECTION_QUESTIONS = "questions_collection"
COLLECTION_TEST_SESSIONS = "test_sessions_collection"
COLLECTION_SUBMITTED = "submitted_tests_collection"
COLLECTION_SUBSCRIPTIONS = "user_subscriptions"
COLLECTION_SOURCES = "source_materials_collection"
COLLECTION_USERS = "users_collection"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=TIMEOUT)

def ensure_collections():
    try:
        # Try to connect and fetch existing collections
        colls = client.get_collections().collections
        existing = [c.name for c in colls]
    except Exception as e:
        print("\n[ERROR] Could NOT connect to Qdrant Cloud during ensure_collections()")
        print("Reason:", str(e))
        print("Skipping collection creation to avoid crashing the backend.\n")
        return  # ← Do NOT crash app

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

    if COLLECTION_SOURCES not in existing:
        client.create_collection(
            collection_name=COLLECTION_SOURCES,
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

    if COLLECTION_SUBSCRIPTIONS not in existing:
        client.create_collection(
            collection_name=COLLECTION_SUBSCRIPTIONS,
            vectors_config=vector_params
        )

    if COLLECTION_USERS not in existing:
        client.create_collection(
            collection_name=COLLECTION_USERS,
            vectors_config=vector_params
        )





    # ---- ADD PAYLOAD INDEXES ----
    def _safe_index(col, field, schema=models.PayloadSchemaType.KEYWORD):  # Added 'schema' with a default
        try:
            client.create_payload_index(
                collection_name=col,
                field_name=field,
                field_schema=schema  # Use the passed schema here
            )
        except Exception as e:
            print(f"[DEBUG] Index already exists or error for {field}: {e}")
            pass

    _safe_index(COLLECTION_MCQ, "generatedQAId")
    _safe_index(COLLECTION_MCQ, "userId")

    # Add this line with your other _safe_index calls (around line 76):
    _safe_index(COLLECTION_MCQ, "public", models.PayloadSchemaType.BOOL)


    _safe_index(COLLECTION_QUESTIONS, "generatedQAId")
    _safe_index(COLLECTION_QUESTIONS, "userId")
    _safe_index(COLLECTION_QUESTIONS, "questionId")

    _safe_index(COLLECTION_TEST_SESSIONS, "userId")
    _safe_index(COLLECTION_TEST_SESSIONS, "testId")

    _safe_index(COLLECTION_SUBMITTED, "userId")
    _safe_index(COLLECTION_SUBMITTED, "testId")

    # Add index for fast lookup by User ID
    _safe_index(COLLECTION_SUBSCRIPTIONS, "userId")
    _safe_index(COLLECTION_SUBSCRIPTIONS, "generatedQAId")

    _safe_index(COLLECTION_SOURCES, "type", models.PayloadSchemaType.KEYWORD)
    _safe_index(COLLECTION_USERS, "userId")



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


def _to_payload_for_source(meta: Dict[str, Any]):
    p = dict(meta)
    # Ensure complex objects are strings
    for k, v in p.items():
        if isinstance(v, (dict, list)):
            p[k] = json.dumps(v)
    return p

def update_answer_flag_in_qdrant(generatedQAId: str, all_have_answers: bool):
    """
    Updates the `answerFound` flag in the Qdrant question bank (COLLECTION_MCQ)
    for a specific generatedQAId.
    """
    try:
        client.set_payload(
            collection_name=COLLECTION_MCQ,
            payload={"answerFound": all_have_answers},
            points=[generatedQAId]
        )
        print(f"[INFO] Updated answerFound={all_have_answers} for generatedQAId={generatedQAId}")
    except Exception as e:
        print(f"[WARN] Failed to update answerFound flag in Qdrant: {e}")


# --- Helper: Normalize text safely ---
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    # remove "Answer:", "Correct Answer:", etc.
    s = re.sub(r'^(correct\s*)?(answer|ans)\s*[:\-]*', '', s, flags=re.IGNORECASE)
    # remove brackets, dots, extra punctuation
    s = re.sub(r'^[\(\[\{]+', '', s)
    s = re.sub(r'[\)\]\}]+$', '', s)
    s = s.strip().rstrip('.')
    return s.strip()

# --- Helper: detect invalid/unusable answers ---
def is_invalid_answer(ans: str) -> bool:
    if not ans:
        return True
    ans_low = ans.lower().strip()
    invalid_patterns = [
        "not true", "no answer", "none", "not given", "n/a", "no correct",
        "no option", "no valid", "invalid", "answer is not", "null", "skip"
    ]
    return any(p in ans_low for p in invalid_patterns)

# --- Helper: map normalized answer text to option key ---
def map_answer_to_option(normalized_answer: str, options: dict) -> str | None:
    if not normalized_answer:
        return None

    norm = normalized_answer.strip().lower()

    # (1) Simple letter only ("a", "b", etc.)
    m = re.match(r'^([a-zA-Z])$', norm)
    if m:
        key = f"({m.group(1).upper()})"
        if key in options:
            return key

    # (2) With brackets or dots: "(A)", "A.", etc.
    m = re.match(r'^\(?([a-zA-Z])\)?\.?$', norm)
    if m:
        key = f"({m.group(1).upper()})"
        if key in options:
            return key

    # (3) Check for match in text (e.g. "Paris" → "(B)")
    normalized_options = {}
    for k, v in (options or {}).items():
        if not v:
            continue
        text = str(v).strip()
        # remove "A. " or "A) "
        text = re.sub(r'^[A-Za-z]\s*[\.\)\-:]\s*', '', text)
        normalized_options[k] = text.lower()

    for k, opt_val in normalized_options.items():
        if norm == opt_val or norm in opt_val or opt_val in norm:
            return k

    return None

# --- Helper: clean text from MCQ ---
def clean_mcq_text(mcq):
    """Remove trailing 'Correct' or 'Correct Answer:' etc. from options & answers"""
    cleaned_opts = {}
    for k, v in (mcq.get("options") or {}).items():
        if not v:
            continue
        v_clean = re.sub(r'\b(Correct|Correct Answer|correct answer)\b', '', str(v), flags=re.IGNORECASE).strip()
        cleaned_opts[k] = v_clean
    mcq["options"] = cleaned_opts

    ans = mcq.get("answer", "")
    if ans:
        ans = re.sub(r'^(Correct\s*Answer\s*[:\-]*)', '', ans, flags=re.IGNORECASE)
        mcq["answer"] = ans.strip()
    return mcq

# --- Helper: split multi answers (like "A,B") ---
def split_multi_answers(s: str) -> list:
    if not s:
        return []
    return [p.strip() for p in re.split(r'[,;/\n]+', s) if p.strip()]





def get_image_data(mcq: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Scans the MCQ dictionary for dynamically named Base64 fields (equationNN, figureNN),
    and separates them from the rest of the metadata.

    Returns:
        1. Consolidated dictionary of {field_name: base64_string}
        2. The cleaned MCQ dictionary without those fields.
    """
    image_fields = {}
    cleaned_mcq = {}

    # Regex to identify dynamic image/equation fields
    # Matches keys starting with 'equation' or 'figure' followed by one or more digits
    image_pattern = re.compile(r'^(equation|figure)\d+$', re.IGNORECASE)

    for key, value in mcq.items():
        if isinstance(value, str) and image_pattern.match(key):
            # Key is like 'equation56', value is the base64 string
            image_fields[key] = value
        else:
            # All other fields (question, options, answer, documentIndex, etc.) are kept
            cleaned_mcq[key] = value

    return image_fields, cleaned_mcq






def store_mcqs(userId, userName, title, description, mcqs, pdf_file, createdAt, is_public=False): # Added is_public param
    userIdClean = str(userId).strip().lower()
    generatedQAId = str(uuid.uuid4())

    # --- 1. Added "public" attribute to Bank Metadata ---
    metadata_for_bank = {
        "userId": userIdClean,
        "userName": userName,
        "title": title,
        "generatedQAId": generatedQAId,
        "description": description,
        "file_name": pdf_file,
        "createdAt": createdAt,
        "public": is_public  # New field (Boolean)

    }

    bank_vector = embed(f"{title} {description}")[0]

    question_points = []
    all_have_answers = True
    BATCH_SIZE = 256

    for i, mcq in enumerate(mcqs):
        mcq = clean_mcq_text(mcq)
        image_fields, mcq_cleaned = get_image_data(mcq)

        # --- Answer Processing ---
        raw_answer = mcq.get("answer", "")
        normalized = normalize_text(raw_answer)

        if is_invalid_answer(normalized):
            canonical_answer = ""
        else:
            options = mcq.get("options", {}) or {}
            parts = split_multi_answers(normalized)
            mapped_keys = [
                map_answer_to_option(normalize_text(part), options)
                for part in parts
                if map_answer_to_option(normalize_text(part), options)
            ]
            canonical_answer = ",".join(mapped_keys) if mapped_keys else ""

        if not canonical_answer:
            all_have_answers = False

        questionId = str(uuid.uuid4())

        # --- 2. Added "difficulty" attribute to Question Payload ---
        q_meta = OrderedDict([
            ("questionId", questionId),
            ("generatedQAId", generatedQAId),
            ("userId", userIdClean),
            ("question", mcq.get("question", "")),
            ("noise", mcq.get("noise", "")),
            ("passage", mcq.get("passage") or ""),
            ("options", json.dumps(mcq.get("options", {}))),
            ("knowledge_base", str(mcq.get("knowledge_base", ""))),
            ("question_type", mcq.get("question_type","")),
            ("answer", canonical_answer),
            # New Field: Defaults to "medium" if not provided in mcq object
            ("difficulty", mcq.get("difficulty", "medium")),
            ("documentIndex", i)
        ])

        q_meta.update(image_fields)

        pred_sub = mcq.get("predicted_subject", {})
        pred_con = mcq.get("predicted_concept", {})

        q_meta["predicted_subject"] = OrderedDict([
            ("label", pred_sub.get("label", "")),
            ("confidence", pred_sub.get("confidence", 0))
        ])

        q_meta["predicted_concept"] = OrderedDict([
            ("label", pred_con.get("label", "")),
            ("confidence", pred_con.get("confidence", 0))
        ])

        q_vec = embed(mcq.get("question", "") or "")[0]
        point = models.PointStruct(
            id=questionId,
            vector=q_vec,
            payload=_to_payload_for_question(q_meta)
        )
        question_points.append(point)

        if len(question_points) >= BATCH_SIZE:
            client.upsert(collection_name=COLLECTION_QUESTIONS, points=question_points)
            question_points = []

    if question_points:
        client.upsert(collection_name=COLLECTION_QUESTIONS, points=question_points)

    metadata_for_bank["answerFound"] = all_have_answers
    bank_point = models.PointStruct(
        id=generatedQAId,
        vector=bank_vector,
        payload=_to_payload_for_bank(metadata_for_bank)
    )
    client.upsert(collection_name=COLLECTION_MCQ, points=[bank_point])

    update_answer_flag_in_qdrant(generatedQAId, all_have_answers)
    print(f"[INFO] All answers found: {all_have_answers}")
    return generatedQAId, all_have_answers


#
#
# def fetch_mcqs(userId: str = None, generatedQAId: str = None, page: int = 1, limit: int = 10):
#     """
#     Fetches MCQ banks and questions with pagination support.
#     :param page: The current page number (starts at 1)
#     :param limit: Number of questions to fetch per page
#     """
#     # Calculate offset for pagination
#     offset = (page - 1) * limit
#
#     # 🟢 A. Fetch by generatedQAId (PAGINATED DETAIL VIEW)
#     if generatedQAId:
#         filt = models.Filter(
#             must=[
#                 models.FieldCondition(key="generatedQAId", match=models.MatchValue(value=generatedQAId))
#             ]
#         )
#         dummy_vector = [0.0] * VECTOR_DIM
#
#         # 1. Fetch parent bank (metadata)
#         bank_hits = client.search(
#             collection_name=COLLECTION_MCQ,
#             query_vector=dummy_vector,
#             query_filter=filt,
#             limit=1,
#             with_payload=True
#         )
#         if not bank_hits:
#             return []
#
#         bank = bank_hits[0].payload or {}
#
#         # 2. Get TOTAL COUNT of questions for this bank
#         # This is vital for the frontend to know how many slides to build in total
#         total_questions = client.count(
#             collection_name=COLLECTION_QUESTIONS,
#             count_filter=filt,
#             exact=True
#         ).count
#
#         # 3. 🔥 FIX: Fetch all questions, then slice in memory
#         # Qdrant's scroll 'offset' is a cursor/scroll_id, not a numeric row offset
#         # For proper pagination, we need to fetch all and slice
#         hits, next_page_offset = client.scroll(
#             collection_name=COLLECTION_QUESTIONS,
#             scroll_filter=filt,
#             limit=total_questions,  # Fetch all questions for this bank
#             with_payload=True,
#         )
#
#         all_mcqs = []
#         for h in hits:
#             payload = h.payload if hasattr(h, "payload") else h.get("payload", {})
#
#             # Standard parsing logic
#             if payload and "options" in payload and isinstance(payload["options"], str):
#                 try:
#                     payload["options"] = json.loads(payload["options"])
#                 except Exception:
#                     pass
#
#             standard_keys = [
#                 "questionId", "generatedQAId", "userId", "question",
#                 "options", "answer", "passage", "noise", "documentIndex",
#                 "predicted_subject", "predicted_concept", "knowledge_base", "question_type", "difficulty", "userName", "linkedSourceId"
#             ]
#
#             ordered_mcq = collections.OrderedDict([(k, payload.get(k)) for k in standard_keys])
#
#             # Inject non-standard fields (Base64 images/Equations)
#             for k, v in payload.items():
#                 if k not in ordered_mcq and isinstance(v, str) and len(v) > 20:
#                     ordered_mcq[k] = v
#
#             all_mcqs.append(ordered_mcq)
#
#         # 4. Sort ALL questions by documentIndex
#         all_mcqs = sorted(all_mcqs, key=lambda x: int(x["documentIndex"]) if x.get("documentIndex") else float("inf"))
#
#         # 5. Slice for the requested page
#         mcq_list = all_mcqs[offset:offset + limit]
#
#         # Attach slice to the metadata
#         bank["mcqs"] = mcq_list
#         bank["pagination"] = {
#             "total_count": total_questions,
#             "current_page": page,
#             "limit": limit,
#             "has_more": (offset + limit) < total_questions
#         }
#
#         return [{
#             "id": generatedQAId,
#             "document": bank.get("title", ""),
#             "metadata": bank
#         }]
#
#     # 🟢 B. Fetch all MCQ banks by userId (DASHBOARD LIST VIEW)
#     elif userId:
#         userIdClean = str(userId).strip().lower()
#         bank_filt = models.Filter(
#             must=[
#                 models.FieldCondition(key="userId", match=models.MatchValue(value=userIdClean))
#             ]
#         )
#         dummy_vector = [0.0] * VECTOR_DIM
#
#         # Fetch all banks metadata
#         banks_hits = client.search(
#             collection_name=COLLECTION_MCQ,
#             query_vector=dummy_vector,
#             query_filter=bank_filt,
#             limit=1000,
#             with_payload=True,
#         )
#
#         bank_map = {}
#         generated_ids = []
#         for b in banks_hits:
#             payload = b.payload or {}
#             gen_id = payload.get("generatedQAId")
#             if gen_id:
#                 bank_map[gen_id] = payload
#                 generated_ids.append(gen_id)
#
#         if not generated_ids:
#             return []
#
#         # Fetch questions for all banks
#         question_filt = models.Filter(
#             must=[
#                 models.FieldCondition(key="generatedQAId", match=models.MatchAny(any=generated_ids))
#             ]
#         )
#
#         all_questions_hits = client.scroll(
#             collection_name=COLLECTION_QUESTIONS,
#             scroll_filter=question_filt,
#             limit=10000,
#             with_payload=True,
#         )[0]
#
#         questions_by_bank = collections.defaultdict(list)
#         for h in all_questions_hits:
#             payload = h.payload or {}
#             gen_id = payload.get("generatedQAId")
#             if gen_id:
#                 if "options" in payload and isinstance(payload["options"], str):
#                     try:
#                         payload["options"] = json.loads(payload["options"])
#                     except Exception:
#                         pass
#                 questions_by_bank[gen_id].append(payload)
#
#         results = []
#         for gen_id, bank_payload in bank_map.items():
#             mcq_list = questions_by_bank.get(gen_id, [])
#             mcq_list = sorted(mcq_list, key=lambda x: int(x.get("documentIndex", 9999)))
#             bank_payload["mcqs"] = mcq_list
#             results.append({
#                 "id": gen_id,
#                 "document": bank_payload.get("title", ""),
#                 "metadata": bank_payload
#             })
#
#         return results
#
#     return []


def fetch_mcqs(userId: str = None, generatedQAId: str = None, page: int = 1, limit: int = 10):
    """
    Fetches MCQ banks and questions with pagination support.
    :param page: The current page number (starts at 1)
    :param limit: Number of questions to fetch per page
    """
    # Calculate offset for pagination
    offset = (page - 1) * limit

    # 🟢 A. Fetch by generatedQAId (PAGINATED DETAIL VIEW)
    if generatedQAId:
        filt = models.Filter(
            must=[
                models.FieldCondition(key="generatedQAId", match=models.MatchValue(value=generatedQAId))
            ]
        )
        dummy_vector = [0.0] * VECTOR_DIM

        # 1. Fetch parent bank (metadata)
        bank_hits = client.search(
            collection_name=COLLECTION_MCQ,
            query_vector=dummy_vector,
            query_filter=filt,
            limit=1,
            with_payload=True
        )
        if not bank_hits:
            return []

        bank = bank_hits[0].payload or {}

        # 🔍 FIX 1: Capture Parent Source ID for inheritance
        parent_source_id = bank.get("linkedSourceId")

        # 2. Get TOTAL COUNT of questions for this bank
        total_questions = client.count(
            collection_name=COLLECTION_QUESTIONS,
            count_filter=filt,
            exact=True
        ).count

        # 3. Fetch all questions, then slice in memory
        hits, next_page_offset = client.scroll(
            collection_name=COLLECTION_QUESTIONS,
            scroll_filter=filt,
            limit=total_questions,
            with_payload=True,
        )

        all_mcqs = []
        for h in hits:
            payload = h.payload if hasattr(h, "payload") else h.get("payload", {})

            # Standard parsing logic
            if payload and "options" in payload and isinstance(payload["options"], str):
                try:
                    payload["options"] = json.loads(payload["options"])
                except Exception:
                    pass

            # 🔍 FIX 2: Parent Inheritance Logic
            # If the question lacks a linkedSourceId (is null/empty), inherit it from the Bank metadata
            if not payload.get("linkedSourceId") and parent_source_id:
                payload["linkedSourceId"] = parent_source_id

            # 🔍 FIX 3: Add "linkedSourceId" to standard_keys so it is never filtered out by the 20-char check
            standard_keys = [
                "questionId", "generatedQAId", "userId", "question",
                "options", "answer", "passage", "noise", "documentIndex",
                "predicted_subject", "predicted_concept", "knowledge_base",
                "question_type", "difficulty", "userName", "linkedSourceId"
            ]

            ordered_mcq = collections.OrderedDict([(k, payload.get(k)) for k in standard_keys])

            # Inject non-standard fields (Base64 images/Equations)
            for k, v in payload.items():
                if k not in ordered_mcq and isinstance(v, str) and len(v) > 20:
                    ordered_mcq[k] = v

            all_mcqs.append(ordered_mcq)

        # 4. Sort ALL questions by documentIndex
        all_mcqs = sorted(all_mcqs, key=lambda x: int(x["documentIndex"]) if x.get("documentIndex") else float("inf"))

        # 5. Slice for the requested page
        mcq_list = all_mcqs[offset:offset + limit]

        # Attach slice to the metadata
        bank["mcqs"] = mcq_list
        bank["pagination"] = {
            "total_count": total_questions,
            "current_page": page,
            "limit": limit,
            "has_more": (offset + limit) < total_questions
        }

        return [{
            "id": generatedQAId,
            "document": bank.get("title", ""),
            "metadata": bank
        }]

    # 🟢 B. Fetch all MCQ banks by userId (DASHBOARD LIST VIEW)
    elif userId:
        userIdClean = str(userId).strip().lower()
        bank_filt = models.Filter(
            must=[
                models.FieldCondition(key="userId", match=models.MatchValue(value=userIdClean))
            ]
        )
        dummy_vector = [0.0] * VECTOR_DIM

        # Fetch all banks metadata
        banks_hits = client.search(
            collection_name=COLLECTION_MCQ,
            query_vector=dummy_vector,
            query_filter=bank_filt,
            limit=1000,
            with_payload=True,
        )

        bank_map = {}
        generated_ids = []
        for b in banks_hits:
            payload = b.payload or {}
            gen_id = payload.get("generatedQAId")
            if gen_id:
                bank_map[gen_id] = payload
                generated_ids.append(gen_id)

        if not generated_ids:
            return []

        # Fetch questions for all banks
        question_filt = models.Filter(
            must=[
                models.FieldCondition(key="generatedQAId", match=models.MatchAny(any=generated_ids))
            ]
        )

        all_questions_hits = client.scroll(
            collection_name=COLLECTION_QUESTIONS,
            scroll_filter=question_filt,
            limit=10000,
            with_payload=True,
        )[0]

        questions_by_bank = collections.defaultdict(list)
        for h in all_questions_hits:
            payload = h.payload or {}
            gen_id = payload.get("generatedQAId")
            if gen_id:
                # 🔍 FIX 4: Parent Inheritance Logic for Dashboard View
                # Retrieve the parent bank using the map to get the source ID
                parent_bank = bank_map.get(gen_id, {})
                parent_source_id = parent_bank.get("linkedSourceId")

                if not payload.get("linkedSourceId") and parent_source_id:
                    payload["linkedSourceId"] = parent_source_id

                if "options" in payload and isinstance(payload["options"], str):
                    try:
                        payload["options"] = json.loads(payload["options"])
                    except Exception:
                        pass
                questions_by_bank[gen_id].append(payload)

        results = []
        for gen_id, bank_payload in bank_map.items():
            mcq_list = questions_by_bank.get(gen_id, [])
            mcq_list = sorted(mcq_list, key=lambda x: int(x.get("documentIndex", 9999)))
            bank_payload["mcqs"] = mcq_list
            results.append({
                "id": gen_id,
                "document": bank_payload.get("title", ""),
                "metadata": bank_payload
            })

        return results

    return []



def fetch_random_mcqs(generatedQAId: str, num_questions: int = None):
    records = fetch_mcqs(generatedQAId=generatedQAId, page=1, limit=1000)
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






# def fetch_question_banks_metadata(userId: str):
#     """
#     Fetches a unified list of QBanks for the user's dashboard:
#     1. QBanks they created (Owners).
#     2. QBanks they subscribed to (Downloads).
#     3. EXCLUDES Flashcards explicitly.
#     """
#     userIdClean = str(userId).strip().lower()
#
#     # --- Step 1: Get all Subscribed Bank IDs ---
#     # We allow this to fail gracefully if the collection doesn't exist yet
#     try:
#         sub_hits = client.scroll(
#             collection_name=COLLECTION_SUBSCRIPTIONS,
#             scroll_filter=models.Filter(
#                 must=[models.FieldCondition(key="userId", match=models.MatchValue(value=userIdClean))]
#             ),
#             limit=1000,
#             with_payload=True
#         )[0]
#         subscribed_ids = [h.payload["generatedQAId"] for h in sub_hits if h.payload]
#     except Exception:
#         subscribed_ids = []
#
#     # --- Step 2: Query MCQ Collection for Owned OR Subscribed Banks ---
#     # We use the 'should' clause for an 'OR' logic
#     # We use 'must_not' to filter out Flashcards
#     merged_filter = models.Filter(
#         should=[
#             models.FieldCondition(key="userId", match=models.MatchValue(value=userIdClean)),
#             models.FieldCondition(key="generatedQAId", match=models.MatchAny(any=subscribed_ids))
#         ],
#         must_not=[
#             models.FieldCondition(key="type", match=models.MatchValue(value="FLASHCARD"))
#         ]
#     )
#
#     banks, _ = client.scroll(
#         collection_name=COLLECTION_MCQ,
#         scroll_filter=merged_filter,
#         limit=500,
#         with_payload=True
#     )
#
#     results = []
#     for bank in banks:
#         payload = bank.payload or {}
#         gen_id = payload.get("generatedQAId")
#
#         # Determine Permissions
#         # If the user is NOT the owner, it's read-only
#         is_owner = payload.get("userId") == userIdClean
#
#         # Fresh Count from Questions Collection
#         count = client.count(
#             collection_name=COLLECTION_QUESTIONS,
#             count_filter=models.Filter(
#                 must=[models.FieldCondition(key="generatedQAId", match=models.MatchValue(value=gen_id))]
#             )
#         ).count
#
#         # Get top subject tags
#         tags = compute_subject_tags_for_bank(gen_id)
#
#         results.append({
#             "generatedQAId": gen_id,
#             "title": payload.get("title", ""),
#             "userName": payload.get("userName", ""),
#             "description": payload.get("description", ""),
#             "createdAt": payload.get("createdAt"),
#             "totalQuestions": count,
#             "tags": tags,
#             "isPublic": payload.get("public", False),
#             "canEdit": is_owner,  # UI uses this to show/hide Edit/Delete buttons
#             "isDownloaded": gen_id in subscribed_ids
#         })
#
#     return results


def fetch_question_banks_metadata(userId: str):
    """
    Fetches a unified list of QBanks for the user's dashboard:
    1. QBanks they created (Owners).
    2. QBanks they subscribed to (Downloads).
    3. EXCLUDES Flashcards explicitly.
    """
    userIdClean = str(userId).strip().lower()

    # --- Step 1: Get all Subscribed Bank IDs ---
    try:
        sub_hits = client.scroll(
            collection_name=COLLECTION_SUBSCRIPTIONS,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="userId", match=models.MatchValue(value=userIdClean))]
            ),
            limit=1000,
            with_payload=True
        )[0]
        subscribed_ids = [h.payload["generatedQAId"] for h in sub_hits if h.payload]
    except Exception:
        subscribed_ids = []

    # --- Step 2: Query MCQ Collection ---
    merged_filter = models.Filter(
        should=[
            models.FieldCondition(key="userId", match=models.MatchValue(value=userIdClean)),
            models.FieldCondition(key="generatedQAId", match=models.MatchAny(any=subscribed_ids))
        ],
        must_not=[
            models.FieldCondition(key="type", match=models.MatchValue(value="FLASHCARD"))
        ]
    )

    banks, _ = client.scroll(
        collection_name=COLLECTION_MCQ,
        scroll_filter=merged_filter,
        limit=500,
        with_payload=True
    )

    results = []
    for bank in banks:
        payload = bank.payload or {}
        gen_id = payload.get("generatedQAId")

        # Determine Permissions
        is_owner = payload.get("userId") == userIdClean

        # Fresh Count from Questions Collection
        count = client.count(
            collection_name=COLLECTION_QUESTIONS,
            count_filter=models.Filter(
                must=[models.FieldCondition(key="generatedQAId", match=models.MatchValue(value=gen_id))]
            )
        ).count

        # Get top subject tags
        tags = compute_subject_tags_for_bank(gen_id)

        results.append({
            "generatedQAId": gen_id,
            "title": payload.get("title", ""),
            "userName": payload.get("userName", ""),
            "description": payload.get("description", ""),
            "createdAt": payload.get("createdAt"),
            "totalQuestions": count,
            "tags": tags,
            "isPublic": payload.get("public", False),
            "canEdit": is_owner,
            "isDownloaded": gen_id in subscribed_ids,

            # --- 🔥 ADD THIS LINE 🔥 ---
            "linkedSourceId": payload.get("linkedSourceId")
        })

    return results

from collections import Counter

def compute_subject_tags_for_bank(generatedQAId, top_k=2):
    subject_counter = Counter()
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_QUESTIONS,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="generatedQAId",
                        match=models.MatchValue(value=generatedQAId)
                    )
                ]
            ),
            limit=200,
            offset=next_offset,
            with_payload=True,
            with_vectors=False
        )

        if not points:
            break

        for point in points:
            payload = point.payload or {}
            pred_sub = payload.get("predicted_subject", {})
            label = pred_sub.get("label")

            if label:
                subject_counter[label] += 1

        if next_offset is None:
            break

    # Take top K subjects by frequency
    return [label for label, _ in subject_counter.most_common(top_k)]







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



def store_submitted_test(userId, testId, testTitle, timeSpent, totalTime, submittedAt,
                         detailed_results, score, total_questions, total_correct,
                         total_mcq=0, total_descriptive=0, mcq_correct=0,
                         subject_analysis=None, concept_analysis=None,
                         grading_status="complete", ai_feedback=None):
    """
    Store submitted test with support for metadata analysis.
    subject_analysis: dict tracking performance per subject
    concept_analysis: dict tracking performance per concept
    """
    try:
        userIdClean = str(userId).strip().lower()
        attemptId = str(uuid.uuid4())

        payload = {
            "attemptId": attemptId,
            "userId": userIdClean,
            "testId": testId,
            "testTitle": testTitle,
            "timeSpent": timeSpent,
            "totalTime": totalTime,
            "submittedAt": submittedAt,
            "score": score,
            "total_questions": total_questions,
            "total_correct": total_correct,
            "total_mcq": total_mcq,
            "total_descriptive": total_descriptive,
            "mcq_correct": mcq_correct,
            "grading_status": grading_status,
            # 🟢 NEW ANALYSIS FIELDS
            "subject_analysis": subject_analysis if subject_analysis else {},
            "concept_analysis": concept_analysis if concept_analysis else {},
            # Store complex lists/dicts as JSON strings for maximum compatibility
            # (Qdrant handles nested dicts, but stringifying ensures easy retrieval in all drivers)
            # "detailed_results": json.dumps(detailed_results),
            # "ai_feedback": json.dumps(ai_feedback) if ai_feedback else json.dumps({})
            "detailed_results": detailed_results if detailed_results else [],
            "ai_feedback": ai_feedback if ai_feedback else {}
        }

        # Generate a dummy or semantic vector for the submission entry
        vec = embed(f"submitted_test:{testId}:{attemptId}")[0]

        p = models.PointStruct(id=attemptId, vector=vec, payload=payload)
        client.upsert(collection_name=COLLECTION_SUBMITTED, points=[p])

        return True, attemptId
    except Exception as e:
        print("store_submitted_test error:", e)
        return False, None



def submitted_tests_by_userId(userId):
    try:
        userIdClean = str(userId).strip().lower()

        filt = models.Filter(
            must=[
                models.FieldCondition(key="userId", match=models.MatchValue(value=userIdClean))
            ]
        )

        dummy_vector = [0.0] * VECTOR_DIM
        hits = client.search(
            collection_name=COLLECTION_SUBMITTED,
            query_vector=dummy_vector,
            query_filter=filt,
            limit=2000,
            with_payload=True
        )

        if not hits:
            return []

        all_attempts = []
        for h in hits:
            payload = _extract_payload(h)
            if not payload:
                continue

            if "detailed_results" in payload and isinstance(payload["detailed_results"], str):
                try:
                    payload["detailed_results"] = json.loads(payload["detailed_results"])
                except Exception:
                    pass

            payload["attemptId"] = payload.get("attemptId") or _extract_id(h)
            payload["testId"] = payload.get("testId")
            all_attempts.append(payload)

        # 🔹 Group attempts by testId, keeping only the latest one per test
        latest_by_test = {}
        for attempt in all_attempts:
            test_id = attempt.get("testId")
            if not test_id:
                continue
            existing = latest_by_test.get(test_id)
            if not existing or attempt.get("submittedAt", "") > existing.get("submittedAt", ""):
                latest_by_test[test_id] = attempt

        # 🔹 Convert dict to sorted list (latest first)
        latest_attempts = sorted(
            latest_by_test.values(),
            key=lambda x: x.get("submittedAt", ""),
            reverse=True
        )

        return latest_attempts

    except Exception as e:
        print("submitted_tests_by_userId error:", e)
        return []
#



def fetch_submitted_test_by_testId(testId):
    try:
        # 🔹 Use a filter since attempts are stored by attemptId, not testId
        filt = models.Filter(
            must=[
                models.FieldCondition(key="testId", match=models.MatchValue(value=testId))
            ]
        )

        dummy_vector = [0.0] * VECTOR_DIM  # Placeholder
        hits = client.search(
            collection_name=COLLECTION_SUBMITTED,
            query_vector=dummy_vector,
            query_filter=filt,
            limit=1000,
            with_payload=True
        )

        if not hits:
            return None

        attempts = []
        for h in hits:
            payload = _extract_payload(h)
            if not payload:
                continue

            # 1. Decode detailed_results (JSON String -> List)
            if "detailed_results" in payload and isinstance(payload["detailed_results"], str):
                try:
                    payload["detailed_results"] = json.loads(payload["detailed_results"])
                except Exception:
                    payload["detailed_results"] = []

            # 2. Decode ai_feedback (JSON String -> Dict)
            if "ai_feedback" in payload and isinstance(payload["ai_feedback"], str):
                try:
                    payload["ai_feedback"] = json.loads(payload["ai_feedback"])
                except Exception:
                    payload["ai_feedback"] = {}

            # 3. Ensure Analysis fields exist (Safety for older records)
            # If Qdrant returns them as objects, these will remain objects.
            payload["subject_analysis"] = payload.get("subject_analysis") or {}
            payload["concept_analysis"] = payload.get("concept_analysis") or {}

            # Ensure IDs are present
            payload["testId"] = payload.get("testId") or testId
            payload["attemptId"] = payload.get("attemptId") or _extract_id(h)

            attempts.append(payload)

        # Sort attempts by submittedAt (latest first)
        attempts.sort(key=lambda x: x.get("submittedAt", ""), reverse=True)

        return attempts

    except Exception as e:
        print("fetch_submitted_test_by_testId error:", e)
        return None



def delete_single_question(questionId):
    try:
        # Use PointIdsList to explicitly tell Qdrant which IDs to remove
        client.delete(
            collection_name=COLLECTION_QUESTIONS,
            points_selector=models.PointIdsList(
                points=[questionId]
            )
        )
        print(f"[INFO] Successfully deleted questionId: {questionId}")
        return True
    except Exception as e:
        print(f"[ERROR] delete_single_question error: {e}")
        return False




def normalize_answer(ans):
    """
    Cleans an answer string for comparison.
    Example: '(A)' -> 'a', 'A.' -> 'a', 'a)' -> 'a'
    """
    if not isinstance(ans, str):
        return str(ans).strip().lower()

    # Remove parentheses, dots, and whitespace, then lowercase
    # This regex removes everything that isn't a letter or a number
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', ans)
    return cleaned.lower()


def update_single_question(questionId, updated_data):
    try:
        # 🔹 Retrieve existing question (Qdrant returns a list)
        existing = client.retrieve(
            collection_name=COLLECTION_QUESTIONS,
            ids=[questionId],
            with_payload=True
        )

        if not existing or len(existing) == 0:
            print(f"[WARN] Question ID {questionId} not found in COLLECTION_QUESTIONS")
            return False

        # Extract payload correctly (take first element)
        existing_payload = _extract_payload(existing[0])

        # 🔹 Preserve key identifiers
        updated_data['questionId'] = questionId
        updated_data['generatedQAId'] = existing_payload.get("generatedQAId")
        updated_data['userId'] = existing_payload.get("userId")

        # 🔹 Normalize options (convert to JSON string)
        if 'options' in updated_data:
            options = updated_data['options']
            if isinstance(options, (dict, list)):
                updated_data['options'] = json.dumps(options)
            elif options is None:
                updated_data['options'] = json.dumps({})
            else:
                # In case frontend sent "(A)": "A. Berlin" etc. inside data
                try:
                    json.loads(options)
                except Exception:
                    updated_data['options'] = json.dumps({
                        k: v for k, v in options.items()
                    }) if isinstance(options, dict) else json.dumps({})

        # 🔹 Merge with existing payload (preserve fields not sent from UI)
        merged_payload = existing_payload.copy()
        merged_payload.update(updated_data)

        # 🔹 Convert payload into Qdrant format
        payload = _to_payload_for_question(merged_payload)

        # 🔹 Update vector based on question text
        question_text = updated_data.get("question", "") or existing_payload.get("question", "")
        vec = embed(question_text)[0]

        # 🔹 Upsert updated point
        p = models.PointStruct(id=questionId, vector=vec, payload=payload)
        client.upsert(collection_name=COLLECTION_QUESTIONS, points=[p], wait=True)

        print(f"[INFO] Question {questionId} updated successfully.")
        return True

    except Exception as e:
        print("update_single_question error:", e)
        return False

def add_single_question(generatedQAId, question_data):
    try:
        bank = client.retrieve(collection_name=COLLECTION_MCQ, ids=[generatedQAId], with_payload=True)
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

# def store_mcqs_for_manual_creation(userId, title, description, mcqs):
#     try:
#         generatedQAId = str(uuid.uuid4())
#         userIdClean = str(userId).strip().lower()
#         createdAt = datetime.now().isoformat()
#         pdf_file = "MANUAL_CREATION"
#         metadata_for_bank = {
#             "userId": userIdClean,
#             "title": title,
#             "generatedQAId": generatedQAId,
#             "description": description,
#             "file_name": pdf_file,
#             "createdAt": createdAt
#
#         }
#         bank_vector = embed(f"{title} {description}")[0]
#         bank_point = models.PointStruct(id=generatedQAId, vector=bank_vector, payload=_to_payload_for_bank(metadata_for_bank))
#         client.upsert(collection_name=COLLECTION_MCQ, points=[bank_point])
#
#         points = []
#         for mcq in mcqs:
#             questionId = str(uuid.uuid4())
#             mcq['generatedQAId'] = generatedQAId
#             mcq['questionId'] = questionId
#             mcq['userId'] = userIdClean
#             options_data = mcq.get("options", "{}")
#             question_text = mcq.get("question", "") or ""
#             q_meta = OrderedDict([
#                 ("questionId", questionId),
#                 ("generatedQAId", generatedQAId),
#                 ("userId", userIdClean),
#                 ("question", mcq.get("question") or ""),
#                 ("noise", mcq.get("noise") or None),
#                 ("image", mcq.get("image") or None),
#                 ("passage", mcq.get("passage") or ""),
#                 ("options", options_data),
#                 ("answer", mcq.get("answer") or ""),
#                 ("documentIndex", mcq.get("documentIndex"))
#             ])
#             payload = _to_payload_for_question(q_meta)
#             vec = embed(question_text)[0]
#             p = models.PointStruct(id=questionId, vector=vec, payload=payload)
#             points.append(p)
#
#         if points:
#             client.upsert(collection_name=COLLECTION_QUESTIONS, points=points)
#         return generatedQAId
#     except Exception as e:
#         print("store_mcqs_for_manual_creation error:", e)
#         return None

#
# def store_mcqs_for_manual_creation(userId, title, description, mcqs, linked_source_id=None):
#     try:
#         generatedQAId = str(uuid.uuid4())
#         userIdClean = str(userId).strip().lower()
#         createdAt = datetime.now().isoformat()
#
#         # Use a distinctive file name for manual entries, or 'SOURCE_LINKED' if a source exists
#         pdf_file = "MANUAL_CREATION" if not linked_source_id else "SOURCE_LINKED"
#
#         metadata_for_bank = {
#             "userId": userIdClean,
#             "title": title,
#             "generatedQAId": generatedQAId,
#             "description": description,
#             "file_name": pdf_file,
#             "createdAt": createdAt,
#             "type": "MANUAL",  # Explicitly mark type
#             "linkedSourceId": linked_source_id  # <--- NEW FIELD ADDED HERE
#         }
#
#         # Create Vector for the Bank itself
#         bank_vector = embed(f"{title} {description}")[0]
#
#         # Upsert Bank Metadata
#         bank_point = models.PointStruct(
#             id=generatedQAId,
#             vector=bank_vector,
#             payload=_to_payload_for_bank(metadata_for_bank)
#         )
#         client.upsert(collection_name=COLLECTION_MCQ, points=[bank_point])
#
#         points = []
#         for mcq in mcqs:
#             questionId = str(uuid.uuid4())
#
#             # Enrich question object with parent IDs
#             mcq['generatedQAId'] = generatedQAId
#             mcq['questionId'] = questionId
#             mcq['userId'] = userIdClean
#
#             options_data = mcq.get("options", "{}")
#             question_text = mcq.get("question", "") or ""
#
#             q_meta = OrderedDict([
#                 ("questionId", questionId),
#                 ("generatedQAId", generatedQAId),
#                 ("userId", userIdClean),
#                 ("question", question_text),
#                 ("noise", mcq.get("noise") or None),
#                 ("image", mcq.get("image") or None),
#                 ("passage", mcq.get("passage") or ""),
#                 ("options", options_data),
#                 ("answer", mcq.get("answer") or ""),
#                 ("documentIndex", mcq.get("documentIndex")),
#
#                 # Optional: You can also tag every individual question with the source ID
#                 ("linkedSourceId", linked_source_id)
#             ])
#
#             payload = _to_payload_for_question(q_meta)
#             vec = embed(question_text)[0]
#             p = models.PointStruct(id=questionId, vector=vec, payload=payload)
#             points.append(p)
#
#         if points:
#             client.upsert(collection_name=COLLECTION_QUESTIONS, points=points)
#
#         return generatedQAId
#
#     except Exception as e:
#         print("store_mcqs_for_manual_creation error:", e)
#         return None





# Import your client, embed, and helper functions (clean_mcq_text, etc.) here

def store_mcqs_for_manual_creation(user_id, user_name, title, description, mcqs, linked_source_id=None):
    """
    Stores manually created MCQs into Qdrant with specific metadata.
    """
    user_id_clean = str(user_id).strip().lower()
    generated_qa_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()

    # --- 1. Bank Metadata (Added linkedSourceId) ---
    metadata_for_bank = {
        "userId": user_id_clean,
        "userName": user_name,
        "title": title,
        "generatedQAId": generated_qa_id,
        "description": description,
        "file_name": "Manual Entry",
        "createdAt": created_at,
        "public": False,
        "linkedSourceId": linked_source_id  # <--- NEW FIELD
    }

    # Create vector for the bank title/desc
    bank_vector = embed(f"{title} {description}")[0]

    question_points = []
    all_have_answers = True
    BATCH_SIZE = 256

    for i, mcq in enumerate(mcqs):
        # Even for manual entry, cleaning ensures consistency
        # Assuming clean_mcq_text and get_image_data are available in your scope
        # If not, you can skip cleaning for manual entry, but keeping structure is key.

        # cleaned_mcq = clean_mcq_text(mcq)
        # image_fields, _ = get_image_data(cleaned_mcq)

        # For manual entry, we might not have 'image_fields', so we default:
        image_fields = {"has_image": False, "image_path": None}

        # Check answers
        raw_answer = mcq.get("answer", "")
        if not raw_answer:
            all_have_answers = False

        question_id = mcq.get("questionId", str(uuid.uuid4()))

        # Extract the injected labels
        pred_sub = mcq.get("predicted_subject", {"label": "Handwritten", "confidence": 1.0})
        pred_con = mcq.get("predicted_concept", {"label": "Handwritten", "confidence": 1.0})

        # --- 2. Construct Qdrant Payload (Matching store_mcqs structure) ---
        q_meta = OrderedDict([
            ("questionId", question_id),
            ("generatedQAId", generated_qa_id),
            ("userId", user_id_clean),
            ("question", mcq.get("question", "")),
            ("noise", ""),  # Manual questions usually have no OCR noise
            ("passage", mcq.get("passage", "")),
            ("options", mcq.get("options", "{}")),  # Ensure it's a stringified JSON
            ("knowledge_base", ""),
            ("question_type", mcq.get("question_type", "DESCRIPTIVE")),
            ("answer", raw_answer),
            ("difficulty", mcq.get("difficulty", "medium")),
            ("documentIndex", i),
            ("linkedSourceId", linked_source_id)  # Optional: Store on question level too
        ])

        # Merge image fields
        q_meta.update(image_fields)

        # Structure the predicted fields exactly like the AI ones
        q_meta["predicted_subject"] = OrderedDict([
            ("label", pred_sub.get("label", "Handwritten")),
            ("confidence", pred_sub.get("confidence", 1.0))
        ])

        q_meta["predicted_concept"] = OrderedDict([
            ("label", pred_con.get("label", "Handwritten")),
            ("confidence", pred_con.get("confidence", 1.0))
        ])

        # Create Vector Point
        q_vec = embed(mcq.get("question", "") or "empty")[0]

        # Assuming _to_payload_for_question handles the OrderedDict conversion
        point = models.PointStruct(
            id=question_id,
            vector=q_vec,
            payload=q_meta  # Directly passing dict/OrderedDict usually works with newer clients, or use your helper
        )
        question_points.append(point)

        # Batch Upsert Questions
        if len(question_points) >= BATCH_SIZE:
            client.upsert(collection_name=COLLECTION_QUESTIONS, points=question_points)
            question_points = []

    # Final Batch
    if question_points:
        client.upsert(collection_name=COLLECTION_QUESTIONS, points=question_points)

    # Store Bank Metadata
    metadata_for_bank["answerFound"] = all_have_answers
    bank_point = models.PointStruct(
        id=generated_qa_id,
        vector=bank_vector,
        payload=metadata_for_bank
    )
    client.upsert(collection_name=COLLECTION_MCQ, points=[bank_point])

    return generated_qa_id



def delete_mcq_bank(generatedQAId):
    try:
        # Retrieve the MCQ bank
        bank = client.retrieve(collection_name=COLLECTION_MCQ, ids=[generatedQAId], with_payload=True)
        if not bank or len(bank) == 0:
            print("bank not found")
            return False

        print("Bank found:", bank)

        # Find all associated questions
        filt = models.Filter(
            must=[models.FieldCondition(key="generatedQAId", match=models.MatchValue(value=generatedQAId))]
        )
        hits = client.scroll(collection_name=COLLECTION_QUESTIONS, limit=1000, scroll_filter=filt)
        hits = _normalize_scroll_result(hits)
        print("Questions found:", hits)

        # Collect IDs to delete
        ids_to_delete = []
        for h in hits:
            if isinstance(h, dict):
                pid = h.get("id") or h.get("point_id") or h.get("pointId")
            else:
                pid = getattr(h, "id", None) or getattr(h, "point_id", None)
            if pid:
                ids_to_delete.append(pid)

        # Delete questions
        if ids_to_delete:
            print("Deleting questions with IDs:", ids_to_delete)
            client.delete(
                collection_name=COLLECTION_QUESTIONS,
                points_selector=models.PointIdsList(points=ids_to_delete)
            )

        # Delete the MCQ bank itself
        print("Deleting MCQ bank with ID:", generatedQAId)
        client.delete(
            collection_name=COLLECTION_MCQ,
            points_selector=models.PointIdsList(points=[generatedQAId])
        )

        return True

    except Exception as e:
        print("delete_mcq_bank error:", e)
        return False

def delete_test_session_by_id(testId):
    try:
        # 1️⃣ Delete from test_sessions_collection
        client.delete(
            collection_name=COLLECTION_TEST_SESSIONS,
            points_selector=models.PointIdsList(points=[testId])
        )

        # 2️⃣ Delete from submitted_tests_collection (if exists)

        filt = models.Filter(
            must=[models.FieldCondition(key="testId", match=models.MatchValue(value=testId))]
        )
        client.delete(
            collection_name=COLLECTION_SUBMITTED,
            points_selector=models.FilterSelector(filter=filt)
        )
        return True

    except Exception as e:
        print("delete_test_session_by_id error:", e)
        return False

def delete_submitted_test_by_id(testId):
    try:
        # 1️⃣ Build filter to match testId
        filt = models.Filter(
            must=[
                models.FieldCondition(
                    key="testId",
                    match=models.MatchValue(value=testId)
                )
            ]
        )
        # 2️⃣ Find all attempts with this testId
        dummy_vector = [0.0] * VECTOR_DIM
        hits = client.search(
            collection_name=COLLECTION_SUBMITTED,
            query_vector=dummy_vector,
            query_filter=filt,
            limit=2000,
            with_payload=False
        )
        if not hits:
            print(f"[INFO] No submitted test attempts found for testId={testId}")
        else:
            # 3️⃣ Collect attempt IDs
            attempt_ids = [_extract_id(h) for h in hits]
            attempt_ids = [x for x in attempt_ids if x]

            if attempt_ids:
                # 4️⃣ Delete points
                client.delete(
                    collection_name=COLLECTION_SUBMITTED,
                    points_selector=attempt_ids
                )
                print(f"[INFO] Deleted {len(attempt_ids)} submitted attempts for testId={testId}")
        return True
    except Exception as e:
        print("delete_submitted_test_by_id error:", e)
        return False

def delete_submitted_test_attempt(attemptId):
        """
        Delete a specific submitted test attempt from Qdrant using attemptId.
        """
        try:
            if not attemptId:
                print("delete_submitted_test_attempt error: attemptId is required")
                return False

            # AttemptId is stored as the point ID in Qdrant
            client.delete(
                collection_name=COLLECTION_SUBMITTED,
                points_selector=models.PointIdsList(points=[attemptId])
            )

            print(f"[Qdrant] Deleted submitted test attempt: {attemptId}")
            return True

        except Exception as e:
            print("delete_submitted_test_attempt error:", e)
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


def update_question_bank_metadata(generatedQAId: str, title: str = None, description: str = None,
                                  is_public: bool = None):
    try:
        results = client.retrieve(
            collection_name=COLLECTION_MCQ,
            ids=[generatedQAId],
            with_payload=True
        )
        if not results:
            return {"success": False}

        payload = _extract_payload(results[0])

        # Update only if provided
        if title is not None: payload["title"] = title
        if description is not None: payload["description"] = description
        if is_public is not None: payload["public"] = is_public  # <--- Save the toggle state

        # Create new vector based on updated text
        vec = embed(f"{payload.get('title', '')} {payload.get('description', '')}")[0]

        client.upsert(
            collection_name=COLLECTION_MCQ,
            points=[models.PointStruct(id=generatedQAId, vector=vec, payload=payload)]
        )
        return {
            "success": True,
            "title_updated": title is not None,
            "description_updated": description is not None,
            "public_updated": is_public is not None
        }
    except Exception as e:
        print(f"[ERROR] update_question_bank_metadata: {e}")
        return {"success": False}



def fetch_question_context(questionId: str):
    """
    Retrieves the question text, passage, and knowledge_base for a specific questionId.
    """
    try:
        # 1. Retrieve the specific point by ID from the questions collection
        results = client.retrieve(
            collection_name=COLLECTION_QUESTIONS,
            ids=[questionId],
            with_payload=True,
            with_vectors=False
        )

        if not results:
            print(f"[WARN] No question found with ID: {questionId}")
            return None

        # 2. Extract the payload using your existing helper
        payload = _extract_payload(results[0])

        # 3. Return only the requested fields
        return {
            "questionId": questionId,
            "question": payload.get("question", ""),
            "passage": payload.get("passage", ""),
            "knowledge_base": payload.get("knowledge_base", "")
        }

    except Exception as e:
        print(f"[ERROR] fetch_question_context error: {e}")
        return None





COLLECTION_SUBSCRIPTIONS = "user_subscriptions"


def subscribe_to_qbank(userId, generatedQAId):
    """Marks a public QBank as 'Downloaded' for a specific user."""
    # Use a deterministic UUID based on User + Bank so they can't subscribe twice
    sub_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{userId}_{generatedQAId}"))

    payload = {
        "userId": str(userId).strip().lower(),
        "generatedQAId": generatedQAId,
        "downloadedAt": datetime.now().isoformat(),
        "is_reference": True  # Metadata to indicate it's a downloaded copy
    }

    # We use a dummy vector for subscriptions as we usually query by userId
    client.upsert(
        collection_name=COLLECTION_SUBSCRIPTIONS,
        points=[models.PointStruct(id=sub_id, vector=[0.0] * VECTOR_DIM, payload=payload)]
    )


def add_subscription_record(userId, generatedQAId):
    """Link a user to a specific question bank."""
    sub_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{userId}_{generatedQAId}"))

    payload = {
        "userId": str(userId).strip().lower(),
        "generatedQAId": generatedQAId,
        "subscribedAt": datetime.now().isoformat(),
    }

    client.upsert(
        collection_name=COLLECTION_SUBSCRIPTIONS,
        points=[models.PointStruct(
            id=sub_id,
            vector=[0.0] * VECTOR_DIM,
            payload=payload
        )]
    )
    return sub_id




def fetch_subscribed_questions(userId):
    userIdClean = str(userId).strip().lower()

    # 1. Find all bank IDs the user has access to
    subs = client.scroll(
        collection_name=COLLECTION_SUBSCRIPTIONS,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="userId", match=models.MatchValue(value=userIdClean))]
        ),
        limit=100
    )[0]

    qbank_ids = [s.payload["generatedQAId"] for s in subs]
    if not qbank_ids:
        return []

    # 2. Fetch questions from the Questions Collection belonging to these banks
    questions, _ = client.scroll(
        collection_name=COLLECTION_QUESTIONS,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="generatedQAId", match=models.MatchAny(any=qbank_ids))]
        ),
        limit=500,
        with_payload=True
    )

    return [q.payload for q in questions]


# vector_db.py

def fetch_public_marketplace():
    """Fetches all curated banks with question counts."""

    ADMIN_ID = "vabtoa3ri7e9juu3cg33vzmw9cs2"
    search_filter = models.Filter(
        must=[
            models.FieldCondition(key="public", match=models.MatchValue(value=True)),
            models.FieldCondition(key="userId", match=models.MatchValue(value=ADMIN_ID))
        ]
    )

    banks, _ = client.scroll(
        collection_name=COLLECTION_MCQ,
        scroll_filter=search_filter,
        limit=100,
        with_payload=True
    )

    results = []
    for b in banks:
        gen_id = b.payload.get("generatedQAId")

        # Get count for each public bank
        count = client.count(
            collection_name=COLLECTION_QUESTIONS,
            count_filter=models.Filter(
                must=[models.FieldCondition(key="generatedQAId", match=models.MatchValue(value=gen_id))]
            )
        ).count

        results.append({
            "generatedQAId": gen_id,
            "title": b.payload.get("title"),
            "description": b.payload.get("description"),
            "totalQuestions": count,  # Added this
            "isPublic": True,
            "canEdit": False
        })
    return results


def toggle_bank_public_status(generatedQAId: str, is_public: bool):
    """Toggle the public flag for a question bank."""
    try:
        results = client.retrieve(
            collection_name=COLLECTION_MCQ,
            ids=[generatedQAId],
            with_payload=True
        )

        if not results:
            return False

        payload = _extract_payload(results[0])
        payload["public"] = is_public

        vec = embed(f"{payload.get('title', '')} {payload.get('description', '')}")[0]
        client.upsert(
            collection_name=COLLECTION_MCQ,
            points=[models.PointStruct(
                id=generatedQAId,
                vector=vec,
                payload=payload
            )]
        )
        return True
    except Exception as e:
        print(f"[ERROR] toggle_bank_public_status: {e}")
        return False


def update_user_metadata_in_qdrant(userId: str, new_username: str):
    print(f"[UPDATE] Starting global name sync for User: {userId} -> {new_username}")

    # We search for BOTH the raw ID and the lowercase ID to be safe
    user_ids_to_check = list(set([str(userId).strip(), str(userId).strip().lower()]))

    sync_filter = models.Filter(
        should=[
            models.FieldCondition(key="userId", match=models.MatchValue(value=uid))
            for uid in user_ids_to_check
        ]
    )

    # 1. Update MCQ Collection (Question Banks)
    # Use a loop to ensure we catch EVERY bank, not just the first 10
    next_offset = None
    banks_updated = 0
    while True:
        banks, next_offset = client.scroll(
            collection_name=COLLECTION_MCQ,
            scroll_filter=sync_filter,
            limit=100,
            offset=next_offset,
            with_payload=True
        )
        for b in banks:
            client.set_payload(
                collection_name=COLLECTION_MCQ,
                payload={"userName": new_username},
                points=[b.id]
            )
            banks_updated += 1
        if not next_offset:
            break

    # 2. Update Questions Collection
    questions_updated = 0
    next_offset = None
    while True:
        q_scroll, next_offset = client.scroll(
            collection_name=COLLECTION_QUESTIONS,
            scroll_filter=sync_filter,
            limit=500,  # Higher limit for questions
            offset=next_offset,
            with_payload=False  # We only need the IDs to update payload
        )

        if not q_scroll:
            break

        # Batch update payloads for performance
        q_ids = [q.id for q in q_scroll]
        client.set_payload(
            collection_name=COLLECTION_QUESTIONS,
            payload={"userName": new_username},
            points=q_ids
        )
        questions_updated += len(q_ids)

        if not next_offset:
            break

    print(f"[SUCCESS] Sync Complete. Updated {banks_updated} banks and {questions_updated} questions.")
    return True


def fetch_community_marketplace(limit=20):
    """
    Fetches public banks created by non-admin users.
    Excludes the hardcoded ADMIN_USER_ID.
    """
    from app import ADMIN_USER_ID  # Importing here to avoid circular imports if necessary

    # Filter: public must be True AND userId must NOT be Admin
    search_filter = models.Filter(
        must=[
            models.FieldCondition(key="public", match=models.MatchValue(value=True))
        ],
        must_not=[
            models.FieldCondition(key="userId", match=models.MatchValue(value=ADMIN_USER_ID))
        ]
    )

    banks, _ = client.scroll(
        collection_name=COLLECTION_MCQ,
        scroll_filter=search_filter,
        limit=limit,
        with_payload=True
    )

    results = []
    for b in banks:
        gen_id = b.payload.get("generatedQAId")

        # Get count for each community bank
        count = client.count(
            collection_name=COLLECTION_QUESTIONS,
            count_filter=models.Filter(
                must=[models.FieldCondition(key="generatedQAId", match=models.MatchValue(value=gen_id))]
            )
        ).count

        results.append({
            "generatedQAId": gen_id,
            "title": b.payload.get("title") or "Untitled Bank",
            "description": b.payload.get("description") or "",
            "totalQuestions": count,
            "userName": b.payload.get("userName") or "Community Member",
            "userId": b.payload.get("userId"),
            "isPublic": True,
            "canEdit": False
        })

    return results




def initialize_bank_record(userId, title="Untitled", description="write a description", record_type="QBANK"):
    try:
        generatedQAId = str(uuid.uuid4())
        userIdClean = str(userId).strip().lower()
        createdAt = datetime.now().isoformat()

        # Determine file tag based on type
        # "MANUAL_FLASHCARD" helps you track origin if you export data later
        pdf_file = "MANUAL_FLASHCARD" if record_type == "FLASHCARD" else "MANUAL_CREATION"

        metadata_for_bank = {
            "userId": userIdClean,
            "title": title,
            "generatedQAId": generatedQAId,
            "description": description,
            "file_name": pdf_file,
            "createdAt": createdAt,
            # These will pass through your _to_payload_for_bank automatically:
            "type": record_type,
            "card_count": 0
        }

        # Embed title/desc for search
        bank_vector = embed(f"{title} {description}")[0]

        bank_point = models.PointStruct(
            id=generatedQAId,
            vector=bank_vector,
            # Your existing function handles the new keys perfectly
            payload=_to_payload_for_bank(metadata_for_bank)
        )

        client.upsert(collection_name=COLLECTION_MCQ, points=[bank_point])
        return generatedQAId

    except Exception as e:
        print("initialize_bank_record error:", e)
        return None


def fetch_user_flashcards(user_id):
    """
    Queries Qdrant for all items where userId matches and type is 'FLASHCARD'.
    Calculates the real-time card count for each deck.
    """
    # 1. Clean the ID
    user_id_clean = str(user_id).strip().lower()

    try:
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="userId",
                    match=models.MatchValue(value=user_id_clean)
                ),
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value="FLASHCARD")
                )
            ]
        )

        results, _ = client.scroll(
            collection_name=COLLECTION_MCQ,
            scroll_filter=search_filter,
            limit=100,
            with_payload=True,
            with_vectors=False
        )

        decks = []
        for point in results:
            p = point.payload

            # --- THE FIX: Real-time Count ---
            # We ask Qdrant: "How many questions have this deck's ID?"
            count_result = client.count(
                collection_name=COLLECTION_QUESTIONS,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="generatedQAId",
                            match=models.MatchValue(value=point.id)
                        )
                    ]
                )
            )
            real_count = count_result.count
            # --------------------------------

            decks.append({
                "id": point.id,
                "title": p.get("title", "Untitled"),
                "subtitle": p.get("description", ""),
                "totalCards": real_count,  # <--- Send the real count!
                "type": p.get("type", "FLASHCARD"),
                "createdAt": p.get("createdAt")
            })

        return decks

    except Exception as e:
        print(f"Error in fetch_user_flashcards: {e}")
        raise e


# vector_db.py

def create_indexes():
    """
    Creates the necessary indices for filtering in Qdrant.
    Run this ONCE to fix the "Index required" error.
    """
    try:
        # 1. Index the 'type' field (Crucial for your error)
        client.create_payload_index(
            collection_name=COLLECTION_MCQ,
            field_name="type",
            field_schema="keyword"  # 'keyword' is used for exact string matching
        )
        print("Index for 'type' created.")

        # 2. Index the 'userId' field (Good practice, speeds up user queries)
        client.create_payload_index(
            collection_name=COLLECTION_MCQ,
            field_name="userId",
            field_schema="keyword"
        )
        print("Index for 'userId' created.")

    except Exception as e:
        print(f"Index creation skipped or failed (might already exist): {e}")



#=======================================================================================================

def store_source_material(userId, title, text_chunks):
    """
    Stores a PDF source.
    :param userId: Owner ID
    :param title: Filename or Title of the PDF
    :param text_chunks: List of dicts, e.g., [{'text': '...', 'page': 1}, ...]
    """
    try:
        userIdClean = str(userId).strip().lower()
        sourceId = str(uuid.uuid4())
        createdAt = datetime.now().isoformat()

        # 1. Store the "File Header" (Metadata)
        # We use the sourceId as the point ID for easy retrieval of the file list
        meta_payload = {
            "userId": userIdClean,
            "sourceId": sourceId,
            "title": title,
            "createdAt": createdAt,
            "type": "FILE_METADATA",  # Distinguishes header from content
            "chunk_count": len(text_chunks)
        }

        # Create a vector for the title (to allow searching for files by name)
        header_vec = embed(title)[0]

        header_point = models.PointStruct(
            id=sourceId,
            vector=header_vec,
            payload=_to_payload_for_source(meta_payload)
        )

        # 2. Process Text Chunks
        points = []
        BATCH_SIZE = 100

        # Extract just the text for embedding
        texts_to_embed = [chunk['text'] for chunk in text_chunks]
        vectors = embed(texts_to_embed)  # Assumes embed() handles list of strings

        for i, chunk in enumerate(text_chunks):
            chunkId = str(uuid.uuid4())

            chunk_payload = {
                "userId": userIdClean,
                "sourceId": sourceId,
                "type": "CONTENT_CHUNK",
                "text": chunk.get('text', ''),
                "page": chunk.get('page', 1),
                "index": i,
                "parent_title": title
            }

            p = models.PointStruct(
                id=chunkId,
                vector=vectors[i],
                payload=_to_payload_for_source(chunk_payload)
            )
            points.append(p)

            # Batch upsert to prevent network timeouts on large files
            if len(points) >= BATCH_SIZE:
                client.upsert(collection_name=COLLECTION_SOURCES, points=points)
                points = []

        # Upsert remaining points
        if points:
            client.upsert(collection_name=COLLECTION_SOURCES, points=points)

        # Upsert the header last
        client.upsert(collection_name=COLLECTION_SOURCES, points=[header_point])

        print(f"[INFO] Stored source '{title}' with {len(text_chunks)} chunks.")
        return sourceId

    except Exception as e:
        print(f"[ERROR] store_source_material: {e}")
        return None


def fetch_user_sources(userId):
    """Returns a list of uploaded PDF files (metadata only)."""
    userIdClean = str(userId).strip().lower()

    # Filter for the "FILE_METADATA" type we created in step 4
    filt = models.Filter(
        must=[
            models.FieldCondition(key="userId", match=models.MatchValue(value=userIdClean)),
            models.FieldCondition(key="type", match=models.MatchValue(value="FILE_METADATA"))
        ]
    )

    results, _ = client.scroll(
        collection_name=COLLECTION_SOURCES,
        scroll_filter=filt,
        limit=100,
        with_payload=True,
        with_vectors=False
    )

    sources = []
    for r in results:
        p = r.payload
        sources.append({
            "sourceId": p.get("sourceId"),
            "title": p.get("title"),
            "createdAt": p.get("createdAt"),
            "chunk_count": p.get("chunk_count", 0)
        })
    return sources


def delete_source_material(sourceId):
    """Deletes a file and ALL its text chunks."""
    try:
        # Filter deletes all points (Metadata + Chunks) that share this sourceId
        client.delete(
            collection_name=COLLECTION_SOURCES,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="sourceId", match=models.MatchValue(value=sourceId))]
                )
            )
        )
        print(f"[INFO] Deleted source {sourceId}")
        return True
    except Exception as e:
        print(f"[ERROR] delete_source_material: {e}")
        return False





def fetch_full_source_text(sourceId):
    """
    Retrieves the full text of a source by stitching together all its chunks.
    """
    try:
        # 1. Fetch all chunks for this source ID
        # We set a high limit to get the whole document
        filt = models.Filter(
            must=[
                models.FieldCondition(key="sourceId", match=models.MatchValue(value=sourceId)),
                models.FieldCondition(key="type", match=models.MatchValue(value="CONTENT_CHUNK"))
            ]
        )

        # Scroll through all chunks (in case it's a massive book)
        all_chunks = []
        next_offset = None

        while True:
            results, next_offset = client.scroll(
                collection_name=COLLECTION_SOURCES,
                scroll_filter=filt,
                limit=500,  # Batch size
                offset=next_offset,
                with_payload=True,
                with_vectors=False
            )

            for point in results:
                all_chunks.append(point.payload)

            if next_offset is None:
                break

        # 2. Sort chunks to ensure correct reading order
        # We sort by 'page' first, then by 'index' (if you used chunking within pages)
        sorted_chunks = sorted(all_chunks, key=lambda x: (int(x.get('page', 0)), int(x.get('index', 0))))

        # 3. Concatenate text
        full_text = "\n\n".join([chunk.get('text', '') for chunk in sorted_chunks])

        return full_text

    except Exception as e:
        print(f"[ERROR] fetch_full_source_text: {e}")
        return None


def update_question_result_in_db(attemptId, questionId, ai_score, ai_feedback, status):
    """
    Updates a single question's grading status inside the 'detailed_results' list
    of a submitted test in Qdrant.
    """
    try:
        # 1. Retrieve the existing submission payload
        points = client.retrieve(
            collection_name=COLLECTION_SUBMITTED,
            ids=[attemptId],
            with_payload=True
        )

        if not points:
            print(f"[ERROR] update_question_result: Attempt {attemptId} not found.")
            return False

        payload = _extract_payload(points[0])

        # 2. Parse detailed_results (handle both List and JSON String formats)
        detailed_results = payload.get("detailed_results", [])
        if isinstance(detailed_results, str):
            try:
                detailed_results = json.loads(detailed_results)
            except Exception:
                detailed_results = []

        # 3. Find and update the specific question
        updated = False
        for item in detailed_results:
            # Comparison as strings to be safe against UUID vs String types
            if str(item.get("questionId")) == str(questionId):
                item["ai_score"] = ai_score
                item["ai_feedback"] = ai_feedback
                item["status"] = status

                # Logic for UI correctness flag (assuming score out of 10)
                # > 5.0 is considered "passing" partial credit
                item["is_correct"] = float(ai_score) >= 5.0
                updated = True
                break

        if not updated:
            print(f"[WARN] Question {questionId} not found in attempt {attemptId}")
            return False

        # 4. Write the modified list back to Qdrant
        # We ensure detailed_results is stored as a list (Qdrant handles parsing)
        client.set_payload(
            collection_name=COLLECTION_SUBMITTED,
            payload={
                "detailed_results": detailed_results
            },
            points=[attemptId]
        )
        return True

    except Exception as e:
        print(f"[ERROR] update_question_result_in_db: {e}")
        return False


def finalize_submission_status(attemptId):
    """
    Recalculates the final score after background grading is complete
    and updates the status to 'complete'.
    """
    try:
        # 1. Fetch current data
        points = client.retrieve(
            collection_name=COLLECTION_SUBMITTED,
            ids=[attemptId],
            with_payload=True
        )

        if not points:
            return False

        payload = _extract_payload(points[0])

        # 2. Parse results
        detailed_results = payload.get("detailed_results", [])
        if isinstance(detailed_results, str):
            detailed_results = json.loads(detailed_results)

        total_questions = int(payload.get("total_questions") or 1)
        if total_questions == 0: total_questions = 1

        # 3. Recalculate Scores
        # We need to re-sum MCQ (boolean) + Descriptive (partial credit 0-1)
        mcq_correct_count = 0
        total_descriptive_points = 0.0

        for res in detailed_results:
            q_type = res.get("question_type", "MCQ").upper()

            if q_type == "MCQ":
                # Check explicit boolean flag from initial submission
                if res.get("is_correct") is True:
                    mcq_correct_count += 1
            else:
                # Descriptive: Normalize AI score (0-10) to partial credit (0-1)
                # Falls back to 0 if not yet graded
                raw_score = float(res.get("ai_score") or 0)
                partial_credit = raw_score / 10.0
                total_descriptive_points += partial_credit

        # Aggregate total correct (integer MCQs + float partials)
        total_correct_aggregate = mcq_correct_count + total_descriptive_points

        # Final Percentage Calculation
        final_score_percent = round((total_correct_aggregate / total_questions) * 100, 2)

        # 4. Update Qdrant
        client.set_payload(
            collection_name=COLLECTION_SUBMITTED,
            payload={
                "score": final_score_percent,
                "total_correct": total_correct_aggregate,
                "grading_status": "complete"
            },
            points=[attemptId]
        )
        print(f"[INFO] Finalized grading for {attemptId}. Score: {final_score_percent}%")
        return True

    except Exception as e:
        print(f"[ERROR] finalize_submission_status: {e}")
        return False





#===========================================================================================================
#Firebase Notification service
#===========================================================================================================



# Add this new function:
def store_user_fcm_token(user_id, token):
    try:
        user_id_clean = str(user_id).strip().lower()
        # We use userId as the point ID (hashing it to UUID if necessary)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, user_id_clean))

        # Check if user exists to preserve other data, or just upsert
        payload = {
            "userId": user_id_clean,
            "fcmToken": token,
            "lastSeen": datetime.now().isoformat()
        }

        # Dummy vector
        vec = [0.0] * VECTOR_DIM

        client.upsert(
            collection_name=COLLECTION_USERS,
            points=[models.PointStruct(id=point_id, vector=vec, payload=payload)]
        )
        return True
    except Exception as e:
        print(f"Error storing FCM token: {e}")
        return False


def get_user_fcm_token(user_id):
    try:
        user_id_clean = str(user_id).strip().lower()
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, user_id_clean))

        points = client.retrieve(
            collection_name=COLLECTION_USERS,
            ids=[point_id],
            with_payload=True
        )

        if points:
            return points[0].payload.get("fcmToken")
        return None
    except Exception as e:
        print(f"Error fetching FCM token: {e}")
        return None
