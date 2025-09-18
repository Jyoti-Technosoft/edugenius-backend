import os
import uuid
import json
from base64 import b64encode
import random
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from collections import OrderedDict

client = chromadb.PersistentClient(
    path="./chromadb_data",
    settings=Settings()
)

collection_name = "mcq_collection"
if collection_name in [c.name for c in client.list_collections()]:
    collection = client.get_collection(name=collection_name)
else:
    collection = client.create_collection(name=collection_name)

# New collection for storing generated tests
test_sessions_collection_name = "test_sessions_collection"
if test_sessions_collection_name in [c.name for c in client.list_collections()]:
    test_sessions_collection = client.get_collection(name=test_sessions_collection_name)
else:
    test_sessions_collection = client.create_collection(name=test_sessions_collection_name)

# # New collection for storing submitted_tests ( Future Scope for analysis and history keeping)
# submitted_tests_collection_name = "submitted_tests_collection"
# if submitted_tests_collection_name in [c.name for c in client.list_collections()]:
#     submitted_tests_collection = client.get_collection(name=submitted_tests_collection_name)
# else:
#     submitted_tests_collection = client.create_collection(name=submitted_tests_collection_name)

# -----------------------------
# Embedding model
# -----------------------------
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def ordered_mcq(mcq):
    return OrderedDict([
        ("question", mcq.get("question") or ""),
        ("noise", mcq.get("noise") or ""),
        ("image", mcq.get("image") or None),
        ("options", mcq.get("options") or {}),
        ("answer", mcq.get("answer") or "")
    ])
def store_mcqs(userId, title, description, mcqs, pdf_file, createdAt):
    generatedQAId = str(uuid.uuid4())
    userId = str(userId).strip()  # ensure clean

    mcqs_ordered = [ordered_mcq(mcq) for mcq in mcqs]
    mcqs_json = json.dumps(mcqs_ordered)


    text_for_embedding = f"{title} {description}"
    embeddings = embedding_model.encode([text_for_embedding]).tolist()
    userIdClean = str(userId).strip().lower()
    metadata = {
        "userId": userIdClean,
        "title": title,
        "generatedQAId": generatedQAId,
        "description": description,
        "file_name": pdf_file,
        "mcqs": mcqs_json,
        "createdAt": createdAt

    }


    collection.add(
        ids=[generatedQAId],
        documents=[userIdClean],
        embeddings=embeddings,
        metadatas=[metadata]
    )

    return generatedQAId




def fetch_mcqs(userId: str = None, generatedQAId: str = None):
    """
    Fetch MCQs and return the full record, in a consistent format.
    """
    if not userId and not generatedQAId:
        print("[DEBUG] Either userId or generatedQAId must be provided.")
        return []

    results = None
    if generatedQAId:
        results = collection.get(ids=[generatedQAId], include=["documents", "metadatas"])
    elif userId:
        userIdClean = str(userId).strip().lower()
        results = collection.get(where={"userId": userIdClean}, include=["documents", "metadatas"])

    if not results or not results.get("ids"):
        print(f"[DEBUG] No MCQs found for the provided criteria.")
        return []

    data = []
    for i in range(len(results["ids"])):
        meta = results["metadatas"][i].copy()

        mcq_list = []
        if "mcqs" in meta:
            if isinstance(meta["mcqs"], str):
                try:
                    mcq_list = json.loads(meta["mcqs"])
                except json.JSONDecodeError:
                    mcq_list = []
            else:
                mcq_list = meta["mcqs"]

        mcq_list = [ordered_mcq(mcq) for mcq in mcq_list]

     # Append the full record dictionary to the data list
        data.append({
            "id": results["ids"][i],  # Add the 'id' key
            "document": results["documents"][i],  # Add the 'document' key
            "metadata": {  # Nest the metadata dictionary
                "userId": meta.get("userId"),
                "title": meta.get("title"),
                "description": meta.get("description"),
                "file_name": meta.get("file_name"),
                "generatedQAId": meta.get("generatedQAId"),
                "mcqs": mcq_list,
                "createdAt": meta.get("createdAt")
            }
        })

    return data



def fetch_random_mcqs(generatedQAId: str, num_questions: int = None):
    """
    Fetches a random sample of MCQs for a given generatedQAId and formats the output
    to be identical to the fetch_mcqs function.
    """
    # 1. Call the existing fetch_mcqs function to get the data
    results = fetch_mcqs(generatedQAId=generatedQAId)

    if not results:
        # If no results, return an empty list to match fetch_mcqs's return type.
        return []

    # 2. Extract the full record and the list of MCQs
    record = results[0]
    original_mcqs = record.get("metadata", {}).get("mcqs", [])

    if not original_mcqs:
        return []

    # 3. Handle random sampling if num_questions is specified
    selected_mcqs = original_mcqs
    if num_questions and num_questions < len(original_mcqs):
        selected_mcqs = random.sample(original_mcqs, num_questions)

    # 4. Shuffle the options within each selected MCQ
    formatted_mcqs = []
    for mcq in selected_mcqs:
        # Create a deep copy to avoid modifying the original list
        shuffled_mcq = mcq.copy()
        options = shuffled_mcq.get("options", {})

        # Shuffle the options and re-create the dictionary to maintain order
        option_items = list(options.items())
        random.shuffle(option_items)
        shuffled_mcq["options"] = OrderedDict(option_items)

        formatted_mcqs.append(shuffled_mcq)

    # 5. Re-assemble the data in the exact format of fetch_mcqs
    record["metadata"]["mcqs"] = formatted_mcqs

    return [record]








def store_test_session(userId, testId, testTitle,testTimeLimit, createdAt, mcqs_data):
    """
    Stores a specific test session with its questions in the dedicated collection.
    Returns True on success, False on failure.
    """
    try:
        mcqs_json = json.dumps(mcqs_data)
        metadata = {
            "userId": userId,
            "testTitle": testTitle,
            "testTimeLimit": testTimeLimit,
            "createdAt": createdAt,
            "questions": mcqs_json  # Store the questions as a JSON string

        }
        test_sessions_collection.add(
            ids=[testId],
            documents=[f"Test session for user {userId}"],
            metadatas=[metadata]
        )
        print(f"[DEBUG] Stored test session with ID: {testId}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to store test session: {e}")
        return False



def fetch_test_session_by_testId(testId):
    """
    Fetches a specific test session by its ID and returns the complete data.
    """
    try:
        results = test_sessions_collection.get(ids=[testId], include=["metadatas"])
        if not results or not results["ids"]:
            return None

        metadata = dict(results["metadatas"][0])

        # Parse the JSON string back into a list of questions
        if "questions" in metadata:
            try:
                metadata["questions"] = json.loads(metadata["questions"])
            except json.JSONDecodeError:
                metadata["questions"] = []

        # Return the full record
        return metadata
    except Exception as e:
        print(f"[ERROR] Failed to fetch test session: {e}")
        return None


def test_sessions_by_userId(userId):
    """
    Fetches all test sessions for a given userId with all their details (questions, answers, etc.).
    """
    try:
        results = test_sessions_collection.get(where={"userId": userId}, include=["metadatas"])
        if not results or not results["ids"]:
            return []

        test_sessions = []
        for i in range(len(results["ids"])):
            session_data = dict(results["metadatas"][i])
            session_data["testId"] = results["ids"][i]

            # Parse the questions JSON string back into a list of questions
            if "questions" in session_data:
                try:
                    session_data["questions"] = json.loads(session_data["questions"])
                except json.JSONDecodeError:
                    session_data["questions"] = []

            test_sessions.append(session_data)

        return test_sessions
    except Exception as e:
        print(f"[ERROR] Failed to fetch test sessions for user: {e}")
        return None