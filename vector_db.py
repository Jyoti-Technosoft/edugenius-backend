
import uuid
import json
from base64 import b64encode
import random
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from collections import OrderedDict
from datetime import datetime

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
submitted_tests_collection_name = "submitted_tests_collection"
if submitted_tests_collection_name in [c.name for c in client.list_collections()]:
    submitted_tests_collection = client.get_collection(name=submitted_tests_collection_name)
else:
    submitted_tests_collection = client.create_collection(name=submitted_tests_collection_name)


# Create a new collection for storing individual questions
questions_collection_name = "questions_collection"
if questions_collection_name in [c.name for c in client.list_collections()]:
    questions_collection = client.get_collection(name=questions_collection_name)
else:
    questions_collection = client.create_collection(name=questions_collection_name)



# -----------------------------
# Embedding model
# -----------------------------
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")



def ordered_mcq(mcq):
    return OrderedDict([
        ("questionId", str(uuid.uuid4())),
        ("question", mcq.get("question") or ""),
        ("noise", mcq.get("noise") or ""),
        ("image", mcq.get("image") or None),
        ("options", mcq.get("options") or {}),
        ("answer", mcq.get("answer") or "")
    ])




def store_mcqs(userId, title, description, mcqs, pdf_file, createdAt):
    generatedQAId = str(uuid.uuid4())
    userId = str(userId).strip()
    userIdClean = str(userId).strip().lower()

    # Step 1: Store the question bank's metadata in the original collection
    metadata_for_bank = {
        "userId": userIdClean,
        "title": title,
        "generatedQAId": generatedQAId,
        "description": description,
        "file_name": pdf_file,
        "createdAt": createdAt
    }
    # Using a simple string for embedding, as per your original code
    text_for_embedding = f"{title} {description}"
    embeddings = embedding_model.encode([text_for_embedding]).tolist()

    collection.add(
        ids=[generatedQAId],
        documents=[userIdClean],
        embeddings=embeddings,
        metadatas=[metadata_for_bank]
    )

    # Step 2: Store each individual question in the new 'questions_collection'
    question_ids = []
    question_metadatas = []
    question_documents = []


    for i, mcq in enumerate(mcqs):
        questionId = str(uuid.uuid4())
        question_ids.append(questionId)

        # Add a reference to the question bank
        mcq['generatedQAId'] = generatedQAId
        mcq['questionId'] = questionId
        mcq['userId'] = userIdClean
        # CRITICAL FIX: Store the sequential index
        mcq['documentIndex'] = i

        # Convert the 'options' dictionary to a JSON string
        options_json = json.dumps(mcq.get("options", {}))

        # Append the question text to the documents list
        question_documents.append(mcq.get("question") or "")

        # Create an ordered dictionary with the core data
        mcq_ordered = OrderedDict([
            ("questionId", mcq.get("questionId")),
            ("generatedQAId", mcq.get("generatedQAId")),
            ("userId", mcq.get("userId")),
            ("question", mcq.get("question") or ""),
            ("noise", mcq.get("noise") or ""),
            ("image", mcq.get("image") or None),
            ("options", options_json),
            ("answer", mcq.get("answer") or ""),
            ("documentIndex", mcq.get("documentIndex"))  # CRITICAL FIX: Include the index
        ])

        # Filter out keys with None values before adding to metadata
        filtered_metadata = {k: v for k, v in mcq_ordered.items() if v is not None}
        question_metadatas.append(filtered_metadata)


    questions_collection.add(
        ids=question_ids,
        documents=question_documents,
        metadatas=question_metadatas
    )

    return generatedQAId


def fetch_mcqs(userId: str = None, generatedQAId: str = None):
    """
    Fetch MCQs and return the full record from the new collections,
    with fields consistently ordered for readability.
    """
    if not userId and not generatedQAId:
        print("[DEBUG] Either userId or generatedQAId must be provided.")
        return []

    results = None
    if generatedQAId:
        results = collection.get(ids=[generatedQAId], include=["documents", "metadatas"])

    elif userId:
        # Keep this lowercase, as it's stored in metadata as lowercase
        userIdClean = str(userId).strip().lower()
        results = collection.get(where={"userId": userIdClean}, include=["documents", "metadatas"])

    if not results or not results.get("ids"):
        return []

    data = []
    for i in range(len(results["ids"])):
        meta = results["metadatas"][i].copy()
        generatedQAId_from_meta = meta.get("generatedQAId")

        if generatedQAId_from_meta:
            # Note: Fetching questions from questions_collection.
            questions_results = questions_collection.get(where={"generatedQAId": generatedQAId_from_meta},
                                                         include=["metadatas"])

            mcq_list = []
            if questions_results.get("metadatas"):
                for q_metadata in questions_results["metadatas"]:

                    #  Revert to json.loads() because store_mcqs used json.dumps()
                    if "options" in q_metadata and isinstance(q_metadata["options"], str):
                        try:
                            q_metadata["options"] = json.loads(q_metadata["options"])
                        except json.JSONDecodeError as e:
                            # Log the error, but don't crash the API
                            print(
                                f"[ERROR] JSON decode failed for options on question {q_metadata.get('questionId')}: {e}")
                            q_metadata["options"] = {}  # Default to an empty dictionary or list

                    # Ensure the data is ordered for a consistent output format
                    ordered_mcq = OrderedDict([
                        ("questionId", q_metadata.get("questionId")),
                        ("generatedQAId", q_metadata.get("generatedQAId")),
                        ("userId", q_metadata.get("userId")),
                        ("question", q_metadata.get("question")),
                        ("options", q_metadata.get("options")),
                        ("answer", q_metadata.get("answer")),
                        ("noise", q_metadata.get("noise")),
                        ("image", q_metadata.get("image")),
                        ("documentIndex", q_metadata.get("documentIndex"))
                    ])
                    mcq_list.append(ordered_mcq)

            # Sort by the explicit documentIndex
            mcq_list = sorted(
                mcq_list,
                key=lambda x: int(x['documentIndex']) if x.get('documentIndex') is not None else float('inf')
            )
        else:
            mcq_list = []

        meta["mcqs"] = mcq_list
        data.append({
            "id": results["ids"][i],
            "document": results["documents"][i],
            "metadata": meta
        })
    return data



def fetch_random_mcqs(generatedQAId: str, num_questions: int = None):
    """
    Fetches a random sample of MCQs for a given generatedQAId from the new collections.
    """
    # 1. Call fetch_mcqs. It now returns questions sorted by documentIndex.
    results = fetch_mcqs(generatedQAId=generatedQAId)

    if not results:
        return []

    # 2. Access the list of MCQs from the 'metadata' key
    record = results[0]
    original_mcqs = record.get("metadata", {}).get("mcqs", [])

    if not original_mcqs:
        return []

    # 3. Handle random sampling if num_questions is specified
    selected_mcqs = original_mcqs
    if num_questions and num_questions < len(original_mcqs):
        # NOTE: If we want the final test to be a random *selection* but *sorted* by the original order,
        # we'd select here, then re-sort by index. But for a random *test*, we usually want random order.
        # Since this function is for generating a test, we will keep the random sample,
        # and then shuffle the options.
        selected_mcqs = random.sample(original_mcqs, num_questions)

    # 4. Shuffle the options within each selected MCQ
    formatted_mcqs = []
    for mcq in selected_mcqs:
        shuffled_mcq = mcq.copy()

        # Options should already be a dictionary because of the logic in fetch_mcqs,
        # but we check and parse just in case.
        options = shuffled_mcq.get("options", {})
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except json.JSONDecodeError:
                options = {}

        # Shuffle the options and re-create the dictionary to maintain order
        option_items = list(options.items())
        random.shuffle(option_items)
        shuffled_mcq["options"] = OrderedDict(option_items)

        formatted_mcqs.append(shuffled_mcq)

    # 5. Re-assemble the data in the exact format of fetch_mcqs
    record["metadata"]["mcqs"] = formatted_mcqs

    return [record]





def store_test_session(userId, testId, testTitle,totalTime, createdAt, mcqs_data):
    """
    Stores a specific test session with its questions in the dedicated collection.
    Returns True on success, False on failure.
    """
    try:
        mcqs_json = json.dumps(mcqs_data)
        metadata = {
            "userId": userId,
            "testTitle": testTitle,
            "totalTime": totalTime,
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



def fetch_test_by_testId(testId):
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
        print(f"[ERROR] Failed to fetch test: {e}")
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


def store_submitted_test(userId, testId, testTitle, timeSpent, totalTime, submittedAt,  detailed_results,
                         score, total_questions):
    """
    Stores the results of a submitted test.
    Returns True on success, False on failure.
    """
    try:
        results_json = json.dumps(detailed_results)
        metadata = {
            "userId": userId,
            "testId": testId,
            "testTitle": testTitle,
            "timeSpent": timeSpent,
            "totalTime": totalTime,
            "submittedAt": submittedAt,
            "score": score,
            "total_questions": total_questions,
            "detailed_results": results_json  # Store the detailed results as a JSON string
        }
        submitted_tests_collection.add(
            ids=[testId],
            documents=[f"Submitted test for user {userId}"],
            metadatas=[metadata]
        )
        print(f"[DEBUG] Stored submitted test with ID: {testId}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to store submitted test: {e}")
        return False


def submitted_tests_by_userId(userId):
    """
    Fetches all submitted test sessions for a given userId.
    """
    try:
        results = submitted_tests_collection.get(where={"userId": userId}, include=["metadatas"])
        if not results or not results["ids"]:
            return []

        submitted_sessions = []
        for i in range(len(results["ids"])):
            session_data = dict(results["metadatas"][i])
            # The testId is the document ID in this case
            session_data["testId"] = results["ids"][i]

            # Parse the detailed_results JSON string back into a list
            if "detailed_results" in session_data:
                try:
                    session_data["detailed_results"] = json.loads(session_data["detailed_results"])
                except json.JSONDecodeError:
                    session_data["detailed_results"] = []

            submitted_sessions.append(session_data)

        return submitted_sessions
    except Exception as e:
        print(f"[ERROR] Failed to fetch submitted test sessions for user: {e}")
        return None




def update_question_bank(generatedQAId, mcqs_data):
    """
    Updates the list of MCQs for an existing question bank.

    Args:
        generatedQAId (str): The ID of the question bank to update.
        mcqs_data (list): The new, complete list of MCQs.

    Returns:
        bool: True on success, False on failure.
    """
    try:
        # Fetch the existing metadata to avoid overwriting other fields
        existing_record = collection.get(ids=[generatedQAId], include=["metadatas"])
        if not existing_record or not existing_record["metadatas"]:
            print(f"[ERROR] Question bank with ID '{generatedQAId}' not found for update.")
            return False

        current_metadata = existing_record["metadatas"][0]

        # Serialize the updated questions and replace the old 'mcqs' key
        current_metadata["mcqs"] = json.dumps(mcqs_data)

        # Use upsert to update the record in the collection
        collection.upsert(
            ids=[generatedQAId],
            metadatas=[current_metadata]
        )

        print(f"[DEBUG] Successfully updated question bank with ID: {generatedQAId}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to update question bank: {e}")
        return False




def delete_single_question(questionId):
    """
    Deletes a single question by its unique ID.
    Returns True on success, False on failure.
    """
    try:
        # ChromaDB's delete operation
        questions_collection.delete(ids=[questionId])
        print(f"[DEBUG] Successfully deleted question with ID: {questionId}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to delete single question: {e}")
        return False





def update_single_question(questionId, updated_data):
    """
    Updates a single question by its unique ID.
    Returns True on success, False on failure.
    """
    try:
        # Fetch the existing record to get its generatedQAId and userId
        existing_record = questions_collection.get(ids=[questionId], include=["metadatas"])
        if not existing_record or not existing_record.get("ids"):
            return False

        existing_meta = existing_record["metadatas"][0]

        # Preserve the existing metadata fields
        updated_data['questionId'] = questionId
        updated_data['generatedQAId'] = existing_meta.get("generatedQAId")
        updated_data['userId'] = existing_meta.get("userId")

        #  Convert the Python dict for 'options' to a JSON string
        if 'options' in updated_data and isinstance(updated_data['options'], dict):
            updated_data['options'] = json.dumps(updated_data['options'])
        # Handle the case where the key might be missing or is None
        elif 'options' in updated_data and updated_data['options'] is None:
            updated_data['options'] = json.dumps({})

        # The ChromaDB upsert call requires a documents parameter, even if empty
        questions_collection.upsert(
            ids=[questionId],
            metadatas=[updated_data],
            documents=[""] # Added in an earlier step to satisfy ChromaDB
        )
        print(f"[DEBUG] Successfully updated question with ID: {questionId}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to update single question: {e}")
        return False


def add_single_question(generatedQAId, question_data):
    """
    Stores a single question and links it to a generatedQAId.
    Returns True on success, False on failure.
    """
    try:
        # Check if the question bank exists
        existing_qb_data = collection.get(ids=[generatedQAId], include=["metadatas"])
        if not existing_qb_data or not existing_qb_data.get("ids"):
            print(f"[ERROR] Question bank with ID '{generatedQAId}' not found.")
            return False

        questionId = str(uuid.uuid4())

        # 1. Inject mandatory IDs
        question_data['questionId'] = questionId
        question_data['generatedQAId'] = generatedQAId
        question_data['userId'] = existing_qb_data.get('metadatas', [{}])[0].get('userId', 'unknown')

        # 2. Extract the question text for the 'documents' field
        question_text = question_data.get('question', '')

        # 3. Convert options to JSON string for metadata compatibility
        if 'options' in question_data and isinstance(question_data['options'], (dict, list)):
            question_data['options'] = json.dumps(question_data['options'])

        # Filter out keys with None values before adding to metadata
        filtered_metadata = {k: v for k, v in question_data.items() if v is not None}

        #  Add the 'documents' argument
        questions_collection.add(
            ids=[questionId],
            documents=[question_text],  # <-- PROVIDED THE QUESTION TEXT HERE
            metadatas=[filtered_metadata]
        )
        print(f"[DEBUG] Added new question with ID: {questionId} to question bank: {generatedQAId}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to add single question: {e}")
        return False





def store_mcqs_for_manual_creation(userId, title, description, mcqs):
    """
    Stores a manually created question bank.
    This logic is based on the original store_mcqs but handles internal IDs and metadata.
    """
    generatedQAId = str(uuid.uuid4())
    userId = str(userId).strip().lower()
    createdAt = datetime.now().isoformat()
    pdf_file = "MANUAL_CREATION"  # Fixed identifier

    # Step 1: Store the question bank's metadata in the original collection
    metadata_for_bank = {
        "userId": userId,
        "title": title,
        "generatedQAId": generatedQAId,
        "description": description,
        "file_name": pdf_file,
        "createdAt": createdAt
    }
    text_for_embedding = f"{title} {description} manual question bank"
    embeddings = embedding_model.encode([text_for_embedding]).tolist()

    collection.add(
        ids=[generatedQAId],
        documents=[title],
        embeddings=embeddings,
        metadatas=[metadata_for_bank]
    )

    # Step 2: Store each individual question in the 'questions_collection'
    question_ids = []
    question_metadatas = []
    question_documents = []

    for mcq in mcqs:
        questionId = str(uuid.uuid4())
        question_ids.append(questionId)

        # Inject/Confirm essential metadata
        mcq['generatedQAId'] = generatedQAId
        mcq['questionId'] = questionId
        mcq['userId'] = userId

        # The 'options' should already be a JSON string here due to API preprocessing.
        options_data = mcq.get("options", "{}")

        # Append the question text to the documents list
        question_documents.append(mcq.get("question") or "")

        # Create an ordered dictionary with the core data
        mcq_ordered = OrderedDict([
            ("questionId", mcq.get("questionId")),
            ("generatedQAId", mcq.get("generatedQAId")),
            ("userId", mcq.get("userId")),
            ("question", mcq.get("question") or ""),
            ("noise", mcq.get("noise") or None),
            ("image", mcq.get("image") or None),
            ("options", options_data),  # Use the pre-processed string
            ("answer", mcq.get("answer") or ""),
            ("documentIndex", mcq.get("documentIndex"))
        ])

        filtered_metadata = {k: v for k, v in mcq_ordered.items() if v is not None}
        question_metadatas.append(filtered_metadata)

    questions_collection.add(
        ids=question_ids,
        documents=question_documents,
        metadatas=question_metadatas
    )

    return generatedQAId


import uuid





def delete_mcq_bank(generatedQAId):
    """
    Deletes an entire question bank and all its associated questions from ChromaDB.

    Args:
        generatedQAId (str): The unique ID of the question bank to delete.

    Returns:
        bool: True if deletion was attempted for all components, False if the
              main question bank was not initially found.
    """
    try:
        # 1. Check if the main question bank exists in the metadata collection.
        # This acts as a check before proceeding to delete thousands of questions.
        bank_exists = collection.get(ids=[generatedQAId], include=[])
        if not bank_exists or not bank_exists.get("ids"):
            print(f"[WARN] Question bank metadata ID '{generatedQAId}' not found in mcq_collection.")
            return False

        # --- Delete Associated Questions (questions_collection) ---

        # NOTE: ChromaDB's delete operation requires a filter when deleting by metadata.
        # We delete all documents where the 'generatedQAId' metadata matches the ID.
        questions_collection.delete(
            where={"generatedQAId": generatedQAId}
        )
        print(f"[DEBUG] Deleted associated questions for ID: {generatedQAId} from questions_collection.")

        # --- Delete Main Question Bank Metadata (mcq_collection) ---

        # We delete the single document using its ID (which is the generatedQAId).
        collection.delete(
            ids=[generatedQAId]
        )
        print(f"[DEBUG] Deleted main metadata entry for ID: {generatedQAId} from mcq_collection.")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to delete question bank '{generatedQAId}': {e}")
        # Return False to trigger a 404 or 500 in the Flask route
        return False


def delete_test_session_by_id(testId):
    """
    Deletes a specific test session from the test_sessions_collection using its ID.

    Args:
        testId (str): The unique ID of the test session to delete.

    Returns:
        bool: True if the operation completed (whether an item was deleted or not),
              False on a critical database error.
    """

        # Use the delete method, specifying the ID
    test_sessions_collection.delete(ids=[testId])
    return True




def delete_submitted_test_by_id(testId):
    """
    Deletes a specific submitted test result from the submitted_tests_collection.
    """
    try:
        result = submitted_tests_collection.delete(
            ids=[testId]
        )

        if result and result.get('ids'):
            print(f"[DEBUG] Deleted submitted test result with ID: {testId}.")
            return True
        else:
            print(f"[WARN] Submitted test ID '{testId}' not found for deletion.")
            return False

    except Exception as e:
        print(f"[ERROR] Failed to delete submitted test result '{testId}': {e}")
        return False



def update_test_session(testId, updated_metadata):
    """
    Updates the metadata for a single test session by its testId.

    Args:
        testId (str): The ID of the test session to update.
        updated_metadata (dict): The dictionary containing all fields (including unchanged ones).

    Returns:
        bool: True on success, False on failure.
    """
    try:
        # Convert the 'questions' list back to a JSON string for ChromaDB storage
        if 'questions' in updated_metadata and isinstance(updated_metadata['questions'], (list, dict)):
            updated_metadata['questions'] = json.dumps(updated_metadata['questions'])

        # Ensure the document field is present for the upsert call
        document_text = updated_metadata.get("testTitle", "Updated Test Session")

        # Perform the upsert (update in place)
        test_sessions_collection.upsert(
            ids=[testId],
            documents=[f"Test session for user {updated_metadata.get('userId', 'N/A')}"],
            metadatas=[updated_metadata]
        )
        print(f"[DEBUG] Successfully updated test session with ID: {testId}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to update test session: {e}")
        return False




def update_question_bank_metadata(generatedQAId: str, title: str = None, description: str = None):
    """
    Updates the title and/or description of the main question bank metadata document.
    """
    update_result = {"success": False, "title_updated": False, "description_updated": False}

    try:
        # 1. Fetch the existing record to get ALL current metadata and documents
        results = collection.get(ids=[generatedQAId], include=["metadatas", "documents", "embeddings"])

        if not results or not results["ids"]:
            print(f"[WARN] Question bank ID '{generatedQAId}' not found for metadata update.")
            return update_result

        existing_meta = results["metadatas"][0].copy()
        document_text = results["documents"][0]
        # Embeddings are needed for the upsert, even if they aren't changing
        existing_embeddings = results["embeddings"][0]

        # 2. Update the fields if new values are provided
        update_required = False

        if title is not None and existing_meta.get("title") != title:
            existing_meta["title"] = title
            update_required = True
            update_result["title_updated"] = True

        if description is not None and existing_meta.get("description") != description:
            existing_meta["description"] = description
            update_required = True
            update_result["description_updated"] = True

        if not update_required:
            update_result["success"] = True
            return update_result  # Nothing to update

        # 3. If title or description changed, update the embedding (CRITICAL for search accuracy)
        if update_result["title_updated"] or update_result["description_updated"]:
            new_text_for_embedding = f"{existing_meta.get('title', '')} {existing_meta.get('description', '')}"
            new_embeddings = embedding_model.encode([new_text_for_embedding]).tolist()
        else:
            new_embeddings = existing_embeddings

        # 4. Perform the upsert (update in place)
        collection.upsert(
            ids=[generatedQAId],
            documents=[document_text],  # Re-use the existing document text (userIdClean)
            embeddings=new_embeddings,
            metadatas=[existing_meta]
        )
        print(f"[DEBUG] Updated metadata for Question Bank ID: {generatedQAId}")
        update_result["success"] = True
        return update_result

    except Exception as e:
        print(f"[ERROR] Failed to update question bank metadata for {generatedQAId}: {e}")
        return update_result
