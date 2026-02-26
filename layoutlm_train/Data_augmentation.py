import json
import random
import os

# --- Configuration ---
# The name of the file to load and save to.
INPUT_FILE = "unified_training_data_bluuhhhhh.json"
# The maximum allowed deviation for the shift in the x and y directions.
# A range of +/- 5 is used to keep the change subtle but effective.
MAX_SHIFT = 10
# The coordinate boundary limit (assuming coordinates are scaled 0-1000)
MAX_COORD = 1000
MIN_COORD = 0
# Number of augmented copies to create (1 means the original dataset size is doubled)
NUM_AUGMENTATION_COPIES = 1


def clip_coord(coord):
    """Ensures a coordinate stays within the 0 to MAX_COORD boundary."""
    return max(MIN_COORD, min(MAX_COORD, coord))


def augment_data(data, shift_x, shift_y):
    """
    Applies a uniform translation shift to all bounding boxes in the dataset
    and returns the new augmented list of tokens.

    The shift_x and shift_y are the same for all tokens in this copy,
    preserving the crucial relative layout structure.
    """
    augmented_data = []

    for item in data:
        # Create a deep copy of the item to avoid modifying the original data in place
        new_item = item.copy()

        # Bounding box coordinates: [x_min, y_min, x_max, y_max]
        bbox = new_item['bbox']

        # Apply the uniform shift and clip the coordinates
        new_bbox = [
            clip_coord(bbox[0] + shift_x),  # x_min
            clip_coord(bbox[1] + shift_y),  # y_min
            clip_coord(bbox[2] + shift_x),  # x_max
            clip_coord(bbox[3] + shift_y)  # y_max
        ]

        new_item['bbox'] = new_bbox
        augmented_data.append(new_item)

    return augmented_data


def process_dataset():
    """Loads the original data, performs augmentation, and saves the combined data."""
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        print("Please ensure your uploaded JSON file is available and named correctly.")
        return

    print(f"Loading data from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r') as f:
            # Assuming the JSON file is a list of token objects
            original_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{INPUT_FILE}'. Check file format.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    print(f"Original dataset size: {len(original_data)} tokens.")

    all_combined_data = original_data.copy()

    for i in range(NUM_AUGMENTATION_COPIES):
        # 1. Choose a uniform shift for the entire dataset copy
        # This is the core spatial jittering logic.
        shift_x = random.randint(-MAX_SHIFT, MAX_SHIFT)
        shift_y = random.randint(-MAX_SHIFT, MAX_SHIFT)

        print(f"\nCreating augmented copy #{i + 1} with uniform shift (X: {shift_x}, Y: {shift_y})...")

        # 2. Perform the augmentation
        augmented_copy = augment_data(original_data, shift_x, shift_y)

        # 3. Append the augmented data to the combined list
        all_combined_data.extend(augmented_copy)

    print(f"\nAugmentation complete. Total dataset size: {len(all_combined_data)} tokens.")

    # 4. Save the combined (original + augmented) data back to the file
    print(f"Saving combined data back to {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'w') as f:
            # Use indent for readability
            json.dump(all_combined_data, f, indent=2)
        print("Successfully updated the dataset with augmented data.")
    except Exception as e:
        print(f"An error occurred while writing the file: {e}")


if __name__ == "__main__":
    process_dataset()
