import json
import random
import sys

def add_label_to_data(json_path, label, output_path="data_labeled.json"):
    """
    Loads the JSON file created by the image annotation code, 
    appends the specified label to each saved set, and writes
    the result to a new JSON file.
    
    Parameters:
        json_path (str): Path to the existing JSON file.
        label (int): The label to add (e.g., 1 for few layers, 0 for not).
        output_path (str): Path for the output labeled JSON file.
    """
    # Load existing data
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Loop over each image and each set, appending the label.
    for img_file in data:
        updated_sets = []
        for flake_set in data[img_file]:
            # Check if this set already has a label (avoid duplicate labeling)
            if len(flake_set) == 2:
                # Append the label, making the set [background, flake, label]
                updated_sets.append(flake_set + [label])
            else:
                # If already labeled, you might decide to update or skip
                updated_sets.append(flake_set)
        data[img_file] = updated_sets
    
    # Write the updated data to a new JSON file.
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Labeled data has been saved to {output_path}")

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def combine_and_shuffle(file1, file2, output_file):
    # Load both labeled datasets.
    data_true = load_json(file1)
    data_false = load_json(file2)

    combined = []

    # Process first file (e.g., labeled true) and add each data entry with its filename.
    for filename, items in data_true.items():
        for item in items:
            combined.append({
                "filename": filename,
                "data": item
            })

    # Process second file (e.g., labeled false) similarly.
    for filename, items in data_false.items():
        for item in items:
            combined.append({
                "filename": filename,
                "data": item
            })

    # Shuffle the combined list randomly.
    random.shuffle(combined)

    # Write the combined and shuffled data to the output file.
    with open(output_file, 'w') as f:
        json.dump(combined, f, indent=4)

    print(f"Combined and shuffled {len(combined)} items saved to {output_file}")


if __name__ == "__main__":
    true_data_points_path = "datapoints/true_data_points.json"
    true_labeled_data_points_path = "datapoints/labeled_true_data_points.json"
    flase_data_points_path = "datapoints/false_data_points.json"
    false_labeled_data_points_path = "datapoints/labeled_false_data_points.json"
    add_label_to_data(true_data_points_path, 1, output_path = true_labeled_data_points_path)
    add_label_to_data(flase_data_points_path, 0, output_path = false_labeled_data_points_path)
    output_path = "final_data.json"
    combine_and_shuffle(true_labeled_data_points_path,false_labeled_data_points_path,output_path)