import os
import shutil
import argparse

def remove_label_from_ann_files(source_dir, target_dir, label_to_remove="expression_CIM"):
    """
    Copies all .ann files from source_dir to target_dir,
    removing all lines that contain the specified label.
    Also copies .txt files only if a corresponding .ann file exists.
    """
    os.makedirs(target_dir, exist_ok=True)

    # Collect all base filenames that have .ann files
    ann_basenames = {os.path.splitext(filename)[0] for filename in os.listdir(source_dir) if filename.endswith(".ann")}

    for basename in ann_basenames:
        ann_filename = f"{basename}.ann"
        txt_filename = f"{basename}.txt"

        source_ann_path = os.path.join(source_dir, ann_filename)
        target_ann_path = os.path.join(target_dir, ann_filename)

        # Process .ann file: remove lines containing the label
        with open(source_ann_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        filtered_lines = [line for line in lines if label_to_remove not in line]

        with open(target_ann_path, 'w', encoding='utf-8') as f:
            f.writelines(filtered_lines)

        # If corresponding .txt file exists, copy it
        source_txt_path = os.path.join(source_dir, txt_filename)
        target_txt_path = os.path.join(target_dir, txt_filename)

        if os.path.exists(source_txt_path):
            shutil.copy2(source_txt_path, target_txt_path)
        else:
            print(f"Warning: Corresponding .txt file not found for {ann_filename}")

    print(f"Finished processing .ann and corresponding .txt files from {source_dir} to {target_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove labeled lines from .ann files.")
    parser.add_argument("--source", default = "../data/brat_format_with_overlap", help="Path to the source directory with .ann files")
    parser.add_argument("--target", default = "../data/brat_format", help="Path to the target directory to save modified .ann files")
    parser.add_argument("--label", default="expression_CIM", help="Label to remove (default: expression_CIM)")

    args = parser.parse_args()

    remove_label_from_ann_files(args.source, args.target, args.label)