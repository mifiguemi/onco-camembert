
import os
import shutil

def copy_txt_without_ann(input_dir, output_dir):
    """
    Copies all .txt files from input_dir to output_dir
    only if they don't have a corresponding .ann file in input_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            basename = os.path.splitext(filename)[0]
            ann_file = f"{basename}.ann"
            ann_path = os.path.join(input_dir, ann_file)
            if not os.path.exists(ann_path):
                src = os.path.join(input_dir, filename)
                dst = os.path.join(output_dir, filename)
                shutil.copy2(src, dst)
                print(f"Copied: {filename}")
            else:
                print(f"Skipped (found .ann): {filename}")

# Example usage:
# copy_txt_without_ann('/path/to/input_dir', '/path/to/output_dir')

if __name__ == "__main__":
    copy_txt_without_ann("../data/brat_format_with_overlap", "../data/brat_format_inference")

