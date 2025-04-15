import os
import shutil

def create_subset(
    source_dir: str,
    dest_dir: str,
    num_samples: int = 300
):
    """
    Creates a subset of the DeepGlobe dataset with `num_samples` image-mask pairs.
    """
    # Get all *_sat.jpg files from source
    sat_files = sorted([
        f for f in os.listdir(source_dir)
        if f.endswith('_sat.jpg')
    ])[:num_samples]

    # Create destination folders
    dest_img_dir = os.path.join(dest_dir, 'images')
    dest_mask_dir = os.path.join(dest_dir, 'masks')
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_mask_dir, exist_ok=True)

    for sat_file in sat_files:
        mask_file = sat_file.replace('_sat.jpg', '_mask.png')

        src_img_path = os.path.join(source_dir, sat_file)
        src_mask_path = os.path.join(source_dir, mask_file)

        dst_img_path = os.path.join(dest_img_dir, sat_file)
        dst_mask_path = os.path.join(dest_mask_dir, mask_file)

        if os.path.exists(src_mask_path):
            shutil.copy2(src_img_path, dst_img_path)
            shutil.copy2(src_mask_path, dst_mask_path)
        else:
            print(f"Missing mask for: {sat_file} — skipping.")

    print(f"\n{len(sat_files)} image-mask pairs copied to: {dest_dir}")


if __name__ == "__main__":
    create_subset(
        source_dir="data/raw/train",         # <--- your actual dataset
        dest_dir="data/sample_train",        # <--- new folder for subset
        num_samples=300                      # <--- change as needed
    )
