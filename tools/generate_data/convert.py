from PIL import Image
import os
import numpy as np
import pandas as pd
import time

DATA_DIR = "longoutput"
OUTPUT = "longruns.npz"

def load_image(image_path):
    # Convert image to grayscale ('L' mode) and ensure it's stored as uint8
    with Image.open(image_path) as img:
        gray_img = img.convert("L")
        return np.array(gray_img, dtype=np.uint8)


def main():
    base_dir = os.path.join(
        os.getcwd(), DATA_DIR
    )  # current directory (should be the output directory)

    all_images = []
    all_metadata = []

    last_print = 0
    last_print_time = time.time()

    count = len(os.listdir(base_dir))
    print(f"Found {count} folders in {base_dir}.")
    for i in range(1, count + 1):
        # Print progress every 10%
        if i - last_print >= (count * 0.1):
            last_duration = time.time() - last_print_time
            time_remaining = last_duration * (count - i) / (count * 0.1)
            print(
                f"Processing folder {i}/{count} | Expected time remaining: {time_remaining:.1f} seconds"
            )
            last_print = i
            last_print_time = time.time()
        folder_name = str(i)
        folder_path = os.path.join(base_dir, folder_name)
        csv_path = os.path.join(folder_path, "data.csv")

        # Load CSV and drop the "frame" column
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue

        if "frame" in df.columns:
            df = df.drop(columns=["frame"])

        # Process each row assuming row order corresponds to the image filename number (1-indexed)
        for idx, row in df.iterrows():
            image_filename = f"frame_{idx + 1}.png"
            image_path = os.path.join(folder_path, image_filename)

            try:
                img_array = load_image(image_path)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue

            all_images.append(img_array)
            all_metadata.append(row.to_numpy())

    print(f"Loaded {len(all_images)} images and {len(all_metadata)} metadata rows.")

    # Convert lists to numpy arrays
    images_np = np.array(all_images)
    print("Converted images to numpy array.")
    labels_np = np.array(all_metadata)
    print("Converted metadata to numpy array.")

    # Save together as a compressed numpy archive (.npz)
    print(f"Saving {OUTPUT}...")
    np.savez_compressed(OUTPUT, images=images_np, labels=labels_np)
    print(f"Saved {OUTPUT} successfully.")

    # Save as uncompressed numpy file (.npy)
    # print(f"Saving {OUTPUT}...")
    # # data = {"images": images_np, "labels": labels_np}
    # # np.save(OUTPUT, data)
    # np.save(OUTPUT, images_np)
    # print(f"Saved {OUTPUT} successfully.")


if __name__ == "__main__":
    main()
