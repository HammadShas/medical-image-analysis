# scripts/data_loading.py

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import time

# Base directory
base_dir = Path("../data/chest_xray")

# Count images function
def count_images(folder):
    """
    Counts the number of images in the NORMAL and PNEUMONIA subfolders.
    Returns a tuple: (normal_count, pneumonia_count)
    """
    normal = len(list((folder / "NORMAL").glob("*.jpeg")))
    pneumonia = len(list((folder / "PNEUMONIA").glob("*.jpeg")))
    return normal, pneumonia

# Display images function
def show_images(subset, label, n=5):
    """
    Displays 'n' sample images for a given subset (train/test/val) and label (NORMAL/PNEUMONIA).
    """
    subset_path = base_dir / subset / label
    images = list(subset_path.glob("*.jpeg"))[:n]

    plt.figure(figsize=(15, 5))
    for i, img_path in enumerate(images):
        img = mimg.imread(img_path)
        plt.subplot(1, n, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"{subset.upper()} - {label}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Count and print image stats
    start = time.time()
    print(f"{'SET':<12} {'NORMAL':>10} {'PNEUMONIA':>12}")
    print("-" * 36)
    for subset in ["train", "test", "val"]:
        subset_path = base_dir / subset
        normal, pneumonia = count_images(subset_path)
        print(f"{subset.capitalize():<12} {normal:>10} {pneumonia:>12}")

    # Show sample images
    for subset in ["train", "test", "val"]:
        for label in ["NORMAL", "PNEUMONIA"]:
            print(f"\nDisplaying {subset.upper()} - {label}")
            show_images(subset, label, n=5)

    end = time.time()
    print(f"\nCompleted in {end - start:.2f} seconds")