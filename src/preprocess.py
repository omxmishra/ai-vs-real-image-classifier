import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil


# Paths
RAW_DIR = Path("../data/raw")
PROCESSED_DIR = Path("../data/processed")
TRAIN_DIR = Path("../data/train")
TEST_DIR = Path("../data/test")

IMG_SIZE = (224, 224)


def resize_images():
    for label in ["real", "ai_generated"]:
        for category in os.listdir(RAW_DIR / label):

            input_path = RAW_DIR / label / category
            output_path = PROCESSED_DIR / label / category

            output_path.mkdir(parents=True, exist_ok=True)

            for img_name in tqdm(os.listdir(input_path), desc=f"{label}-{category}"):
                img_path = input_path / img_name
                save_path = output_path / img_name

                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(IMG_SIZE)
                    img.save(save_path)
                except:
                    continue


def split_data():
    for label in ["real", "ai_generated"]:
        for category in os.listdir(PROCESSED_DIR / label):

            category_path = PROCESSED_DIR / label / category
            images = os.listdir(category_path)

            train_imgs, test_imgs = train_test_split(
                images, test_size=0.2, random_state=42
            )

            (TRAIN_DIR / label / category).mkdir(parents=True, exist_ok=True)
            (TEST_DIR / label / category).mkdir(parents=True, exist_ok=True)

            for img in train_imgs:
                shutil.copy(category_path / img, TRAIN_DIR / label / category / img)

            for img in test_imgs:
                shutil.copy(category_path / img, TEST_DIR / label / category / img)


def run_preprocessing():
    print("Resizing images...")
    resize_images()

    print("Splitting dataset...")
    split_data()

    print("Preprocessing completed ✅")


if __name__ == "__main__":
    run_preprocessing()