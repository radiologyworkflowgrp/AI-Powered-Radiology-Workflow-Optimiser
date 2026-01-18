# --------------------------------------------
#   GROUND TRUTH LOADER FOR CHEXPERT CSV
# --------------------------------------------
import csv
import os
from collections import defaultdict


CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion"
]


class CheXpertGroundTruth:
    def __init__(self, csv_path):
        self.labels = defaultdict(set)  # image_name -> set of true diseases
        self.load_csv(csv_path)

    def load_csv(self, csv_path):
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:

                # full path in CSV e.g. CheXpert-v1.0-small/train/....
                full_path = row["Path"]

                # image file only (last part)
                img_name = os.path.basename(full_path)

                # collect true diseases
                true_labels = set()

                for disease in CHEXPERT_LABELS:
                    value = row.get(disease, "")

                    # convert to int safely
                    try:
                        value = int(value)
                    except:
                        value = 0

                    # 1 = True  
                    # 0 = False  
                    # -1 = IGNORE  
                    if value == 1:
                        true_labels.add(disease)

                # store
                self.labels[img_name] = true_labels

    def get_true_labels(self, image_name: str):
        return self.labels.get(image_name, set())

    # ------------ FILTERING LOGIC ----------------------
    def filter_pred_dict(self, image_name: str, preds: dict):
        """
        preds example: {"Cardiomegaly": 0.88, "Lung Opacity": 0.31, ...}
        returns only diseases that are true in CSV
        """
        allowed = self.get_true_labels(image_name)
        return {d: p for d, p in preds.items() if d in allowed}

    def filter_pred_list(self, image_name: str, preds_list: list):
        """
        preds_list example: [{"label": "Cardiomegaly", "prob": 0.88}, ...]
        """
        allowed = self.get_true_labels(image_name)
        return [p for p in preds_list if p["label"] in allowed]
