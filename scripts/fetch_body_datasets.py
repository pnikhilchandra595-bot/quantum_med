import urllib.request
import os

target_dir = os.path.join("data", "kaggle_uci_datasets")
os.makedirs(target_dir, exist_ok=True)

downloads = [
    # Nephrology: Chronic Kidney Disease
    ("kidney_disease.csv", "https://raw.githubusercontent.com/vamshikrishnajr/chronic-kidney-disease-diagnosis/master/kidney_disease.csv"),
    # Pulmonology: Survey Lung Cancer
    ("survey_lung_cancer.csv", "https://raw.githubusercontent.com/yacwagh/Lung-cancer-prediction/master/survey%20lung%20cancer.csv"),
    # Brain: Brain tumor feature extraction dataset
    ("brain_tumor.csv", "https://raw.githubusercontent.com/datasets/brain-tumor/master/data/brain_tumor.csv")
]

for fname, url in downloads:
    dest = os.path.join(target_dir, fname)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(dest, 'wb') as out_file:
            data = resp.read()
            out_file.write(data)
        print(f"Downloaded {fname}: {len(data)} bytes")
    except Exception as e:
        print(f"Failed {fname} from {url}: {e}")
