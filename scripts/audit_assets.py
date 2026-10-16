import os
import re
import glob

patterns = [
    re.compile(r'[\'\"](/clinical_scans/[^\'\"]+)[\'\"]'),
    re.compile(r'[\'\"](/fundus_images/[^\'\"]+)[\'\"]')
]

ts_files = glob.glob('frontend/src/**/*.tsx', recursive=True)
missing = []
found = set()

for f in ts_files:
    with open(f, 'r', encoding='utf-8') as fp:
        content = fp.read()
    for pat in patterns:
        for match in pat.findall(content):
            local_path = os.path.join('frontend/public', match.lstrip('/'))
            if not os.path.exists(local_path):
                missing.append((f, match, local_path))
            else:
                found.add((f, match))

print(f"Total unique valid image references: {len(found)}")
print(f"Total missing image references: {len(missing)}")
if missing:
    for f, m, lp in missing:
        print(f"  [BUG/FLAG] Missing file: {m} (in {f})")
else:
    print("  [SUCCESS] All static image assets resolve 100% cleanly on disk!")
