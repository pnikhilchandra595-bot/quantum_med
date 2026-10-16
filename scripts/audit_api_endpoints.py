import urllib.request
import urllib.error
import json
import time

BASE_URL = "http://localhost:8000"

endpoints_to_test = [
    ("GET", "/", None),
    ("GET", "/health", None),
    ("GET", "/api/v1/datasets", None),
    ("GET", "/api/v1/models", None),
    ("GET", "/api/v1/multimodal/ecg-lead?rhythm=sinus", None),
    ("GET", "/api/v1/multimodal/ecg-lead?rhythm=afib", None),
    ("GET", "/api/v1/multimodal/ecg-lead?rhythm=vtach", None),
    ("GET", "/api/v1/multimodal/ecg-lead?rhythm=ischemia", None),
    ("POST", "/api/v1/predict", {
        "model_name": "rf",
        "dataset": "wdbc",
        "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871] * 3,
        "n_qubits": 4
    }),
    ("GET", "/api/v1/benchmark", None),
]

print("Auditing FastAPI REST Endpoints...")
all_passed = True

for method, path, body in endpoints_to_test:
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode('utf-8') if body else None
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            content = json.loads(resp.read().decode('utf-8'))
            elapsed_ms = (time.time() - t0) * 1000
            print(f"  [PASS] {method} {path} -> Status {status} ({elapsed_ms:.1f}ms)")
    except urllib.error.HTTPError as e:
        print(f"  [FAIL] {method} {path} -> HTTP Error {e.code}: {e.read().decode()}")
        all_passed = False
    except Exception as e:
        print(f"  [FAIL] {method} {path} -> Exception: {e}")
        all_passed = False

if all_passed:
    print("\nAPI AUDIT RESULT: 100% Endpoints Operational and Validated.")
else:
    print("\nAPI AUDIT RESULT: Failures detected in API endpoints.")
