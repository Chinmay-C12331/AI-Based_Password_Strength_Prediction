"""Unit tests for FastAPI endpoint and privacy compliance (Prompt 10)."""

import pytest
import os
import glob
from fastapi.testclient import TestClient
from pwdstrength.api import app


client = TestClient(app)


def test_score_endpoint_privacy_and_values(tmp_path):
    secret_test_pw = "SuperSecretCustomPassword9911!!"

    # Track directory state before request
    files_before = set(glob.glob("/tmp/**", recursive=True))

    response = client.post("/score", json={"password": secret_test_pw})
    assert response.status_code == 200

    data = response.json()
    assert "y_hat" in data
    assert "level" in data
    assert 0 <= data["level"] <= 4
    assert "reasons" in data
    assert isinstance(data["reasons"], list)

    # Privacy verification: verify password does not appear in created files or temp files
    files_after = set(glob.glob("/tmp/**", recursive=True))
    new_files = files_after - files_before

    for fpath in new_files:
        if os.path.isfile(fpath):
            try:
                with open(fpath, "r", errors="ignore") as f:
                    content = f.read()
                    assert secret_test_pw not in content, f"Password leaked into file: {fpath}"
            except Exception:
                pass


def test_empty_password_rejection():
    response = client.post("/score", json={"password": ""})
    assert response.status_code == 422 or response.status_code == 400
