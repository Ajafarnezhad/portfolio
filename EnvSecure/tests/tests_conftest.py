import pytest
from envsecure.core import EncryptionCore

@pytest.fixture
def core():
    return EncryptionCore(time_cost=1, memory_cost=1024, parallelism=1)  # Low costs for testing

@pytest.fixture
def temp_env_file(tmp_path):
    file = tmp_path / ".env"
    file.write_text("KEY=VALUE")
    return str(file)