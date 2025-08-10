import pytest
from hypothesis import given, strategies as st
from envsecure.core import EncryptionCore

def test_instance(core):
    assert isinstance(core, EncryptionCore)

@given(st.text(min_size=8))
def test_generate_key(core, password, temp_env_file):
    key = core.generate_key(password, temp_env_file, save_salt=True)
    assert isinstance(key, bytes)
    assert len(key) == 44  # Base64 encoded 32 bytes

def test_encrypt_decrypt(core, temp_env_file):
    password = "strongpassword123"
    key = core.generate_key(password, temp_env_file, save_salt=True)
    core.encrypt(temp_env_file, key)
    core.decrypt(f"{temp_env_file}.envs", key)
    with open(temp_env_file, "r") as f:
        assert f.read() == "KEY=VALUE"

def test_invalid_password(core, temp_env_file):
    password = "strongpassword123"
    key = core.generate_key(password, temp_env_file, save_salt=True)
    core.encrypt(temp_env_file, key)
    wrong_key = core.generate_key("wrong", temp_env_file, load_existing_salt=True)
    with pytest.raises(InvalidToken):
        core.decrypt(f"{temp_env_file}.envs", wrong_key)

def test_weak_password(core):
    assert not core.validate_password("weak", min_score=3)
    assert core.validate_password("StrongP@ssw0rd123!", min_score=3)