import os
from dotenv import load_dotenv
from .core import EncryptionCore

def load_encrypted_env(filename: str, password: str) -> None:
    core = EncryptionCore()
    key = core.generate_key(password, filename, load_existing_salt=True)
    temp_decrypted = f"{filename}.temp"
    with open(filename, "rb") as enc_file, open(temp_decrypted, "wb") as dec_file:
        fernet = Fernet(key)
        chunk = enc_file.read(1024 * 1024)
        dec_file.write(fernet.decrypt(chunk))
    load_dotenv(temp_decrypted)
    os.remove(temp_decrypted)