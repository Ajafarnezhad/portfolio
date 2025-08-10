import typer
from typing import Optional
from .core import EncryptionCore
from .integrations import load_encrypted_env
from .utils import prompt_password, validate_file_extension

app = typer.Typer(rich_markup_mode="rich")

@app.command()
def encrypt(
    filename: str = typer.Argument(..., help="File to encrypt."),
    min_score: int = typer.Option(3, help="Min password strength score (0-4)."),
    audit_log: bool = typer.Option(False, help="Enable audit logging."),
):
    """Encrypt a .env file."""
    if not validate_file_extension(filename, ".env"):
        typer.echo("Invalid file extension. Must be .env.")
        raise typer.Exit(1)
    password = prompt_password("Enter password for encryption: ")
    core = EncryptionCore()
    if not core.validate_password(password, min_score):
        typer.echo("Password too weak. Aborting.")
        raise typer.Exit(1)
    key = core.generate_key(password, filename, save_salt=True)
    result = core.encrypt(filename, key, audit_log)
    typer.echo(result)

@app.command()
def decrypt(
    filename: str = typer.Argument(..., help="File to decrypt."),
    audit_log: bool = typer.Option(False, help="Enable audit logging."),
):
    """Decrypt a .env.envs file."""
    if not validate_file_extension(filename, ".envs"):
        typer.echo("Invalid file extension. Must be .envs.")
        raise typer.Exit(1)
    password = prompt_password("Enter password for decryption: ")
    core = EncryptionCore()
    key = core.generate_key(password, filename, load_existing_salt=True)
    result = core.decrypt(filename, key, audit_log)
    typer.echo(result)

@app.command()
def load(
    filename: str = typer.Argument(..., help="Encrypted .env to load."),
):
    """Decrypt and load .env into os.environ."""
    password = prompt_password("Enter password to load .env: ")
    load_encrypted_env(filename, password)
    typer.echo("Loaded encrypted .env into environment.")

if __name__ == "__main__":
    app()