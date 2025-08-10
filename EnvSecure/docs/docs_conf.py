# Sphinx config
project = 'EnvSecure'
copyright = '2025, Amirhossein Jafarnezhad'
author = 'Amirhossein Jafarnezhad'
release = '1.0.0'

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = 'alabaster'