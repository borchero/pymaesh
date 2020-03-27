import datetime
import os

project = 'PyMaesh'
version = os.getenv('CIRCLE_TAG', '0.0.0')
release = os.getenv('CIRCLE_TAG', '0.0.0')

author = 'Oliver Borchert'
copyright = f'{datetime.datetime.now().year}, {author}' # pylint: disable=redefined-builtin

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme'
]
templates_path = []
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = []
