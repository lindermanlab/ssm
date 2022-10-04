# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ssm'
copyright = '2022, Scott Linderman'
author = 'Scott Linderman'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx_gallery.gen_gallery',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Sphinx-gallery configuration ---------------------------------------------------
# https://sphinx-gallery.github.io/stable/configuration.html

sphinx_gallery_conf = {
    'filename_pattern': '/*.py',
    'ignore_pattern': r'Poisson-HMM-Demo\.py',
    'examples_dirs': ['../notebooks'],
    'gallery_dirs': ['auto_examples'],
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_theme_options = {
    'repository_url': 'https://github.com/lindermanlan/ssm-docs',
    'repository_branch': 'main',
    'launch_buttons': {
        'colab_url': 'https://colab.research.google.com'
    },
}
html_static_path = ['_static']
