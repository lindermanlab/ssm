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
    'myst_nb',
    'sphinx_gallery.gen_gallery',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    # "html_admonition",
    # "html_image",
    "colon_fence",
    # "smartquotes",
    # "replacements",
    # "linkify",
    # "substitution",
]
nb_execution_timeout = 600

# -- Sphinx-gallery configuration ---------------------------------------------------
# https://sphinx-gallery.github.io/stable/configuration.html

from sphinx_gallery.sorting import FileNameSortKey
sphinx_gallery_conf = {
    'filename_pattern': '/*.py',
    'ignore_pattern': r'Poisson-HMM-Demo\.py',
    'examples_dirs': ['../notebooks'],
    'within_subsection_order': FileNameSortKey,
    'gallery_dirs': ['auto_examples'],
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_theme_options = {
    'repository_url': 'https://github.com/lindermanlan/ssm-docs',
    'repository_branch': 'main',
    'launch_buttons': {
        'colab_url': 'https://colab.research.google.com',
        'binderhub_url': 'https://mybinder.org'
    },
}
html_static_path = ['./_static']