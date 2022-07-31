# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'SIGLE'
copyright = '2022, Q.Duchemin & Y.De Castro'
author = 'Q.Duchemin & Y.De Castro'



import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../'))
import PSILOGIT
import sphinx_gallery
from docs.source.github_link import make_linkcode_resolve

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
mathjax_path="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    "sphinx.ext.linkcode",
    'sphinx_gallery.gen_gallery',
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo' # 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

add_module_names = False




# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
# texinfo_documents = [
#     (master_doc, 'celer', u'celer Documentation',
#      author, 'celer', 'One line description of project.',
#      'Miscellaneous'),
# ]


# intersphinx_mapping = {
#     # 'numpy': ('https://docs.scipy.org/doc/numpy/', None),
#     # 'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
#     'matplotlib': ('https://matplotlib.org/', None),
#     'sklearn': ('http://scikit-learn.org/stable', None),
# }


sphinx_gallery_conf = {
    'doc_module': ('PSILOGIT', 'sklearn'),
    'reference_url': dict(PSILOGIT=None),
    'examples_dirs': '../../examples',
    'gallery_dirs': 'auto_examples',
    'reference_url': {
        'PSILOGIT': None,
    }
}

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    "PSILOGIT",
    "https://github.com/quentin-duchemin/"
    "SIGLE/blob/{revision}/"
    "{package}/{path}#L{lineno}",
)


def setup(app):
    app.add_css_file('style.css')