# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
project = 'OARS'
copyright = '2024, Peter Barkley and Robert Bassett'
author = 'Peter Barkley and Robert Bassett'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # 'easydev.copybutton',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    # 'sphinxcontrib_autodocgen',
    'myst_parser',
]

napoleon_custom_sections = [('Returns', 'params_style'),
                            ('Attributes', 'params_style')]

import oars

autodocgen_config = [{
    'modules': [oars],
    'generated_source_dir': './autodocgen-output/',

    # if module matches this then it and any of its submodules will be skipped
    'skip_module_regex': '(.*[.]__|myskippedmodule)',

    # produce a text file containing a list of everything documented. you can use this in a test to notice
    # when you've intentionally added/removed/changed a documented API
    'write_documented_items_output_file': 'autodocgen_documented_items.txt',

    # customize autodoc on a per-module basis
    'autodoc_options_decider': {
        'mymodule.FooBar': {'inherited-members': True},
    },

    # choose a different title for specific modules, e.g. the toplevel one
    'module_title_decider': lambda modulename: 'API Reference' if modulename == 'mymodule' else modulename,
}]

autoclass_content = 'both'

# Include or not the special methods
napoleon_include_special_with_doc = False

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
