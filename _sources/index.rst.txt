.. EBES documentation master file, created by
   sphinx-quickstart on Mon Jul  1 00:02:45 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EBES's documentation!
================================

Welcome to EBES (Easy Benchmarking for Event Sequences), a robust benchmarking tool designed for assessing event sequences. 
Each event is defined by a combination of categorical and continuous features, and the time intervals between events can vary randomly rather than being uniform.
Event sequences encompass a broad category that includes time series (including irregularly sampled data) and marked temporal point processes.
In this context, "assessment" refers to classification or regression tasks where the target is linked to the entire sequence.

EBES strives to standardize the evaluation of models tailored for event sequences.
The package provides implementations of various models, a streamlined data preprocessing and loading pipeline, and tools for hyperparameter optimization. 
Configuration is handled through separate YAML files for models, datasets, and evaluation protocols (experiments).
This approach ensures that you can implement a model once, eliminating the need for adjustments across different datasets, thereby enhancing efficiency and consistency.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

    Installation <get_started/install.rst>
    Quick start <get_started/quick_start.rst>

.. toctree::
   :maxdepth: 2
   :caption: User Guide

    Benchmark design <user_guide/design.rst>
    Data format used <user_guide/data_format.rst>
    Configuration files format <user_guide/configs.rst>

.. toctree::
   :maxdepth: 2
   :caption: API Reference

    Modules <reference/modules.rst>



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
