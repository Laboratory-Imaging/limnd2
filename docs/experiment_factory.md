# Experiment factory module

This module contains helper classes and functions for creating ExperimentLevel instances using simplified parameters for each experiment type.
Those instances should be used with Nd2Writer instance for altering / creating .nd2 files.

!!! warning
    Since this module is used to creating experiment data structures, you should not use any part of this module if you only read an .nd2 file.

::: limnd2.experiment_factory
    options:
      show_bases: false     # override global settings
