Configuration format
====================

Overview
--------

The configuration is done via YAML files located in `configs` folder in the repository root.
We rely on the extended YAML syntax of `Omega Conf <https://omegaconf.readthedocs.io/en/2.3_branch/index.html>`_.
The idea behind the configuration design is to reduce the complexity of configuration from AB to A + B, e.g. to add new method or dataset you should only write config file for that method or dataset solely instead of configuring it for each dataset in the benchmark.

To configure a run you should configure at least *dataset*, *method* and *experiment*.
These config are combined to form a holistic run confg.
To combine the configs we use the ``omegaconf.OmegaConf.merge`` function.
It processes the configs in order in which they are passed, interpolates fields and handles overrides, so it is possible to refer to some dataset config fields in the method config, or override some values.
The latter option is heavily used to launch runs methods for the best hyper-parmeters of the particular method on the particular dataset.

Dataset
-------

Dataset confgs are located in the `configs/datasets` folder and have the following structure.

