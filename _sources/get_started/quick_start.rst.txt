Quick start
===========

#. Clone the repo and proceed to the root:

   .. code-block:: bash

      git clone https://github.com/########/EBES.git
      cd EBES

#. Download the data. 
   For example, for X5 Retail Hero dataset, first go to the `data page <https://ods.ai/competitions/x5-retailhero-uplift-modeling/data>`_ and download teh archive. 
   Assuming the archive is downloaded ``to EBES/data/x5-retail/retailhero-uplift.zip`` then:

   .. code-block:: bash

      cd data/x5-retail
      unzip retailhero-uplift.zip  # will create data subfolder with CSV-files
      mkdir preprocessed  # data in EBES format will be stored here
      mkdir preprocessed/cat_codes  # categorical features are encoded into numbers, here will be the mapping

#. Build Docker image and run it:

   .. code-block:: bash

      docker build -t ebes .
      docker run --gpu all --ipc host -v /path/to/EBES:/workspace ebes bash

#. Save the dataset in the EBES format. 
   Starting from now everything should be run inside a docker container.

    .. code-block:: bash

       cd /workspace
       python preprocess/x5-retail.py \
           --data-path data/x5-retail/data \
           --save-path data/x5-retail/preprocessed \
           --cat-codes-path data/x5-retail/preprocessed/cat_codes

#. Run experiment. 
   The majority of experiment configuration is done using YAML config files.
   They are located in ``configs`` folder in the repository.
   To run an experiment you shout choose the dataset on which to run the experiment, the method to benchmark (e.g. vanilla GRU, CoLES, mTAND, etc.), the experiment type (e.g. perform single test run, or launch HPO, etc).
   You can also deside to patch default config with the best found hyper-parameters for the particular method on the particular dataset. 
   These patches are located in ``configs/specify/{dataset}/{method}/best.yaml``.
   For example, to train the best (according to out HPO results) GRU, run from the repository root:

    .. code-block:: bash

       python main.py \
           -d x5-retail  `# dataset config` \
           -m gru        `# method config` \
           -e test       `# experiment config, test for simple single train and test` \
           -s best       `# pick the best config found for gru and x5-retail specifically` \
           -g 'cuda:1'   `# run on gpu 1` \
           --tqdm        `# enable train loop progress`

   During train the folder ``log/{dataset}/{method}/{experiment}`` is created with logs, checkpoints and evaluation results.

   Available experiments (``-e`` option) are:

   * ``test`` --- perform a single run.
   * ``correlation`` --- perform multiple runs in parallel with different random seeds.
   * ``optuna`` --- perform HPO.
     You can parallelize optuna search by launching parallel scripts. 
     In this case they will share the same storage (see `optuna docs <https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html>`_).

#. Analyze results. The notebook with logs analysis is located at ``notebooks/collect_results.ipynb``.


