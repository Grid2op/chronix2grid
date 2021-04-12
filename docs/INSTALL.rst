Installation
------------

Requirements
^^^^^^^^^^^^^^^^^^^^^^^^
Python >= 3.6

[Optional, recommended] Step 1: Create a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``pip3 install -U virtualenv``

``python3 -m virtualenv venv_chronix2grid``

[First Option] Step 2 - Install from pypi
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``source venv_chronix2grid/bin/activate``

``pip install Chronix2Grid==1.0.1``

or if you want to install optional dependencies (e.g. tensorflow if you want to use GAN generation for solar and wind)

``pip install Chronix2Grid[optional]==1.0.1``


[Second Option] Step 2 - Install from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``source venv_chronix2grid/bin/activate``

``git clone https://github.com/mjothy/ChroniX2Grid.git``

``cd ChroniX2Grid/``

``pip install -U .``

or if you want to install optional dependencies (e.g. tensorflow if you want to use GAN generation for solar and wind)

``pip install -U .[optional]``


Additional install required for dispatch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To achieve economic dispatch with :class:`chronix2grid.dispatch.PypsaEconomicDispatch.PypsaDispatcher`, you may want to install pypsa manually

``pip install pypsa==0.17.0``

[Optional] Compile and output the sphinx doc (this documentation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run
``./docs/make.bat html``
