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

``pip install Chronix2Grid``


[Second Option] Step 2 - Install from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``source venv_chronix2grid/bin/activate``

``git clone https://github.com/mjothy/ChroniX2Grid.git``

``cd ChroniX2Grid/``

``pip install -U .``


[Optional] Compile and output the sphinx doc (this documentation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run
``./docs/make.bat html``
