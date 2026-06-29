Installation
============

Syntropy requires Python 3.9 or newer.

From PyPI
---------

.. code-block:: bash

   pip install syntropyx

The package is then imported as ``syntropy``:

.. code-block:: python

   import syntropy

.. note::

   The distribution is named ``syntropyx`` on PyPI because the name ``syntropy``
   was already taken. The ``x`` is only a packaging workaround; the library
   itself is imported and referred to as **Syntropy**.

Optional dependencies
---------------------

The neural (normalizing-flow) estimators require additional packages, which can
be installed via the ``neural`` extra:

.. code-block:: bash

   pip install "syntropyx[neural]"

This pulls in ``torch`` and ``nflows``.

From source
-----------

To install the latest development version:

.. code-block:: bash

   git clone https://github.com/thosvarley/syntropy.git
   cd syntropy
   pip install -e ".[dev]"

Dependencies
------------

Core dependencies (installed automatically):

* ``numpy >= 1.20.0``
* ``scipy >= 1.7.0``
* ``networkx >= 2.6.0``

Optional (``neural`` extra):

* ``torch >= 1.9.0``
* ``nflows >= 0.14``
