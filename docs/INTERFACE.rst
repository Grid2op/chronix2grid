.. _model-interface:

Interface API Overview
============================

In the workflow illustration below, each class, method and function is designed to be edited for your own generation models

.. image:: ../pictures/workflow.png

Interface example
==================

Chronix2grid currently integrates a whole generation model (see :ref:`implemented-models` for full description of the algorithms).

Additionally, an alternative method has been interfaced for solar and wind generation with GAN based
on a `research paper <https://arxiv.org/abs/1707.09676>`_. It enables to replay a trained neural network with tensorflow designed for generation,
and then compare it with its training data. See example in *input_data/generation/case118_l2rpn_neurips_1x_GAN* and **input_data/kpi/case118_l2rpn_neurips_1x_GAN*

GeneratorBackend
==================

.. autoclass:: chronix2grid.GeneratorBackend.GeneratorBackend
    :members:

Constants
==================

.. automodule:: chronix2grid.constants
    :members:

ConfigManager
==================

.. automodule:: chronix2grid.config
    :members:

ConsumptionGeneratorBackend
==============================

.. automodule:: chronix2grid.generation.consumption.ConsumptionGeneratorBackend
    :members:

RenewableBackend
==================

.. automodule:: chronix2grid.generation.renewable.RenewableBackend
    :members:

LossBackend
==================

.. automodule:: chronix2grid.generation.loss.LossBackend
    :members:

DispatchBackend
==================

.. automodule:: chronix2grid.generation.dispatch.DispatchBackend
    :members:

Dispatcher
==================

.. autoclass:: chronix2grid.generation.dispatch.EconomicDispatch.Dispatcher
    :members:


KPI preprocessing
=================

.. automodule:: chronix2grid.kpi.preprocessing.pivot_KPI
    :members:

EconomicDispatchValidator
=================================

.. automodule:: chronix2grid.kpi.deterministic.kpis
    :members:


