*********************************
Description of implemented models
*********************************

Introduction
============

Some generation models are already implemented within chronix2grid as examples for other model implementation.
This chapter describes the methods they include and how to set their configuration

Load generation
=================

C

Solar generation
=================

C

Wind generation
=================

C

Loss generation
=================

A simple module is actually implemented.
It reads a csv containing a yearly loss pattern chronic (5 min time step in the example provided), given as an absolute power value in MW.
Two inputs are necessary, with example provided in *getting_started/example/input*:

* A csv file containing the yearly loss pattern  in *patterns/loss_pattern.csv*
* A json parameter file that indicates the path to loss pattern in *case118_l2rpn_wcci/generation/params_loss.json*


Economic dispatch generation (hydro and thermic generators)
====================================================================

C

