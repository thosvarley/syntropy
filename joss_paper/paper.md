---
title: "Syntropy: a Python package for discrete and continuous information theory."
tags:
    - information theory 
    - data science
    - complex systems 
authors:
    - name: Thomas F. Varley
      orcid: 0000-0002-3317-9882 
      affiliation: "1"
affiliations:
    - name: Vermont Complex Systems Institute, University of Vermont, Burlington, VT, USA. 
      index: 1
date: 01 May 2025
bibliography: references.bib
---

# Summary:
Information theory (a branch of mathematics concerned with the logic of inference and uncertainty) has emerged as a kind of *lingua franca* for the study of complex systems [@varley_information_2024], finding applications in neuroscience [@timme_synergy_2014][@varley_serotonergic_2024], climatology [@goodwell_its_2020][@goodwell_debatesdoes_2020], developmental biology [@blackiston_revealing_2025], economics [@kim_predicting_2020][@rajpal_synergistic_2025], sociology [varley_untangling_2022] and more. 
The goal of the Syntropy package is to provide an accessible, easy-to-use API for scientists interested in applying information-theoretic analyses to arbitrary data sets. 
To this end, Syntropy implements a variety of different measures, as well as a variety of estimators for different types of data (discrete, continuous, time series, etc). 
Being primarily concerned with multivariate information theory and the application to complex systems, a core part of the Syntropy package is implementing the various information decompositions and higher-order mutual information measures that have been developed in recent years for the study of redundant and synergistic interactions in complex systems. 

The Syntropy package takes considerable inspiration from the Networkx package [@hagberg_exploring_2008]. 
By providing an accessible, high-level interface for complex analyses, the developers of Networkx have made network science and graph-theoretic tools available to a much more diverse range of scientists than would ever have been possible if every researchers was required to develop their own functions. 
The goal of Syntropy is to be the Networkx of information theory"; lowering the barrier to entry fro information theory so that powerful tools are accessible to scientists from many different fields. 
Accordingly, the Syntropy package is written in pure Python, with minimal external dependencies, all of which are standard libraries for scientific computing in Python, including ``numpy``, ``scipy``, and ``networkx`` packages. 

While the core information-theoretic functions have been implemented and tested for correctness, the goal is for Syntropy to be a "living package": updating the library of functions and estimators as the field of multivariate information theory continues to develop. Future work may also include porting the library to other scientific programming languages such as Julia, R, or MATLAB. 

# Description of package
The Syntropy package is broken up into two main arms: discrete estimators (which operate on discrete, multivariate probability distribution represented by Python dictionaries), and continuous estimators (currently focused on parametric Gaussian estimators based on time series and covariance matrices). 
Within each branch, we have several sub-modules that describe different families of analysis. 
There is a ``shannon`` module which contains basic Shannon information theory functions (entropy, conditional entropy, mutual information, Kullback-Leibler divergence, etc). 
There is a ``multivariate mi`` module that includes higher-order generalizations of mutual information and their extensions (total correlation, dual total correlation, O-information, etc [@rosas_quantifying_2019]). 
The ``decomposition`` module contains the partial information [@williams_nonnegative_2010], partial entropy [@ince_partial_2017], and generalized information decompositions [@varley_generalized_2024]. 
The ``temporal`` module contains a small number of functions designed for time series analysis specifically [@blanc_quantifying_2008][@sparacino_decomposing_2025]. 
Finally, both branches have a ``utils`` module which contains basic operations on discrete and Gaussian probability distributions, as well as some example distributions of theoretical note (such as the James Triadic and Dyadic distribution [@james_multivariate_2017]. 
These primarily support the higher-level functions, but may be of interest to those working on more basic probability theory.

Wherever possible, we have tried to ensure overlap between the two branches, ensuring that each information theoretic function has both discrete and continuous implementations (e.g. there is a discrete and continuous entropy estimator, a discrete and continuous mutual information estimator, etc). 
This ensures that researchers can select the most appropriate tool for the data they have, rather than being forced to transform their data to make it fit (e.g. discretizing continuous data). 

# Statement of need
While many different computing packages for information-theoretic analysis exist, they are typically restricted to only a single subset of analyses or data type. 
For example, the DIT package [@james_dit_2018] is specific to discrete information theory and uses a customized distribution object. 
The IDTxl [@wollstadt_idtxl_2019] package implements several different classes of estimators, but is focused specifically on information dynamics and time series analysis, as is the JIDT package [@lizier_jidt_2014]. 
Information decomposition analysis in particular is an outstanding issue: DIT and IDTxl have limited support for the partial information decomposition (PID) of discrete random variables, but no other package also implements the PID for continuous Gaussian random variables, or supports the partial entropy decomposition (PED), generalized information decomposition (GID), or alpha-synergy decomposition. 
Several other classes of analyses are also included in Syntropy that are not present in other primary packages, including spectral estimators for Gaussian autoregressive processes, and Lempel-Ziv estimators for discrete dynamic processes. 
 
# Acknowledgments 
This work was supported by the Cold Regions Research and Engineering Laboratory, under contract number W913E524C0012. TFV would also like to thank Dr. Olaf Sporns and Dr. Maria Pope for extensive support and discussion about all things information theory over many years, and Dr. Josh Bongard for support while finalizing this package. 

# References
