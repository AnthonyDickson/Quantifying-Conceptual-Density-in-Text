# COSC480 Project - Quantifying Conceptual Density in Text
A project looking at quantifying the complexity of the underlying knowledge domain for a given text document. 
A document is placed on a spectrum ranging from sparse (concepts are mostly independent) to dense (concepts are tightly integrated).

This repo contains code that attempts to perform the quantification of this idea of conceptual density.
The main approaches are:
 - A co-occurrence graph
 - A graph encoding the relationships between concepts focusing on sections and section headers.
 
The report for this project is currently hosted in [this GitHub repository](https://github.com/eight0153/Quantifying-Conceptual-Density-in-Text).

# Getting Started
1.  Set up your python environment.
    If you are using conda you can do this with the command:
    ```shell
    $ conda env create -f environment.yml
    ```
    and then activate the environment using:
    ```shell
    $ conda activate cosc480
    ```
    Otherwise, make sure you have the packages listed in 
    `environment.yml` installed.
    
2.  Run the setup script:
    ```shell
    $ python setup.py
    ```

3.  [Install GraphViz](https://graphviz.gitlab.io/download/) if you want to visualise the generated graph structures.
