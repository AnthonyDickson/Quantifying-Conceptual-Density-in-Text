# COSC480 Project - Quantifying Conceptual Density in Text
A project looking at quantifying the complexity of the underlying knowledge domain for a given text document. 
A document is placed on a spectrum ranging from sparse (concepts are mostly independent) to dense (concepts are tightly integrated).

This repo contains code that attempts to perform the quantification of this idea of conceptual density.
The main approach is to build up a mind-map like graph structure and derive a score from the graph structure.

The original aims and objectives can be found [here](https://github.com/eight0153/cosc480/blob/master/reports/aims/aims.pdf).
The main report can be found [here](https://github.com/eight0153/cosc480/blob/master/reports/technical_report/latex/report.pdf).

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

3.  Install [GraphViz](https://graphviz.gitlab.io/download/) if you want to visualise the generated graph structures.

4.  Install [Java SE](https://www.oracle.com/technetwork/java/javase/overview/index.html) 1.8+

5.  Download the Standford [CoreNLP](https://stanfordnlp.github.io/CoreNLP/) package and extract the contents to a
    directory called `stanford-corenlp` in the root directory of this project.

6.  Run the main script to start quantifying conceptual density!
    ```shell
    $ python -m qcd docs/bread.xml
    ```

    To see the help message type:
    ```shell
    $ python -m qcd --help
    ```
