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
    Skip to step 7. if you are not planning on running scripts that use CoreNLP.

4.  Install [Java SE](https://www.oracle.com/technetwork/java/javase/overview/index.html) 1.8+

5.  Download the Standford [CoreNLP](https://stanfordnlp.github.io/CoreNLP/) package and extract the contents somewhere.
    Then add the path to where you extracted the files to the environment variable `CORENLP_HOME`:
    ```shell
    $ export CORENLP_HOME=path/to/corenlp
    ```
    This sets the environment variable for the current shell.
    
    To make this environment variable more permanent run:
    ```shell
    $ echo 'export CORENLP_HOME=path/to/corenlp"' >> ~/.bashrc
    ```
    These changes will take effect when you next login.

6.  Start up the CoreNLP server:
    ```shell
    $ bash corenlp_server/run.sh
    ```
    
    To see the help message type:
    ```shell
    $ bash corenlp_server/run.sh -h
    ```    

7.  Run the main script to start quantifying conceptual density!
    ```shell
    $ python -m qcd docs/bread.xml
    ```

    To see the help message type:
    ```shell
    $ python -m qcd --help
    ```
