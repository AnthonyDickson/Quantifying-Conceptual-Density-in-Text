[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3474469.svg)](https://doi.org/10.5281/zenodo.3474469)

# COSC480 Project - Quantifying Conceptual Density in Text
This repository contains the source code for my COSC480 project, "Quantifying Conceptual Density in Text".

Copied from the abstract of the report, a description of the project:
> "Conceptual density" is a term that describes the degree to which concepts in 
> a domain are integrated, or interdependent.  There is a hypothesis that text 
> documents with high conceptual density are harder to process.  At present, 
> the concept of “conceptual density” has been informally defined.  In this 
> pilot study, I investigate the idea of conceptual density in the context of 
> expository text documents, a graph-based model for quantifying conceptual 
> density and ways to evaluate this model.  Finally, I discuss open problems 
> and direction for future work related to conceptual density.

The original aims and objectives can be found [here](https://github.com/eight0153/cosc480/blob/master/reports/aims/aims.pdf).
The main report can be found [here](https://github.com/eight0153/cosc480/blob/master/reports/technical_report/latex/report.pdf).

# Getting Started
1.  Set up your python environment.
    If you are using conda you can do this with the command:
    ```shell script
    conda env create -f environment.yml
    ```
    and then activate the environment using:
    ```shell script
    conda activate cosc480
    ```
    Otherwise, make sure you have the packages listed in 
    `environment.yml` installed.
    
2.  Run the setup script:
    ```shell script
    python setup.py
    ```

3.  Install [GraphViz](https://graphviz.gitlab.io/download/) if you want to visualise the generated graph structures.
    Skip to step 7. if you are not planning on running scripts that use CoreNLP.

4.  Install [Java SE](https://www.oracle.com/technetwork/java/javase/overview/index.html) 1.8+

5.  Download the Standford [CoreNLP](https://stanfordnlp.github.io/CoreNLP/) package and extract the contents somewhere.
    Then add the path to where you extracted the files to the environment variable `CORENLP_HOME`:
    ```shell script
    export CORENLP_HOME=path/to/corenlp
    ```
    This sets the environment variable for the current shell.
    
    To make this environment variable more permanent run:
    ```shell script
    echo export CORENLP_HOME=path/to/corenlp >> ~/.bashrc
    ```
    These changes will take effect when you open a new shell.

6.  Start up the CoreNLP server:
    ```shell script
    bash corenlp_server/run.sh
    ```
    
    To see the help message type:
    ```shell script
    bash corenlp_server/run.sh -h
    ```
    
    When using the default settings, you can access the server from [localhost:9000](http://localhost:9000/) and
    use the web interface to issue queries.

7.  Run the main script to start quantifying conceptual density!
    ```shell script
    python -m qcd docs/bread.xml
    ```

    To see the help message type:
    ```shell script
    python -m qcd --help
    ```

## Annotating Documents
There is a separate GitHub repository for a web application that facilitates
annotation of documents and generating the XML documents with annotations.
This web app can be accessed from [here](https://cosc480-document-annotator.herokuapp.com/documents)
and the GitHub repository can be found [here](https://github.com/eight0153/cosc480-annotator).

## Evaluating the Model
The easiest way for evaluating the model of conceptual density on a document 
goes like this:

1.  Go to the [web application](https://cosc480-document-annotator.herokuapp.com/documents)
    and create a document.
    
2.  Annotate the document.

3.  Download the annotated XML version of the document either via the web 
    application interface or command line.
    
    For example, we can download the annotated XML document via command line
    with the following:
    ```shell script
    curl -o annotations.xml https://cosc480-document-annotator.herokuapp.com/api/documents/1/xml
    ``` 
    
    The exact URL for a given document can be found through the link in download
     button in web app interface, or by using the URL in the above example and 
     replacing the document ID with the ID of the document you want. Document 
     IDs can be found in the URL when viewing a document in the web app or by 
     inspecting the JSON response from the endpoint [cosc480-document-annotator.herokuapp.com/api/documents](https://cosc480-document-annotator.herokuapp.com/api/documents/).
     
     It may be the case that annotations need to spread across multiple copies of a document (e.g. overlapping annotations).
     In this case you either evaluate the conceptual density model on each document separately or merge the XML documents
     with:
     ```shell script
    python -m qcd.merge_xml doc-1.xml doc-2.xml ... doc-n.xml -output-path annotations.xml
    ``` 
    and evaluate the model using the merged XML document.

4.  Run the evaluation script on a given document:
    ```shell script
    python -m qcd.evaluate annotations.xml
    ```
