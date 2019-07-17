#!/usr/bin/env bash

# Argument parsing adapted from https://stackoverflow.com/a/5476278
usage="$(basename "$0") [-h] [-a s] [-c n] [-m n] [-o n] [-p n] [-s s] [-t n]
    -- Run an Stanford CoreNLP server locally.

where:
    -h              Show this help text
    -a ANNOTATORS   The annotators to preload into cache (default:
                        \"tokenize,ssplit,pos,parse,depparse,openie\")
    -c MAX_CHARS    The maximum text length in characters (default: 10000)
    -m MAX_MEMORY   The maximum amount of memory to allocate to the JVM in GB
                        (default: 2)
    -o TIMEOUT      The timeout in milliseconds (default: 30000)
    -p PORT         The port to run the server on (default: 9000)
    -s SERVER_PROPS The server properties file to use (default:
                        \"default.props\")
    -t N_THREADS    The number of threads to use (default: 6) See
                        https://stanfordnlp.github.io/CoreNLP/cmdline.html#configuring-corenlp-properties
                        for more details.
"

maxMemory=2 # in GB
port=9000
timeout=30000 # in milliseconds
threads=6
maxCharLength=10000
serverProperties="default.props"
preloadList="tokenize,ssplit,pos,parse,depparse,openie"

while getopts ':ht:j::e' option; do
  case "$option" in
    h) echo "$usage"
       exit
       ;;
    a) preloadList=$OPTARG
       ;;
    m) maxMemory=$OPTARG
       ;;
    p) port=$OPTARG
       ;;
    s) serverProperties=$OPTARG
       ;;
    t) threads=$OPTARG
       ;;
    :) printf "missing argument for -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
   \?) printf "illegal option: -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
  esac
done
shift $((OPTIND - 1))

if [[ "$CORENLP_HOME" == "" ]]
then
    echo "The environment variable CORENLP_HOME is not set."
    echo "Set this to the path to where the CoreNLP jars are located."
    exit 1
fi

java -Xmx${maxMemory}G -cp "${CORENLP_HOME}/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port ${port} -timeout ${timeout} -threads ${threads} -maxCharLength ${maxCharLength} -serverProperties ${serverProperties} -preload ${preloadList}
