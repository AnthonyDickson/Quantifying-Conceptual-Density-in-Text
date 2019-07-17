import os
import sys
import time

from stanfordnlp.server.client import ShouldRetryException
from stanfordnlp.server import CoreNLPClient

if __name__ == '__main__':
    if not os.environ['CORENLP_HOME']:
        print('ERROR: Environment variable $CORENLP_HOME is not set.', file=sys.stderr)
        print('Set $CORENLP_HOME to point to the directory containing the CORENLP jars.', file=sys.stderr)
        exit(1)

    # set up the client
    print('---')
    print('starting up Java Stanford CoreNLP Server...')

    # set up the client
    with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'parse', 'depparse', 'openie'],
                       timeout=30000, memory='2G') as client:
        timeout = 30  # seconds

        for i in range(timeout):
            print('\rWaiting for server to startup' + '.' * (i % 3) + ' ' * 3, end='')
            try:
                if client.is_alive():
                    break
                else:
                    time.sleep(1)
            except ShouldRetryException:
                time.sleep(1)
        else:
            print('Timed out after %d seconds.' % timeout, file=sys.stderr)
            exit(1)

        print('\nServer started.')

        while client.is_alive():
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                print('Stopping server...')
                break
