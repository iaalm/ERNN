import os
import sys
import tempfile
import subprocess
import logging

logging.basicConfig(
    # level=logging.INFO,
    format='%(asctime)-15s %(name)-8s %(levelname)-8s %(message)s')

# logger = logging.getLogger(__name__)
log_file = os.path.join(sys.argv[1], 'log')
with open(log_file) as fd:
    data = [i.strip().split() for i in fd if i.startswith('born')]
logging.info('Data loaded.')

iset = set(i[3] for i in data)
logging.info('Writing to dot.')

fd = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
fd.close()

p = subprocess.Popen(['dot', '-Tpng', '-o', fd.name],
                     stdout=subprocess.PIPE,
                     stdin=subprocess.PIPE)
txt = "DiGraph {\n"
for i in data:
    if i[5] in iset:
        txt = txt + "{} -> {};\n".format(i[3], i[5])
txt = txt + "}\n"
p.communicate(txt.encode('utf-8'))

logging.info('Show.')
os.system('eog {}'.format(fd.name))
os.unlink(fd.name)
