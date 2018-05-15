import os
import sys
import tempfile
import subprocess

log_file = os.path.join(sys.argv[1], 'log')
with open(log_file) as fd:
    data = [i.strip().split() for i in fd if i.startswith('born')]

iset = set(i[3] for i in data)

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

os.system('eog {}'.format(fd.name))
os.unlink(fd.name)
