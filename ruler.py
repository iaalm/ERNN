import os
import json
import random
import pickle
import fcntl
import networkx as nx
from matplotlib import pyplot as plt
from network import network, NotPossibleError
from layer import *
from xmlrpc.server import SimpleXMLRPCServer
from functools import partial
from termcolor import colored
from multiprocessing import Pool


def get_score(path, name):
    with open(os.path.join(path, name, 'cell.lua')) as fd:
        try:
            score = float(fd.readline().split(' ')[1].strip())
        except ValueError:
            score = -0.01
    return (name, score)


class simpleFileSystemRuler:
    def __init__(self, workdir, n_live, n_hidden):
        self.workdir = workdir
        self.n_hidden = n_hidden

        dirs = os.listdir(workdir)
        for i in ['live', 'dead', 'born']:
            if i not in dirs:
                os.mkdir(os.path.join(self.workdir, i))
        live_path = os.path.join(self.workdir, 'live')
        lock_path = os.path.join(self.workdir, 'lock')
        if not os.path.exists(lock_path):
            max_id = max([0]+[int(i) for i in os.listdir(live_path)])
            with open(lock_path, 'w') as fd:
                print(max_id + 1, file=fd)

        self.lock = open(lock_path, 'r+')
        # init first
        dirs = os.listdir(live_path)
        for _ in range(n_live - len(dirs)):
            net = network(n_hidden)
            nid = self.newId()
            npath = os.path.join(self.workdir, 'live', nid)
            os.mkdir(npath)
            with open(os.path.join(npath, 'cell.pickle'), 'wb') as fd:
                pickle.dump(net, fd)
            net.writeLua(os.path.join(npath, 'cell.lua'))

    def newId(self):
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        self.lock.seek(0)
        new_id = int(self.lock.read())
        self.lock.seek(0)
        self.lock.write(str(new_id + 1))
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        return str(new_id)

    def born(self):
        live_path = os.path.join(self.workdir, 'live')
        cands = os.listdir(os.path.join(live_path))
        cand = random.choice(cands)
        print('------born net from ' + cand)
        pickle_path = os.path.join(live_path, cand, 'cell.pickle')
        with open(pickle_path, 'rb') as fd:
            net = pickle.load(fd)
        net = self.mutate(net)
        nid = self.newId()
        print('------new id ', nid)
        npath = os.path.join(self.workdir, 'born', nid)
        os.mkdir(npath)
        with open(os.path.join(npath, 'cell.pickle'), 'wb') as fd:
            pickle.dump(net, fd)
        os.system('cp -lr neuraltalk2-ERNN %s' % os.path.join(npath, 'code'))
        net.writeLua(os.path.join(npath, 'code', 'cell.lua'))

        return npath

    def fight(self, path):
        json_path = os.path.join(path, 'code', 'model_.json')
        with open(json_path) as fd:
            data = json.load(fd)
        val_max = {}
        val_data = data["val_lang_stats_history"]
        val_data = sorted(val_data.items(), key=lambda t: int(t[0]))

        max_result = -0.01
        val_max = {}
        for k, v in val_data:
            if v['CIDEr'] > max_result:
                max_result = v['CIDEr']
                for metric in v:
                    val_max[metric] = v[metric]
                val_max['pos_max'] = k
        # write cell.lua
        with open(os.path.join(path, 'cell.lua'), 'w') as fd:
            print('-- %f' % max_result, file=fd)
            print('--[', file=fd)
            print(val_max)
            print('--]', file=fd)
            with open(os.path.join(path, 'code', 'cell.lua'), 'r') as cell:
                fd.write(cell.read())

        live_path = os.path.join(self.workdir, 'live')
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        p = random.choice(os.listdir(live_path))
        mpath = os.path.join(live_path, p)
        with open(os.path.join(mpath, 'cell.lua')) as fd:
            try:
                data = float(fd.readline().split(' ')[1].strip())
            except ValueError:
                data = -0.01
        min_live = data
        min_path = mpath
        if min_live <= max_result:
            print('mv %s %s' % (min_path, os.path.join(self.workdir, 'dead')))
            os.system('mv %s %s' % (min_path, os.path.join(self.workdir, 'dead')))
            print('rm -rf %s' % os.path.join(path, 'code'))
            os.system('rm -rf %s' % os.path.join(path, 'code'))
            print('mv %s %s' % (path, live_path))
            os.system('mv %s %s' % (path, live_path))
        else:
            print('rm -rf %s' % path)
            os.system('rm -rf %s' % path)

        fcntl.flock(self.lock, fcntl.LOCK_UN)

    def mutate(self, net):
        def randomLayer():
            layers = ((linearLayer, 1),
                      (reluLayer, 2),
                      (dropoutLayer, 2),
                      (sigmoidLayer, 1),
                      (tanhLayer, 1),
                      (caddLayer, 2),
                      (cmulLayer, 1.5))
            count = sum([i[1] for i in layers])
            rv = random.random() * count
            current_sum = 0
            for i in layers:
                current_sum = current_sum + i[1]
                if current_sum > rv:
                    return i[0]()

        mutate_weight = [3, 5, 10, 13]

        made = 0
        changes = 1 + int(random.expovariate(1))
        while made < changes:
            try:
                rv = random.randrange(mutate_weight[-1])
                if rv < mutate_weight[0]:
                    # add
                    e = random.choice(net.G.edges())
                    net.addNodeOnEdge(randomLayer(), e)
                elif mutate_weight[0] <= rv < mutate_weight[1]:
                    # replace
                    node = net.randomNode(None)
                    if not net.replaceNode(node, randomLayer()):
                        print('mutate fail')
                        made = made - 1
                elif mutate_weight[1] <= rv < mutate_weight[2]:
                    # change connect
                    node = net.randomNode(None, withOutput=True)
                    if not net.changeNodeConnect(node):
                        print('mutate fail')
                        made = made - 1
                else:
                    # remove
                    node = net.randomNode(None)
                    if not net.removeNode(node):
                        print('mutate fail')
                        made = made - 1
                made = made + 1
            except NotPossibleError:
                print('not possible mutate random')
                continue
        net.simplify()
        return net


class rpcFileSystemRuler:
    def __init__(self, workdir, n_live, n_hidden):
        self.workdir = workdir
        self.n_hidden = n_hidden

        self.pool = Pool(16)
        dirs = os.listdir(workdir)
        for i in ['live', 'dead', 'born']:
            if i not in dirs:
                os.mkdir(os.path.join(self.workdir, i))
        live_path = os.path.join(self.workdir, 'live')
        self.lock = max([0]+[int(i) for i in os.listdir(live_path)])

        # init first
        dirs = os.listdir(live_path)
        for _ in range(n_live - len(dirs)):
            net = network(n_hidden)
            nid = self.newId()
            npath = os.path.join(self.workdir, 'live', nid)
            os.mkdir(npath)
            with open(os.path.join(npath, 'cell.pickle'), 'wb') as fd:
                pickle.dump(net, fd)
            net.writeLua(os.path.join(npath, 'cell.lua'))

    def newId(self):
        self.lock = self.lock + 1
        return str(self.lock)

    def born(self):
        live_path = os.path.join(self.workdir, 'live')
        cands = os.listdir(os.path.join(live_path))
        cand = random.choice(cands)
        print(colored("="*30, 'blue'))
        print('born net from ' + cand)
        pickle_path = os.path.join(live_path, cand, 'cell.pickle')
        with open(pickle_path, 'rb') as fd:
            net = pickle.load(fd)
        net = self.mutate(net)
        nid = self.newId()
        print('new id ', colored(nid, 'cyan'))
        npath = os.path.join(self.workdir, 'born', nid)
        os.mkdir(npath)
        with open(os.path.join(npath, 'cell.pickle'), 'wb') as fd:
            pickle.dump(net, fd)
        lua_code = net.getLua()
        max_iters = int(int(nid) + 2500)

        return {'id': nid, 'lua': lua_code, 'task': 'neuraltalk2-ERNN',
                'args': {'num_rnn': self.n_hidden, 'max_iters': max_iters}}

    def fight(self, nid, metric):
        print(colored("="*30, 'yellow'))
        print('rpc ret:', nid)
        print(colored(metric, 'magenta'))
        path = os.path.join(self.workdir, 'born', nid)
        with open(os.path.join(path, 'cell.pickle'), 'rb') as fd:
            net = pickle.load(fd)
        # plt.clf()
        # nx.draw(net.G)
        # plt.savefig(os.path.join(path, 'cell.png'))
        max_result = metric['CIDEr']
        with open(os.path.join(path, 'cell.lua'), 'w') as fd:
            print('-- %f' % max_result, file=fd)
            print('--[', file=fd)
            print(metric, file=fd)
            print('--]', file=fd)
            fd.write(net.getLua())
        live_path = os.path.join(self.workdir, 'live')

        id_path = os.listdir(live_path)
        scores = self.pool.imap_unordered(partial(get_score, live_path), id_path)
        scores = sorted(scores, key=lambda x: x[1])
        bad_part = scores[:int(len(scores)/2)]
        p = random.choice(bad_part)
        print('fight with: ', p)
        min_path = os.path.join(live_path, p[0])
        min_live = p[1]
        if min_live < max_result:
            # print('mv %s %s' % (min_path, os.path.join(self.workdir, 'dead')))
            print(colored('win', 'green'))
            os.system('mv %s %s' %
                      (min_path, os.path.join(self.workdir, 'dead')))
            # print('mv %s %s' % (path, live_path))
            os.system('mv %s %s' % (path, live_path))
        else:
            print(colored('lose', 'red'))
            # print('rm -rf %s' % path)
            os.system('rm -rf %s' % path)
        return True

    def mutate(self, net):
        def randomLayer():
            layers = ((linearLayer, 1),
                      (reluLayer, 2),
                      (dropoutLayer, 2),
                      (sigmoidLayer, 1),
                      (tanhLayer, 1),
                      (caddLayer, 2),
                      (cmulLayer, 1.5))
            count = sum([i[1] for i in layers])
            rv = random.random() * count
            current_sum = 0
            for i in layers:
                current_sum = current_sum + i[1]
                if current_sum > rv:
                    return i[0]()

        mutate_weight = [5, 7, 12, 13]

        made = 0
        changes = 1 + int(random.expovariate(1))
        while made < changes:
            try:
                rv = random.randrange(mutate_weight[-1])
                if rv < mutate_weight[0]:
                    # add
                    e = random.choice(net.G.edges())
                    net.addNodeOnEdge(randomLayer(), e)
                elif mutate_weight[0] <= rv < mutate_weight[1]:
                    # replace
                    node = net.randomNode(None)
                    if not net.replaceNode(node, randomLayer()):
                        # print('mutate fail')
                        made = made - 1
                elif mutate_weight[1] <= rv < mutate_weight[2]:
                    # change connect
                    node = net.randomNode(None, withOutput=True)
                    if not net.changeNodeConnect(node):
                        # print('mutate fail')
                        made = made - 1
                else:
                    # remove
                    node = net.randomNode(None)
                    if not net.removeNode(node):
                        # print('mutate fail')
                        made = made - 1
                made = made + 1
            except NotPossibleError:
                # print('not possible mutate random')
                continue
        net.simplify()
        return net

    def listen(self, port):
        print('listening at port:', port)
        server = SimpleXMLRPCServer(("0.0.0.0", port))
        server.register_function(self.born, 'born')
        server.register_function(self.fight, 'fight')

        server.serve_forever()
