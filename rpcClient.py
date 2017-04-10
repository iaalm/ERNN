#! /usr/bin/python3

import os
import json
import argparse
import xmlrpc.client

parser = argparse.ArgumentParser()
parser.add_argument("--url",
                    default='http://10.60.150.247:8080',
                    help="model dirs")
parser.add_argument('--gpuid', default=0,  help="gpuid")
args = parser.parse_args()


def rpcWorker(url, gpuid):
    s = xmlrpc.client.ServerProxy(url)
    if not os.path.exists('work_%d' % gpuid):
        os.system('cp -lr neuraltalk2-ERNN work_%d' % gpuid)
    os.chdir('work_%d' % gpuid)
    flag = True
    while(flag):
        try:
            ret = s.born()
        except:
            continue
        try:
            nid, lua_code = ret['id'], ret['lua']
            print('nid', nid)
            with open('cell.lua', 'w') as fd:
                fd.write(lua_code)

            os.system('CUDA_VISIBLE_DEVICES=%d th train.lua' % gpuid)

            with open('model_.json') as fd:
                data = json.load(fd)
            val_max = {}
            val_data = data["val_lang_stats_history"]
            val_data = sorted(val_data.items(), key=lambda t: int(t[0]))

            max_result = 0
            for k, v in val_data:
                if v['CIDEr'] > max_result:
                    max_result = v['CIDEr']
                    for metric in v:
                        val_max[metric] = v[metric]
                    val_max['pos_max'] = k
        except KeyboardInterrupt:
            if flag:
                print('return next loop')
                flag = False
            else:
                print('return now')
                break
        finally:
            s.fight(nid, {'CIDEr': -1})

if __name__ == '__main__':
    rpcWorker(args.url, int(args.gpuid))
