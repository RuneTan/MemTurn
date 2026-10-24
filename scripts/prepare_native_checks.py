"""Prepare official text-environment assets and explicit native interface-check manifests."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import shutil
import stat
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    data = ROOT / '.cache' / 'alfworld'
    os.environ['ALFWORLD_DATA'] = str(data)
    output = ROOT / 'runs' / 'native_setup_20261024'
    output.mkdir(parents=True,exist_ok=True)
    python = ROOT / '.venv-native-linux/bin/python'
    worker = ROOT / 'scripts/native_worker.py'
    command = ['wsl','-d',args.wsl_distro,'--exec',str(python),str(worker)]
    if args.kind == 'scienceworld':
        runtime = json.loads((ROOT/'.cache/native/java17/source.json').read_text(encoding='utf-8'))
        os.environ['JAVA_HOME'] = runtime['java_home']
        os.environ['PATH'] = runtime['java_home']+'/bin:'+os.environ.get('PATH','')
        from scienceworld import ScienceWorldEnv
        env = ScienceWorldEnv(envStepLimit=12)
        try:
            names = env.get_task_names()
            name = names[0]
            env.load(name,0,'')
            variations = env.get_variations_train()
            tasks = [dict(id=f'scienceworld_interface_{v}',family='scienceworld',version=name,
                          max_steps=12,metadata={'task_name':name,'variation':v,'evaluation_only':True})
                     for v in variations[:2]]
        finally:
            env.close()
        result = {'environment':{'kind':'scienceworld','worker_command':command,
                  'java_home':runtime['java_home'],'simplifications':''},
                  'tasks':tasks,'actions':['look around'],
                  'native_version':importlib.metadata.version('scienceworld')}
    else:
        temporary = ROOT/'.cache/native/downloads'
        temporary.mkdir(parents=True,exist_ok=True)
        tempfile.tempdir = str(temporary)
        downloader = runpy.run_path(str(python.parent/'alfworld-download'),run_name='_native_download_functions')
        origins=[]
        for key in ('JSON_FILES_URL','PDDL_FILES_URL','TW_PDDL_FILES_URL'):
            archive = Path(downloader['download'](downloader[key],dst=str(data)))
            with zipfile.ZipFile(archive) as zipped:
                for item in zipped.infolist():
                    if not (data/item.filename).resolve().is_relative_to(data.resolve()):
                        raise ValueError('native assets contain an unsafe path')
                    if stat.S_ISLNK(item.external_attr >> 16):
                        raise ValueError('native assets contain a link')
                zipped.extractall(data)
            origins.append({'url':downloader[key],'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()})
        import alfworld.info as info
        logic=data/'logic'; logic.mkdir(exist_ok=True)
        shutil.copy2(info.ALFRED_PDDL_PATH,logic/'alfred.pddl')
        shutil.copy2(info.ALFRED_TWL2_PATH,logic/'alfred.twl2')
        games=sorted(str(p) for p in data.rglob('game.tw-pddl') if '/train/' in str(p))
        if not games: raise ValueError('official download contains no training games')
        config={'dataset':{'data_path':str(data/'json_2.1.1/train'),
                 'eval_id_data_path':str(data/'json_2.1.1/valid_seen'),
                 'eval_ood_data_path':str(data/'json_2.1.1/valid_unseen'),
                 'num_train_games':-1,'num_eval_games':-1},
                'env':{'type':'AlfredTWEnv','task_types':[1,2,3,4,5,6],
                       'goal_desc_human_anns_prob':0,'domain_randomization':False,'expert_type':'handcoded'},
                'general':{'training_method':'dqn'},
                'rl':{'training':{'max_nb_steps_per_episode':12}}}
        # Discover the official dataset directory instead of assuming an archive layout.
        config['dataset']['data_path']=str(Path(games[0]).parents[2])
        import yaml
        path=output/'alfworld.yaml'
        path.write_text(yaml.safe_dump(config),encoding='utf-8')
        result={'environment':{'kind':'alfworld','worker_command':command,
                 'config_path':str(path),'split':'train','data_path':str(data)},
                'tasks':[dict(id=f'alfworld_interface_{i}',family='alfworld',version='native',
                              seed=i,max_steps=12,metadata={'gamefile':p,'evaluation_only':True})
                         for i,p in enumerate(games[:2])],
                'actions':['look'],'native_version':importlib.metadata.version('alfworld'),'origins':origins}
    (output/f'{args.kind}.check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'kind':args.kind,'tasks':len(result['tasks']),'output':str(output)},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--kind',choices=('scienceworld','alfworld'),required=True)
    parser.add_argument('--wsl-distro',default='SMEFTsim')
    main(parser.parse_args())
