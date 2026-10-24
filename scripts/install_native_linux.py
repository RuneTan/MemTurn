"""Install native simulators and a verified portable Java runtime inside this checkout."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from urllib.request import Request, urlopen
import venv


def main():
    if sys.platform != 'linux':
        raise SystemExit('Run with Linux Python (including WSL); no global packages are changed.')
    root = Path(__file__).resolve().parents[1]
    cache = root / '.cache' / 'native'
    cache.mkdir(parents=True, exist_ok=True)
    environment = root / '.venv-native-linux'
    if not (environment / 'bin/python').exists():
        venv.EnvBuilder(with_pip=True, symlinks=False).create(environment)
    python = environment / 'bin/python'
    pip_env = dict(os.environ, PATH=str(environment / 'bin') + ':' + os.environ.get('PATH', ''))
    subprocess.run([str(python), '-m', 'pip', 'install', '--cache-dir', str(cache / 'pip'),
                    'scienceworld==1.2.3', 'alfworld==0.4.2', 'PyYAML==6.0.3'], env=pip_env, check=True)
    runtime = cache / 'java17'
    manifest = runtime / 'source.json'
    if not manifest.exists():
        runtime.mkdir(exist_ok=True)
        api = ('https://api.adoptium.net/v3/assets/latest/17/hotspot?architecture=x64&'
               'image_type=jre&os=linux&vendor=eclipse')
        with urlopen(Request(api,headers={'User-Agent':'MemTurn-native-setup/0.1'}), timeout=60) as response:
            asset = json.load(response)[0]
        package = asset['binary']['package']
        archive = runtime / 'runtime.tar.gz'
        digest = hashlib.sha256()
        with urlopen(Request(package['link'],headers={'User-Agent':'MemTurn-native-setup/0.1'}), timeout=60) as response, archive.open('wb') as output:
            while block := response.read(1024 * 1024):
                output.write(block); digest.update(block)
        if digest.hexdigest() != package['checksum']:
            raise RuntimeError('Java download checksum mismatch')
        with tarfile.open(archive) as bundle:
            members = bundle.getmembers()
            for member in members:
                target = (runtime / member.name).resolve()
                if not target.is_relative_to(runtime.resolve()):
                    raise ValueError('runtime archive contains an unsafe path')
                if member.issym() or member.islnk():
                    base = target.parent if member.issym() else runtime
                    if not (base / member.linkname).resolve().is_relative_to(runtime.resolve()):
                        raise ValueError('runtime archive contains an unsafe link')
            bundle.extractall(runtime)
        java_home = next(p.parent.parent for p in runtime.glob('*/bin/java'))
        manifest.write_text(json.dumps({'api':api, 'url':package['link'], 'sha256':package['checksum'],
                                       'java_home':str(java_home)}, indent=2), encoding='utf-8')
    runtime_info = json.loads(manifest.read_text(encoding='utf-8'))
    env = dict(os.environ, JAVA_HOME=runtime_info['java_home'],
               ALFWORLD_DATA=str(root / '.cache' / 'alfworld'),
               PATH=runtime_info['java_home'] + '/bin:' + os.environ.get('PATH', ''))
    subprocess.run([str(Path(runtime_info['java_home']) / 'bin/java'), '-version'], env=env, check=True)
    freeze = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], env=env, text=True)
    (cache / 'requirements.resolved.txt').write_text(freeze, encoding='utf-8')
    print(json.dumps({'python':str(python), 'java_home':runtime_info['java_home'],
                      'alfworld_data':env['ALFWORLD_DATA'], 'assets':'ALFWorld games still need alfworld-download'}))


if __name__ == '__main__':
    main()
