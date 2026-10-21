"""Detached CPU-only image build and installed bootstrap verification."""
import datetime
import json
import re
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/llm/.local/share/qwen-fp8-release-20261015')
TAG = 'pro6000d/qwen36-fp8:20261015-r1'
OUT = ROOT / 'build-r1'


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2) + '\n')


def run():
    state = {'state': 'running', 'tag': TAG, 'gpu_used': False}
    dump('status.json', state)
    try:
        prior = subprocess.run(['docker', 'image', 'inspect', TAG], capture_output=True)
        if prior.returncode == 0:
            raise RuntimeError('refusing to replace existing release tag')
        command = ['docker', 'build', '--network=none', '--pull=false', '--tag', TAG,
                   '--iidfile', str(OUT / 'image-id.txt'), str(ROOT / 'context-r1')]
        dump('build-command.json', command)
        with (OUT / 'build.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
        image_id = (OUT / 'image-id.txt').read_text().strip()
        image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image_id]))[0]
        dump('image-inspect.json', image)
        assert image['Config']['Entrypoint'] == ['vllm', 'serve']
        assert image['Architecture'] == 'amd64' and image['Os'] == 'linux'
        results = []
        for case, env, tail, expected in [
            ('matrix', [], ['/opt/pro6000d/verify_install.py', '--matrix'], 0),
            ('invalid', ['PRO6000D_RELEASE_V2=2'], ['-c', 'print("must not reach here")'], 78),
        ]:
            cidfile = OUT / (case + '-container-id.txt')
            command = ['docker', 'run', '--rm', '--name', 'codex-fp8-release-build-' + case,
                       '--cidfile', str(cidfile),
                       '--network=none', '--entrypoint', 'python3']
            for item in env:
                command += ['--env', item]
            command += [image_id, '-I', *tail]
            try:
                with (OUT / (case + '.stdout')).open('w') as out, (OUT / (case + '.stderr')).open('w') as err:
                    result = subprocess.run(command, stdout=out, stderr=err, timeout=90)
            finally:
                # --rm only applies after the container actually exits; a
                # timed-out Docker CLI can otherwise leave it running.
                cid = cidfile.read_text().strip() if cidfile.exists() else ''
                if re.fullmatch(r'[a-f0-9]{64}', cid):
                    found = subprocess.run(['docker', 'inspect', cid], capture_output=True, timeout=20)
                    if found.returncode == 0:
                        subprocess.run(['docker', 'stop', '--timeout', '10', cid], capture_output=True, timeout=30)
                        subprocess.run(['docker', 'rm', cid], capture_output=True, timeout=30)
            results.append({'case': case, 'command': command, 'exit_code': result.returncode,
                            'expected_exit_code': expected, 'passed': result.returncode == expected})
            dump('cpu-checks.json', results)
            if result.returncode != expected:
                raise RuntimeError('CPU image check failed: ' + case)
        state.update(state='completed', image_id=image_id)
    except Exception as exc:
        state.update(state='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        state['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        dump('status.json', state)


if __name__ == '__main__':
    if '--run' in sys.argv:
        run()
    else:
        OUT.mkdir(exist_ok=False)
        with (OUT / 'detached.log').open('w') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--run'],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'pid': process.pid, 'output': str(OUT)}))
