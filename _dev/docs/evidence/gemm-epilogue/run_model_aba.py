"""Run a fresh, serial, unprofiled A1/B/A2. Only owns its named containers.

Run detached on the GPU host so a local SSH disconnect does not interrupt the
sequence. Failure stops the sequence and cleans up only its current arm.
Embedding and the original non-embedding services are verified, never changed.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--library-source', required=True)
    p.add_argument('--library-sha256', required=True)
    args = p.parse_args()
    root = args.root
    output = root / args.label
    output.mkdir(parents=True, exist_ok=False)
    scripts = root / 'work/gemm-epilogue'
    events = []

    def event(**data):
        data['at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        events.append(data)
        (output / 'sequence.json').write_text(json.dumps(events, indent=2) + '\n')
        print(json.dumps(data), flush=True)

    def verify(path):
        subprocess.run(['python3', str(scripts / 'verify_service_state.py'),
            '--initial-state', str(root / 'initial-state.json'),
            '--output', str(path)], check=True)

    for arm in ('a1', 'b', 'a2'):
        name = 'codex-fp8-epi-' + args.label + '-' + arm
        destination = output / arm
        mode = 'candidate' if arm == 'b' else 'baseline'
        verify(output / (arm + '-services-before.json'))
        event(arm=arm, phase='launch', name=name, mode=mode)
        command = ['python3', str(scripts / 'launch_model_arm.py'),
            '--root', str(root), '--name', name,
            '--label', str(Path(args.label) / arm), '--mode', mode]
        if mode == 'candidate':
            command += ['--library-source', args.library_source,
                        '--library-sha256', args.library_sha256]
        try:
            subprocess.run(command, check=True)
            subprocess.run(['python3', str(scripts / 'run_api_arm.py'),
                '--name', name, '--output', str(destination),
                '--tag', 'gemm-epilogue-fixed-aba'], check=True)
            event(arm=arm, phase='measurements_completed')
        finally:
            # Fixed fresh name created for this sequence; never targets originals.
            state = subprocess.run(['docker', 'inspect', name],
                                   capture_output=True, text=True)
            if state.returncode == 0:
                destination.mkdir(parents=True, exist_ok=True)
                with (destination / 'server.log').open('w') as stream:
                    subprocess.run(['docker', 'logs', name], stdout=stream,
                                   stderr=subprocess.STDOUT, check=True)
                subprocess.run(['docker', 'stop', '--time', '60', name], check=True)
            verify(output / (arm + '-services-after.json'))
        event(arm=arm, phase='stopped_owned_arm')
    event(phase='completed')


if __name__ == '__main__':
    main()
