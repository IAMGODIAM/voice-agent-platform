import subprocess, sys
cmd = sys.argv[1]
result = subprocess.run(
    ['ssh', 'windows-lan', f'powershell -NoProfile -Command "{cmd}"'],
    capture_output=True, text=True, timeout=15
)
print(result.stdout)
if result.stderr:
    print(result.stderr, file=sys.stderr)
sys.exit(result.returncode)
