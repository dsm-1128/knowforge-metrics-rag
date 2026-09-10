"""检查 Git 待发布文件；只输出文件名与行号，不输出敏感值。"""
import re
import subprocess
from pathlib import Path

from dotenv import dotenv_values


def main():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    names = set(filter(None, git('ls-files', '--cached', '--others', '--exclude-standard', '-z').decode().split('\0')))
    values = dotenv_values(root / '.env')
    secrets = [v for k, v in values.items() if v and len(v) >= 8 and any(t in k.upper() for t in ('KEY', 'PASSWORD', 'TOKEN', 'SECRET'))]
    pattern = re.compile(r'lsv2_[A-Za-z0-9_]{15,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY')
    errors = []
    def inspect(name, data, source):
        if len(data) > 25 * 1024 * 1024:
            errors.append(f'{source}: large file: {name}')
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            return
        for number, line in enumerate(text.splitlines(), 1):
            if any(value in line for value in secrets) or pattern.search(line):
                errors.append(f'{source}: possible credential: {name}:{number}')
    for name in sorted(names):
        path = root / name
        if path.is_file():
            inspect(name, path.read_bytes(), 'working tree')
    staged = list(filter(None, git('ls-files', '--cached', '-z').decode().split('\0')))
    for name in staged:
        inspect(name, git('show', ':' + name), 'index')
    ignored_tracked = git('ls-files', '--cached', '--ignored', '--exclude-standard', '-z').decode().split('\0')
    errors.extend('Tracked ignored file: ' + name for name in ignored_tracked if name)
    for name in ('.env.example', '.env.local.example'):
        for key, value in dotenv_values(root / name).items():
            if any(t in key.upper() for t in ('API_KEY', 'PASSWORD', 'TOKEN', 'SECRET')) and value not in (None, '', 'CHANGE_ME'):
                errors.append(f'Template credential must be empty or CHANGE_ME: {name}:{key}')
    if errors:
        print('\n'.join(errors))
        return 1
    print(f'PASS: {len(names)} candidate files checked, including index contents. This scan is not a guarantee that all confidential business data is detected.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
