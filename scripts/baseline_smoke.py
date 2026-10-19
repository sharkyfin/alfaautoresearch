"""Execute the unmodified organizer tool function without its unavailable MCP imports."""
import ast, json, os, subprocess, sys, tempfile
from pathlib import Path
source=Path('artifacts/official/baseline.py').read_text()
tree=ast.parse(source)
node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run_python')
node.decorator_list=[]
module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
ns=dict(subprocess=subprocess,sys=sys,os=os,json=json)
exec(compile(module,'official_tool','exec'),ns)
with tempfile.TemporaryDirectory() as d:
    before=Path.cwd();os.chdir(d)
    try:
        result=json.loads(ns['run_python']("from pathlib import Path; Path('submission.csv').write_text('id,y\\n1,0\\n'); print('ok')"))
        assert result['exit_code']==0 and Path('submission.csv').read_text()=='id,y\n1,0\n'
        assert 'OPENAI_API_KEY' not in json.loads(ns['run_python']('import os,json;print(json.dumps(dict(os.environ)))'))['stdout']
        print(json.dumps({'official_tool_smoke':'passed','cwd':d,'timeout_sec':1800,'env_forwarded':False}))
    finally: os.chdir(before)
