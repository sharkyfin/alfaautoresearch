"""Use real public summaries with a local fake transport; no network or API spending."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from atlas.task import load
from atlas.contract import read_inputs
from atlas.agent import QwenClient,researcher_context,compact_context
records=[]
for kind in ['regression','classification']:
 data=load(read_inputs(ROOT/'artifacts/data'/kind/'task.md'))
 history=[json.loads(s) for s in (ROOT/'runs/public-final'/f'{kind}_42'/'experiments.jsonl').read_text().splitlines()]
 champion=next(r for r in reversed(history) if r.get('accepted'))
 context=researcher_context(data,history,champion,1000)
 received=[]
 def fake(body,timeout):
  received.append(body)
  return {'choices':[{'message':{'content':'{"model_family":"hist","statement":"local mock hypothesis"}'}}],'usage':{'total_tokens':1000}}
 client=QwenClient('disabled',transport=fake);client.mode='live';client.configured=True;client.open=False;client.model='mock-only'
 action=client.propose(context)
 assert action and len(received)==1
 payload=received[0]['messages'][1]['content'];decoded=json.loads(payload)
 assert len(payload.encode())<=14000 and decoded['objective']['metric']==data.spec.metric.name
 assert decoded['champion']['recipe'] and decoded['untrusted_data']['description']
 records.append({'task':kind,'original_bytes':len(json.dumps(context,ensure_ascii=False).encode()),'sent_mock_bytes':len(payload.encode()),'mock_transport_calls':len(received),'network_calls':0,'status':'passed'})
(ROOT/'reports/public-research-context.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
