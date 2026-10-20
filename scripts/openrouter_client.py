"""Explicit development-only OpenRouter client; never imported by solution.py."""
from __future__ import annotations
import copy,json,os,urllib.request
from pathlib import Path
from atlas.agent import QwenClient

MODEL='minimax/minimax-m3.1-flash:free'
BASE_URL='https://openrouter.ai/api/v1'


def load_key(path: Path):
    """Read only the dedicated key; no shell evaluation or organizer-key fallback."""
    key=os.environ.get('OPENROUTER_API_KEY','').strip()
    if key:return key
    if not path.is_file():raise ValueError('Set OPENROUTER_API_KEY or provide --key-file')
    matches=[]
    for line in path.read_text().splitlines():
        name,sep,value=line.strip().removeprefix('export ').partition('=')
        if sep and name.strip()=='OPENROUTER_API_KEY':matches.append(value.strip().strip('\"\''))
    if len(matches)!=1 or not matches[0]:raise ValueError('Key file must contain one nonempty OPENROUTER_API_KEY')
    return matches[0]


class OpenRouterClient(QwenClient):
    provider='openrouter-free'
    origin='openrouter'

    def __init__(self,key,max_calls=4,max_tokens=24000,transport=None):
        if not key:raise ValueError('An OpenRouter key is required')
        if not 1<=max_calls<=8:raise ValueError('Free research allows between 1 and 8 calls per run')
        self._wire_transport=transport
        super().__init__('disabled',max_calls=max_calls,max_tokens=max_tokens,transport=self._send,output_tokens=4096)
        self.mode='live';self.endpoint=BASE_URL;self.model=MODEL;self.key=key;self.configured=True
        self.events.append({'status':'researcher_backend','provider':self.provider,'model':self.model,'paid_fallback':False})

    def _send(self,body,timeout):
        # Fixed model + zero per-token prices prevent a silent paid fallback.
        if self.model!=MODEL or not self.model.endswith(':free'):raise ValueError('Only the pinned free model is enabled')
        body=copy.deepcopy(body);body['model']=MODEL
        body['provider']={'max_price':{'prompt':0,'completion':0}}
        # MiniMax rejects reasoning.enabled=false; reserve space for mandatory thinking.
        body['reasoning']={'effort':'low','exclude':True}
        if self._wire_transport:result=self._wire_transport(body,timeout)
        else:
            request=urllib.request.Request(BASE_URL+'/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json','X-Title':'ATLAS development'},method='POST')
            with urllib.request.urlopen(request,timeout=timeout) as response:
                raw=response.read(65537)
            if len(raw)>65536:raise ValueError('Oversized API response')
            result=json.loads(raw)
        cost=result.get('usage',{}).get('cost')
        self.events.append({'status':'api_usage','provider':self.provider,'model':result.get('model'),'cost_usd':cost})
        if cost is not None and float(cost)>0:raise ValueError('Unexpected nonzero price on the free endpoint')
        return result
