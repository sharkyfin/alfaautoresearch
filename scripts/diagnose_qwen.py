"""Measure organizer API connection phases without logging credentials or bodies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import httpx
    from scripts.run_organizer import load_environment
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-file', type=Path, default=ROOT/'.env.organizer.local')
    parser.add_argument('--read-timeout', type=float, default=40)
    parser.add_argument('--context-run', type=Path, help='Use a completed agent run and its train-only researcher context')
    parser.add_argument('--output-tokens', type=int, default=64)
    parser.add_argument('--reasoning', choices=['none', 'minimal', 'low'])
    parser.add_argument('--report', type=Path, default=ROOT/'reports/qwen-connection.json')
    args = parser.parse_args()
    if not 1 <= args.read_timeout <= 50:
        parser.error('read-timeout must be between 1 and 50 seconds')
    if not 64 <= args.output_tokens <= 2400:
        parser.error('output-tokens must be between 64 and 2400')
    settings = load_environment(args.key_file)
    endpoint = settings['OPENAI_BASE_URL'].rstrip('/')+'/chat/completions'
    model = settings.get('OPENAI_MODEL') or 'qwen/qwen3.8-27b'
    started = time.monotonic()
    events = []
    def trace(name, info):
        event = {'phase': name, 'seconds': round(time.monotonic()-started, 4)}
        if 'exception' in info:
            event['exception_class'] = type(info['exception']).__name__
        events.append(event)
    report = {'endpoint_host': urlparse(endpoint).hostname, 'requested_model': model,
              'connect_timeout': 5, 'read_timeout': args.read_timeout, 'events': events}
    messages = [{'role': 'user', 'content': 'Return only the JSON object {"ok":true}.'}]
    if args.context_run:
        from atlas.agent import SYSTEM, compact_context, researcher_context
        from atlas.contract import read_inputs
        from atlas.task import load
        run = args.context_run.resolve()
        data = load(read_inputs(run.parent/'task/task.md'))
        summary = json.loads((run/'summary.json').read_text())
        data.spec.validation_plan = summary['validation']
        history = [json.loads(row) for row in (run/'experiments.jsonl').read_text().splitlines()]
        champion = json.loads((run/'champion.json').read_text())
        context = compact_context(researcher_context(data, history, champion, 300))
        messages = [{'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': json.dumps(context, ensure_ascii=False)}]
        report['context_runtime_code_sha256'] = summary['runtime_code_sha256']
    report['request_characters'] = sum(len(message['content']) for message in messages)
    report['max_output_tokens'] = args.output_tokens
    body = {'model': model, 'messages': messages,
            'max_tokens': args.output_tokens, 'temperature': 0, 'response_format': {'type': 'json_object'}}
    if args.reasoning:
        body['reasoning'] = {'effort': args.reasoning, 'exclude': True}
        report['reasoning_effort'] = args.reasoning
    try:
        timeout = httpx.Timeout(args.read_timeout, connect=5, write=5, pool=5)
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            with client.stream('POST', endpoint, json=body,
                               headers={'Authorization': 'Bearer '+settings['OPENAI_API_KEY']},
                               extensions={'trace': trace}) as response:
                report['http_status'] = response.status_code
                raw = bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > 65536:
                        raise ValueError('Oversized response')
                response.raise_for_status()
        payload = json.loads(raw)
        report.update(status='ok', response_model=payload.get('model'), usage=payload.get('usage'),
                      finish_reason=payload.get('choices', [{}])[0].get('finish_reason'))
        content = payload.get('choices', [{}])[0].get('message', {}).get('content')
        report['response_characters'] = len(content) if isinstance(content, str) else 0
        if isinstance(content, str):
            try:
                action = json.loads(content)
                report['json_action'] = isinstance(action, dict)
                if isinstance(action, dict):
                    report['model_family'] = action.get('model_family')
                    report['new_feature_expressions'] = len(action.get('feature_expressions', []))
            except ValueError:
                report['json_action'] = False
    except (httpx.HTTPError, ValueError, OSError) as error:
        report.update(status='failed', exception_class=type(error).__name__)
    report['elapsed_seconds'] = round(time.monotonic()-started, 4)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
