"""Local coding-turn probes. Transport success is not a numerical quality gate."""
import json
import time
import urllib.error
import urllib.request

CODE = '''// Review this patch and suggest a minimal correction with tests.
template<class T> class Buffer {
    std::vector<T> values;
public:
    void append(const T& value) { values.push_back(value); }
    const T& at(size_t index) const { return values.at(index); }
    size_t size() const { return values.size(); }
};
TEST(Buffer, PreservesInsertionOrder) {
    Buffer<int> items;
    items.append(17);
    items.append(23);
    EXPECT_EQ(items.at(0), 17);
    EXPECT_EQ(items.at(1), 23);
}
// Tool result: build completed; two tests passed. Check edge cases next.
'''


def request(base, path, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=timeout)


def completion(base, prompt, cache, predict=4):
    start = time.perf_counter()
    first = None
    content = []
    tokens = []
    final = None
    with request(base, '/completion', dict(prompt=prompt, n_predict=predict,
                 temperature=0, seed=42, cache_prompt=cache, id_slot=0,
                 return_tokens=True, stream=True)) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            if line[6:].strip() == b'[DONE]':
                break
            event = json.loads(line[6:])
            if 'error' in event:
                raise RuntimeError(event['error'])
            if first is None and (event.get('content') or event.get('tokens')):
                first = time.perf_counter()
            content.append(event.get('content', ''))
            tokens.extend(event.get('tokens', []))
            if event.get('stop'):
                final = event
    if final is None or 'timings' not in final:
        raise RuntimeError('completion missing final timings')
    elapsed = time.perf_counter() - start
    return dict(timings=final['timings'], total_ms=elapsed*1000,
                ttft_ms=None if first is None else (first-start)*1000,
                content=''.join(content), tokens=tokens)


def _wait_ready(base, stop_event=None):
    deadline = time.monotonic() + 300
    while True:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError('server probe cancelled')
        try:
            with request(base, '/health', timeout=1) as response:
                if json.load(response).get('status') == 'ok':
                    break
        except (OSError, urllib.error.URLError):
            if time.monotonic() > deadline:
                raise RuntimeError('server did not become healthy')
        time.sleep(1)


def _fixture(base):
    with request(base, '/tokenize', dict(content=CODE*160, add_special=False)) as response:
        return json.load(response)['tokens']


def prewarm_server(port, sizes, ctx, stop_event=None):
    base = f'http://127.0.0.1:{port}'
    _wait_ready(base, stop_event)
    source = _fixture(base)
    for size in sizes:
        if size > len(source):
            raise RuntimeError(f'prewarm size {size} exceeds coding fixture')
        completion(base, source[:size], False, predict=4)


def run_probe(port, suite, target, ctx, stop_event=None, prewarm=()):
    base = f'http://127.0.0.1:{port}'
    _wait_ready(base, stop_event)
    source = _fixture(base)
    if suite == 'smoke':
        sizes = [128, 513, 1025, 128]
    elif suite == 'short':
        sizes = [64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024]
    else:
        sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8128]
    if len(source) < max(sizes) + 1024:
        raise RuntimeError('coding token fixture too short')
    for size in prewarm:
        if size > len(source):
            raise RuntimeError(f'prewarm size {size} exceeds coding fixture')
        # Build and retain the graph signature, but do not seed the prompt
        # cache. The measured request starts from a fresh slot state.
        completion(base, source[:size], False, predict=4)
    with open(str(target) + '.http.jsonl', 'x') as out:
        def record(row):
            out.write(json.dumps(row) + '\n')
            out.flush()
            print(json.dumps({k: v for k, v in row.items() if k not in ['content', 'tokens']}), flush=True)
        for size in sizes:
            result = completion(base, source[:size], False)
            record(dict(kind='fresh', requested_tokens=size, **result))
        prefix_size = min(2048 if suite == 'smoke' else 8192, ctx//2)
        prompt = source[:prefix_size]
        seed = completion(base, prompt, False, predict=1)
        record(dict(kind='cache-seed', requested_tokens=len(prompt), **seed))
        prompt += seed['tokens']
        suffixes = [64, 129] if suite == 'smoke' else [64, 128, 256, 512, 1024]
        if suite == 'soak':
            suffixes += [65]*20
        for suffix in suffixes:
            prompt += source[prefix_size:prefix_size+suffix]
            result = completion(base, prompt, True, predict=1)
            record(dict(kind='cached', suffix_tokens=suffix, requested_tokens=len(prompt), **result))
            if result['timings'].get('cache_n', 0) <= 0:
                raise RuntimeError('cached coding turn did not reuse any prefix')
            prompt += result['tokens']
        chat = dict(messages=[dict(role='user', content='Write a Python function that adds two integers.')],
                    temperature=0, max_tokens=64, stream=False,
                    chat_template_kwargs=dict(enable_thinking=False))
        with request(base, '/v1/chat/completions', chat) as response:
            result = json.load(response)
        if not result.get('choices') or not result['choices'][0]['message'].get('content'):
            raise RuntimeError('chat API missing final answer content')
        record(dict(kind='chat-api', response=result))
