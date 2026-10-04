"""Gemini transport for AgentVideo's unchanged Agent/Verifier/Bus interfaces.

This is a provider adapter, not a port of the upstream Metal simulation backend.
No key enters agent messages, the browser, a trace, or a repository config.
"""
import base64
import json
import re
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from .gemini_edit_planner import NoRedirect


class CallBudget:
    def __init__(self, calls=3, output_tokens=8192):
        if type(calls) is not int or not 1 <= calls <= 100:
            raise ValueError('Explicit bounded call count required')
        if type(output_tokens) is not int or not 1 <= output_tokens <= 100000:
            raise ValueError('Explicit bounded output token allocation required')
        self.calls, self.tokens = calls, output_tokens
        self.used = 0
        self.lock = threading.Lock()

    def reserve(self, tokens):
        with self.lock:
            if self.calls < 1 or self.tokens < tokens:
                raise RuntimeError('AgentVideo shared API budget exhausted')
            self.calls -= 1
            self.tokens -= tokens
            self.used += 1


class GeminiLLM:
    def __init__(self, model_id, *, budget, key_loader=None, transport=None, minimum_output_tokens=2048):
        if not re.fullmatch(r'gemini-[A-Za-z0-9.-]{1,80}', model_id):
            raise ValueError('Explicit Gemini model required; no silent Qwen fallback')
        self.model_id, self.budget = model_id, budget
        if type(minimum_output_tokens) is not int or not 1 <= minimum_output_tokens <= 8192:
            raise ValueError('Invalid provider output floor')
        self.minimum_output_tokens=minimum_output_tokens
        self.key_loader, self.transport = key_loader, transport
        self.usage = []
        self._lock = threading.Lock()

    def release(self):
        """No resident local model or GPU allocation to release."""

    def chat(self, messages, max_tokens=400, temp=0.3, prefill=''):
        return self._call(messages, max_tokens, temp, prefill)

    def _call(self, messages, max_tokens, temp, prefill='', image=None):
        if type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
            raise ValueError('Invalid output limit')
        if not 0 <= temp <= 2 or not isinstance(prefill, str) or len(prefill) > 1000:
            raise ValueError('Invalid generation options')
        if not isinstance(messages, list) or not 1 <= len(messages) <= 32:
            raise ValueError('Bounded messages required')
        system, contents = [], []
        total = 0
        for message in messages:
            role, content = message['role'], message['content']
            if role not in ('system', 'user', 'assistant') or not isinstance(content, str):
                raise ValueError('Invalid message')
            total += len(content)
            if role == 'system':system.append(content)
            else:contents.append(dict(role='model' if role == 'assistant' else 'user', parts=[dict(text=content)]))
        if total > 64000 or not contents or contents[-1]['role'] != 'user':
            raise ValueError('Bounded conversation ending in user required')
        # Gemini does not implement MLX token-prefix continuation. Request a complete
        # answer with the prefix instead; upstream parser/verifier still decides.
        if prefill:
            contents[-1]['parts'].append(dict(text='Return the COMPLETE answer beginning with: '+prefill))
        if image is not None:contents[-1]['parts'].append(image)
        # Upstream budgets (e.g. 150 for a script) were tuned for non-thinking Qwen.
        # Reserve the full provider allocation, including reasoning headroom, rather
        # than silently charging the run only for the small requested answer.
        allocated=max(max_tokens,self.minimum_output_tokens)
        payload=dict(contents=contents, generationConfig=dict(maxOutputTokens=allocated, temperature=temp))
        if system:payload['systemInstruction']=dict(parts=[dict(text='\n\n'.join(system))])
        self.budget.reserve(allocated)
        started = time.perf_counter()
        try:
            if self.transport:
                result = self.transport(self.model_id, payload)
            else:
                result = self._request(payload)
        except Exception:
            with self._lock:
                self.usage.append(dict(model=self.model_id,seconds=time.perf_counter()-started,
                    finish_reason='TRANSPORT_ERROR',requested_tokens=max_tokens,
                    allocated_tokens=allocated,usage=None))
            raise
        candidates=result.get('candidates') or []
        candidate=candidates[0] if candidates else {}
        with self._lock:
            self.usage.append(dict(model=self.model_id, seconds=time.perf_counter()-started,
                                   finish_reason=candidate.get('finishReason'),
                                   requested_tokens=max_tokens,allocated_tokens=allocated,
                                   usage=result.get('usageMetadata', {})))
        if candidate.get('finishReason') != 'STOP':
            reason=candidate.get('finishReason','NO_CANDIDATE')
            if reason not in ('MAX_TOKENS','SAFETY','RECITATION','OTHER','NO_CANDIDATE'):reason='OTHER'
            raise RuntimeError(f'Incomplete/blocked Gemini answer ({reason}); not passed to tolerant upstream parser')
        text = ''.join(p.get('text', '') for p in candidate.get('content', {}).get('parts', []) if not p.get('thought'))
        if not text.strip():raise RuntimeError('Empty Gemini answer')
        return text

    def _request(self, payload):
        from .runway_settings import CredentialStore
        key = self.key_loader() if self.key_loader else CredentialStore('gemini').load_for_api()
        req = urllib.request.Request('https://generativelanguage.googleapis.com/v1beta/models/'+self.model_id+':generateContent',
            data=json.dumps(payload).encode(), headers={'Content-Type':'application/json','x-goog-api-key':key}, method='POST')
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect).open(req, timeout=60) as response:
                raw=response.read(1048577)
            if len(raw)>1048576:raise ValueError('Oversized response')
            return json.loads(raw)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'Gemini HTTP {error.code}; no automatic network retry') from None
        except Exception:
            raise RuntimeError('Gemini transport failed; no automatic network retry') from None


class GeminiVLM(GeminiLLM):
    def ask(self, image, prompt, max_tokens=400):
        data=Path(image).read_bytes()
        if len(data)>3*2**20:raise ValueError('Preview exceeds 3 MiB')
        if data.startswith(b'\x89PNG\r\n\x1a\n'):mime='image/png'
        elif data.startswith(b'\xff\xd8\xff'):mime='image/jpeg'
        else:raise ValueError('PNG/JPEG preview required')
        part=dict(inlineData=dict(mimeType=mime, data=base64.b64encode(data).decode()))
        return self._call([dict(role='user', content=prompt)], max_tokens, 0, image=part)

    def looking_at(self, image):
        owner=self
        class BoundVision:
            model_id=owner.model_id
            def chat(self, messages, max_tokens=400, temp=0, prefill=''):
                prompt='\n\n'.join(m['role']+': '+m['content'] for m in messages)
                if prefill:prompt+='\nReturn complete answer beginning with: '+prefill
                return owner.ask(image, prompt, max_tokens)
        return BoundVision()
