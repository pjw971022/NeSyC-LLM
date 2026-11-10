"""One chat call against any OpenAI-compatible cloud endpoint.

Credentials are read from the environment only (OPENAI_API_KEY, OPENAI_BASE_URL) - never write a key into this repo.
Answers are cached on disk, so an interrupted run resumes and a finished run replays without spending quota.
"""
import hashlib
import json
import os

CACHE_PATH = os.environ.get('LLM_CACHE', './llm_cache.json')


def _request(prompt, model_name, temperature, max_tokens):
    from openai import OpenAI  # picks up OPENAI_API_KEY / OPENAI_BASE_URL; retries rate limits with backoff
    response = OpenAI(max_retries=8).chat.completions.create(
        model=model_name, temperature=temperature, max_tokens=max_tokens,
        messages=[{'role': 'user', 'content': prompt}])
    return response.choices[0].message.content or ''


def llm(prompt, model_name, temperature=0.0, max_tokens=256):
    key = hashlib.sha256(json.dumps([model_name, temperature, max_tokens, prompt]).encode()).hexdigest()
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            cache = json.load(f)
    if key not in cache:
        cache[key] = {'prompt': prompt, 'response': _request(prompt, model_name, temperature, max_tokens)}
        with open(CACHE_PATH + '.tmp', 'w') as f:  # lean: rewrites the whole file per new answer; sqlite past ~10k calls
            json.dump(cache, f, indent=1)
        os.replace(CACHE_PATH + '.tmp', CACHE_PATH)  # atomic, so an interrupt cannot truncate the cache
    return cache[key]['response']
