"""One bounded owner-authorized SFX request. No video upload, retries or key logging.

Provider contract: https://elevenlabs.io/docs/api-reference/text-to-sound-effects/convert
This is text-conditioned sound, NOT visually synchronized audio or Gaussian motion.
"""
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import time

import requests

from .runway_settings import CredentialStore

ENDPOINT = 'https://api.elevenlabs.io/v1/sound-generation'
MODEL = 'eleven_text_to_sound_v2'
MAX_BYTES = 256 * 1024
# Reserve MP3 encoder padding inside the five-second decoded-container bound.
MAX_REQUEST_SECONDS = 4.8
MAX_AUDIO_SECONDS = 5.0
REQUEST_DEADLINE_SECONDS = 75


class SoundError(RuntimeError):
    """Only fixed public codes may cross the HTTP/job boundary."""
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def mp3_duration(data):
    """Bounded Layer III frame scan; no external process or untrusted decoder.

    The fixed MP3 output is validated frame by frame. Duration includes encoder
    delay/padding, so this intentionally overbounds audible duration.
    """
    if not isinstance(data, bytes) or not 24 <= len(data) <= MAX_BYTES:
        raise SoundError('invalid_audio')
    offset = 0
    if data[:3] == b'ID3':
        if len(data) < 10 or any(x & 128 for x in data[6:10]):
            raise SoundError('invalid_audio')
        tag_size = sum(data[6+i] << (21-7*i) for i in range(4))
        offset = 10 + tag_size + (10 if data[5] & 16 else 0)
    duration = 0.0
    frames = 0
    bitrates = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)
    while offset < len(data):
        if len(data)-offset == 128 and data[offset:offset+3] == b'TAG':
            offset += 128
            break
        if len(data)-offset < 4:
            raise SoundError('invalid_audio')
        header = int.from_bytes(data[offset:offset+4], 'big')
        # Fixed endpoint output must be MPEG-1, Layer III, 44.1kHz.
        version, layer = (header >> 19) & 3, (header >> 17) & 3
        rate_index, sample_index = (header >> 12) & 15, (header >> 10) & 3
        if (header >> 21 != 0x7ff or version != 3 or layer != 1
                or not 1 <= rate_index <= 14 or sample_index != 0):
            raise SoundError('invalid_audio')
        frame_length = (144000 * bitrates[rate_index]) // 44100 + ((header >> 9) & 1)
        if offset + frame_length > len(data):
            raise SoundError('invalid_audio')
        duration += 1152 / 44100
        frames += 1
        offset += frame_length
        if duration > MAX_AUDIO_SECONDS:
            raise SoundError('audio_duration_limit')
    if not frames or duration < 0.4:
        raise SoundError('invalid_audio')
    return duration


def validate_sound_request(prompt, output, duration_seconds):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000:
        raise ValueError('Sound prompt must contain 1–2000 characters')
    if (type(duration_seconds) not in (float, int) or not math.isfinite(duration_seconds)
            or not 0.5 <= duration_seconds <= MAX_REQUEST_SECONDS):
        raise ValueError('Sound duration exceeds bounded request')
    output = Path(output)
    if output.exists():
        raise ValueError('Sound output already exists; no automatic retry')
    return output


def _sound_worker(prompt, output, duration_seconds, channel):
    """Child-only network and disk transaction; no key crosses this IPC pipe."""
    try:
        result = _generate_sound_request(prompt, output, duration_seconds=duration_seconds)
        payload = {'result':result}
    except SoundError as error:
        payload = {'error':error.code}
    except Exception:
        payload = {'error':'sound_worker_failed'}
    try:
        channel.send_bytes(json.dumps(payload).encode('utf-8'))
    finally:
        channel.close()


def _supervised_sound(prompt, output, duration_seconds, *, context=None):
    """Bound wall time even if a server drips headers/chunk framing forever.

    Requests' socket timeouts bound inactivity, not wall-clock time. A dedicated
    child process can be terminated without leaving an in-flight Python thread.
    A timeout may occur after provider acceptance: never automatically retry it.
    """
    context = context or multiprocessing.get_context('spawn')
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(target=_sound_worker,args=(prompt,str(output),duration_seconds,sender),daemon=True)
    started = False
    try:
        worker.start(); started = True
        sender.close()
        if not receiver.poll(REQUEST_DEADLINE_SECONDS):
            raise SoundError('request_deadline_ambiguous')
        try:
            payload = json.loads(receiver.recv_bytes(16384))
        except (EOFError,OSError,ValueError,UnicodeError):
            raise SoundError('sound_worker_failed') from None
        if not isinstance(payload,dict):
            raise SoundError('sound_worker_failed')
        if 'error' in payload:
            # These values originate from our child, never raw provider content.
            raise SoundError(payload['error'])
        if not isinstance(payload.get('result'),dict):
            raise SoundError('sound_worker_failed')
        return payload['result']
    finally:
        sender.close(); receiver.close()
        if started:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)
            if worker.is_alive():
                worker.kill(); worker.join(timeout=2)
            worker.close()


def generate_sound(prompt, output, *, duration_seconds=MAX_REQUEST_SECONDS, session=None, store=None):
    output = validate_sound_request(prompt, output, duration_seconds)
    if session is not None or store is not None:
        # Explicit dependency injection is for offline tests. The Studio does not
        # expose this option: production always uses the supervised child.
        return _generate_sound_request(prompt, output,duration_seconds=duration_seconds,session=session,store=store)
    if not CredentialStore('elevenlabs').configured():
        raise SoundError('credential_missing')
    return _supervised_sound(prompt, output, duration_seconds)


def _generate_sound_request(prompt, output, *, duration_seconds=MAX_REQUEST_SECONDS, session=None, store=None):
    output = validate_sound_request(prompt, output, duration_seconds)
    store = store or CredentialStore('elevenlabs')
    if not store.configured():
        raise SoundError('credential_missing')
    try:
        key = store.load_for_api()
    except Exception:
        raise SoundError('credential_unavailable') from None
    owned = session is None
    session = session or requests.Session()
    session.trust_env = False
    response = None
    started = time.monotonic()
    body = dict(text=prompt.strip(), model_id=MODEL, duration_seconds=float(duration_seconds),
                loop=False, prompt_influence=0.3)
    try:
        # Requests default adapter retries=0. Redirects cannot forward credentials.
        response = session.post(ENDPOINT, params={'output_format':'mp3_44100_128'},
            headers={'xi-api-key':key, 'Content-Type':'application/json'}, json=body,
            timeout=(10, 60), allow_redirects=False, stream=True)
        key = None
        if response.status_code != 200:
            codes = {401:'auth_failed',403:'permission_denied',402:'credit_required',429:'rate_limited'}
            raise SoundError(codes.get(response.status_code, 'provider_rejected'))
        if response.headers.get('Content-Type','').split(';',1)[0].strip().lower() not in ('audio/mpeg','audio/mp3','application/octet-stream'):
            raise SoundError('invalid_audio_type')
        declared = response.headers.get('Content-Length')
        if declared is not None and (not declared.isdigit() or not 0 < int(declared) <= MAX_BYTES):
            raise SoundError('audio_byte_limit')
        chunks = []
        size = 0
        for block in response.iter_content(chunk_size=8192):
            if time.monotonic() - started > 75:
                raise SoundError('request_timeout')
            size += len(block)
            if size > MAX_BYTES:
                raise SoundError('audio_byte_limit')
            chunks.append(block)
        audio = b''.join(chunks)
        measured_duration = mp3_duration(audio)
        billed = response.headers.get('character-cost','')
        billing_characters = int(billed) if billed.isdigit() and len(billed) <= 10 else None
    except requests.RequestException:
        # Never persist RequestException text: it may contain URLs or headers.
        raise SoundError('transport_failed') from None
    finally:
        key = None
        if response is not None:
            response.close()
        if owned:
            session.close()
    metadata = dict(provider='elevenlabs',model=MODEL,output='sound.mp3',mime='audio/mpeg',
        requested_duration_seconds=float(duration_seconds),container_duration_seconds=measured_duration,
        bytes=len(audio),sha256=hashlib.sha256(audio).hexdigest(),
        api_seconds=time.monotonic()-started,requests=1,billing_characters=billing_characters,
        cost_usd=None,video_modified=False,gaussians_modified=False,synchronized_to_video=False,
        review='Unreviewed audio; no verified audiovisual synchronization',
        source='https://elevenlabs.io/docs/api-reference/text-to-sound-effects/convert')
    output.mkdir(parents=True, exist_ok=False)
    temporary = output/'sound.mp3.tmp'
    try:
        temporary.write_bytes(audio)
        os.replace(temporary, output/'sound.mp3')
        pending = output/'sound.json.tmp'
        pending.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        os.replace(pending, output/'sound.json')
    except OSError:
        raise SoundError('artifact_write_failed') from None
    return metadata
