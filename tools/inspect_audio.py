import soundfile as sf
import sys

if len(sys.argv) < 2:
    print('usage: inspect_audio.py <path>')
    sys.exit(2)

p = sys.argv[1]
print('path:', p)
try:
    data, sr = sf.read(p)
    shape = getattr(data, 'shape', None)
    channels = 1 if data.ndim == 1 else data.shape[1]
    duration = None
    try:
        duration = len(data) / float(sr)
    except Exception:
        pass
    print('samplerate:', sr)
    print('shape:', shape)
    print('channels:', channels)
    print('duration_seconds:', duration)
except Exception as e:
    print('error:', e)
    import traceback; traceback.print_exc()
