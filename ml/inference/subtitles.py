"""SRT / VTT / TXT export from timed segments [{'start': s, 'end': s, 'text': str}]."""


def _t(sec, comma):
    ms = int(round(sec * 1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{',' if comma else '.'}{ms:03d}"


def _check(segs):
    prev = 0.0
    for s in segs:
        if s["end"] <= s["start"] or s["start"] < prev - 1e-9:
            raise ValueError(f"invalid/overlapping segment: {s}")
        prev = s["end"]
    return segs


def to_srt(segs):
    return "".join(f"{i}\n{_t(s['start'], True)} --> {_t(s['end'], True)}\n{s['text']}\n\n"
                   for i, s in enumerate(_check(segs), 1))


def to_vtt(segs):
    return "WEBVTT\n\n" + "".join(f"{_t(s['start'], False)} --> {_t(s['end'], False)}\n{s['text']}\n\n" for s in _check(segs))


def to_txt(segs):
    return "\n".join(s["text"] for s in _check(segs)) + "\n"
