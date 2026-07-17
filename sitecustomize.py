import ssl

_original = ssl.create_default_context

def _patched(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
    try:
        return _original(purpose, cafile=cafile, capath=capath, cadata=cadata)
    except Exception:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

ssl.create_default_context = _patched
ssl._create_default_https_context = _patched
