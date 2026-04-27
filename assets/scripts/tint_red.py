import time


def load():
    return {
        "name": "Tint Red",
        "version": "1.0",
        "expected_input": "rgb8",
        "supports_streaming": True,
    }


def apply(rgb, w, h, emit_partial, should_cancel):
    buf = bytearray(rgb)
    stride = w * 3

    emit_every = max(1, h // 10)
    for y in range(h):
        if should_cancel():
            return bytes(buf)

        row_off = y * stride
        for x in range(w):
            i = row_off + x * 3
            r = buf[i] + 80
            buf[i] = r if r < 255 else 255

        if (y + 1) % emit_every == 0 or (y + 1) == h:
            emit_partial(bytes(buf), (y + 1) / h)
            time.sleep(0.04)

    return bytes(buf)


def unload():
    pass
