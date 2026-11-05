from starship.train.acceptance import stream_passes


def test_stream_passes_only_exact_salto_above_rate_floor():
    exact = {
        "rate/landed": 0.5,
        "landed/attitude/flips": 1.0,
        "landed/attitude/reflips": 0.0,
    }
    assert stream_passes(exact, 0.5)
    assert not stream_passes({**exact, "rate/landed": 0.499}, 0.5)
    assert not stream_passes({**exact, "landed/attitude/flips": 2.0}, 0.5)
    assert not stream_passes({**exact, "landed/attitude/reflips": 1.0}, 0.5)
    assert not stream_passes({}, 0.5)
