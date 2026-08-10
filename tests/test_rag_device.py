from app.retrieval.rag_device import configure_cpu_threads, resolve_rag_device


class _FakeCuda:
    def __init__(self, *, available: bool, count: int) -> None:
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count


class _FakeTorch:
    def __init__(self, *, available: bool, count: int) -> None:
        self.cuda = _FakeCuda(available=available, count=count)
        self.thread_calls: list[int] = []

    def set_num_threads(self, value: int) -> None:
        self.thread_calls.append(value)


def test_explicit_cpu_device_is_respected(monkeypatch) -> None:
    monkeypatch.setenv("TEACHER_RAG_DEVICE", "cpu")
    fake_torch = _FakeTorch(available=True, count=8)

    assert resolve_rag_device(torch_module=fake_torch) == "cpu"


def test_cuda_zero_is_default_when_cuda_is_available(monkeypatch) -> None:
    monkeypatch.delenv("TEACHER_RAG_DEVICE", raising=False)
    fake_torch = _FakeTorch(available=True, count=8)

    assert resolve_rag_device(torch_module=fake_torch) == "cuda:0"


def test_requested_cuda_falls_back_when_cuda_is_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("TEACHER_RAG_DEVICE", "cuda:3")
    fake_torch = _FakeTorch(available=False, count=0)

    assert resolve_rag_device(torch_module=fake_torch) == "cpu"


def test_out_of_range_cuda_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setenv("TEACHER_RAG_DEVICE", "cuda:8")
    fake_torch = _FakeTorch(available=True, count=8)

    assert resolve_rag_device(torch_module=fake_torch) == "cpu"


def test_invalid_device_value_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setenv("TEACHER_RAG_DEVICE", "gpu")
    fake_torch = _FakeTorch(available=True, count=8)

    assert resolve_rag_device(torch_module=fake_torch) == "cpu"


def test_configure_cpu_threads_uses_environment(monkeypatch) -> None:
    monkeypatch.setenv("TEACHER_RAG_CPU_THREADS", "6")
    fake_torch = _FakeTorch(available=False, count=0)

    assert configure_cpu_threads(torch_module=fake_torch) == 6
    assert fake_torch.thread_calls == [6]


def test_invalid_cpu_thread_count_falls_back_to_four(monkeypatch) -> None:
    monkeypatch.setenv("TEACHER_RAG_CPU_THREADS", "not-a-number")
    fake_torch = _FakeTorch(available=False, count=0)

    assert configure_cpu_threads(torch_module=fake_torch) == 4
    assert fake_torch.thread_calls == [4]

