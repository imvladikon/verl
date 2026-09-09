import pytest

from verl.workers.rollout.sglang_rollout.utils import merge_visible_devices


@pytest.mark.parametrize(
    ("workers", "expected"),
    [
        (["10,2", "1,2"], "1,2,10"),
        ([" 2, 0 ", "02,,0"], "0,2"),
        (["", " , "], ""),
        (["GPU-bbbb", "GPU-bbbb"], "GPU-bbbb"),
        (["GPU-bbbb,GPU-aaaa", " GPU-bbbb "], "GPU-bbbb,GPU-aaaa"),
        (["MIG-GPU-bbbb/1/0", "MIG-GPU-bbbb/2/0"], "MIG-GPU-bbbb/1/0,MIG-GPU-bbbb/2/0"),
        (["GPU-bbbb,2", "2,GPU-aaaa"], "GPU-bbbb,2,GPU-aaaa"),
    ],
)
def test_worker_devices_retain_identity_and_existing_ordinal_order(workers, expected):
    assert merge_visible_devices(workers) == expected
