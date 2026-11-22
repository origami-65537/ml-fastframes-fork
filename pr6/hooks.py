"""Memory logging around LingBot DiT graph captures and failures (PR 6 bench)."""
import torch


def mem(tag):
    gib = 2 ** 30
    print(f"MEM {tag}: allocated={torch.cuda.memory_allocated() / gib:.3f} GiB "
          f"reserved={torch.cuda.memory_reserved() / gib:.3f} GiB "
          f"max_allocated={torch.cuda.max_memory_allocated() / gib:.3f} GiB", flush=True)


def install():
    from apps.lingbot import dit
    original = dit.LingBotDiT._capture
    count = [0]

    def _capture(self, fn):
        count[0] += 1
        n = count[0]
        mem(f"before DiT capture #{n}")
        try:
            graph = original(self, fn)
        except BaseException:
            mem(f"at DiT capture #{n} failure")
            raise
        mem(f"after DiT capture #{n}")
        return graph

    dit.LingBotDiT._capture = _capture
