from communex.module.module import Module, endpoint
class SumNumbers(Module):
    def __init__(self, start: int = 5, end: int = 100) -> None:
        super().__init__()
        self.start = start
        self.end = end

    @endpoint
    def compute_sum(self) -> int:
        return sum(range(self.start, self.end + 1))

    @endpoint
    def get_metadata(self) -> dict:
        return {"start": self.start, "end": self.end}

if __name__ == "__main__":
    s = SumNumbers()
    total_sum = s.compute_sum()
    print(f"The sum of numbers from {s.start} to {s.end} is: {total_sum}")
    metadata = s.get_metadata()
    print(f"Metadata: {metadata}")
