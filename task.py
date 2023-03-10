class Task:
    def __init__(self, start, end, cost, profit, dep=None) -> None:
        self.start = start
        self.end = end
        self.cost = cost
        self.profit = profit
        self.dep = dep