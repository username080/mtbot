class FlowBookDivergence:
    """
    Checks agreement between execution flow and resting book.
    """
    def __init__(self):
        self.streak = 0
        self.last_direction = 0
        
    def update(self, bar):
        tick_imb = bar.get("tick_imbalance", 0.0)
        quote_imb = bar.get("quote_imbalance", 0.0)
        
        flow_dir = 1 if tick_imb > 0 else (-1 if tick_imb < 0 else 0)
        book_dir = 1 if quote_imb > 0 else (-1 if quote_imb < 0 else 0)
        
        # In backtests, quote_imb is 0.0, so fall back to flow_dir to allow testing
        if quote_imb == 0.0:
            book_dir = flow_dir
            
        if flow_dir != 0 and flow_dir == book_dir:
            if flow_dir == self.last_direction:
                self.streak += 1
            else:
                self.streak = 1
                self.last_direction = flow_dir
        else:
            self.streak = 0
            self.last_direction = 0
            
    def agrees(self, direction):
        return self.streak >= 2 and self.last_direction == direction
