import time

class Profiler:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.total_samples = 0

    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        self.total_samples = 0

    def stop(self):
        """Stop the timer."""
        self.end_time = time.time()

    def update(self, batch_size):
        """Update the total number of samples processed."""
        self.total_samples += batch_size

    def throughput(self):
        """Calculate the throughput."""
        self.stop()
        
        if self.start_time is None or self.end_time is None:
            raise ValueError("Profiler has not been properly started or stopped.")
        
        elapsed_time = self.end_time - self.start_time
        if elapsed_time == 0:
            raise ValueError("Elapsed time is zero, cannot calculate throughput.")
        
        return self.total_samples / elapsed_time

    def reset(self):
        """Reset the profiler."""
        self.start_time = None
        self.end_time = None
        self.total_samples = 0


# 示例用法
if __name__ == "__main__":
    import random

    profiler = Profiler()
    
    # 模拟训练过程
    profiler.start()
    
    for _ in range(100):  # 假设有 100 个批次
        batch_size = random.randint(1, 100)  # 每个批次的样本数量随机
        time.sleep(random.uniform(0.01, 0.1))  # 模拟训练时间
        profiler.update(batch_size)
    
    profiler.stop()
    
    print(f"吞吐量: {profiler.throughput()} 样本/秒")
