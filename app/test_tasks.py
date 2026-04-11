from app.tasks import TrafficTasks


def run_demo():
    tasks = TrafficTasks()

    for level in ["easy", "medium", "hard"]:
        result = tasks.evaluate_task(level)
        print(level, "score:", result["score"])
        print(level, "adaptive reward:", result["adaptive_reward"])
        print(level, "fixed baseline:", result["baseline_reward"])
        print(level, "throughput:", result["adaptive_metrics"]["total_throughput"])
        print(level, "average queue:", result["adaptive_metrics"]["average_queue"])
        print("------")


if __name__ == "__main__":
    run_demo()
