from app.tasks import TrafficTasks, grade

tasks = TrafficTasks()

for level in ["easy", "medium", "hard"]:
    score = tasks.run_task(level)
    print(level, "raw score:", score)
    print(level, "graded:", grade(score))
    print("------")