from app.env import TrafficEnv
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler


# 🔥 simple health server
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")


def start_server():
    server = HTTPServer(("0.0.0.0", 7860), Handler)
    server.serve_forever()


def run_task(level: str):
    env = TrafficEnv()
    total_reward = 0

    state = env.reset()

    for _ in range(10):
        action = 1
        state, reward, done, _ = env.step(action)

        total_reward += getattr(reward, "value", reward)

        if done:
            break

    return total_reward


def main():
    print("🚦 Traffic OpenEnv running...")

    # 🔥 start health server in background
    threading.Thread(target=start_server, daemon=True).start()

    while True:
        for level in ["easy", "medium", "hard"]:
            score = run_task(level)
            print(f"{level} score: {score}")

        time.sleep(5)


if __name__ == "__main__":
    main()