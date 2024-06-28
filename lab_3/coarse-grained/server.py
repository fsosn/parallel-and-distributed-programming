from multiprocessing.managers import BaseManager
import queue
import sys


class QueueManager(BaseManager):
    pass


def main(ip, port):
    in_queue = queue.Queue()
    out_queue = queue.Queue()
    QueueManager.register("in_queue", callable=lambda: in_queue)
    QueueManager.register("out_queue", callable=lambda: out_queue)
    manager = QueueManager(address=(ip, int(port)), authkey=b"password")
    server = manager.get_server()
    server.serve_forever()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python server.py <ip> <port>")
        sys.exit(1)
    main(*sys.argv[1:])
