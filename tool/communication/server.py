import multiprocessing
import time

def do_socket(conn, addr, ):
    try:

        while True:
            if conn.poll(1) == False:
                time.sleep(0.5)
                continue
            data = conn.recv()  # 等待接受数据
            conn.send('sucess')
            # ***********************
            # 要执行的程序写在这里
            # ***********************
            print(data)

    except Exception as e:
        print('Socket Error', e)

    finally:
        try:
            conn.close()
            print('Connection close.', addr)
        except:
            print('close except')


def run_server(host, port):
    from multiprocessing.connection import Listener
    server_sock = Listener((host, port))

    print("Sever running...", host, port)

    pool = multiprocessing.Pool(10)
    while True:
        # 接受一个新连接:

        conn = server_sock.accept()
        addr = server_sock.last_accepted
        print('Accept new connection', addr)

        # 创建进程来处理TCP连接:
        pool.apply_async(func=do_socket, args=(conn, addr,))


if __name__ == '__main__':
    server_host = '127.0.0.1'
    server_port = 8000
    run_server(server_host, server_port)
