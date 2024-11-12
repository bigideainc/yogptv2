from getpass import getpass

import paramiko


def test_ssh():
    host = "24.83.13.62"  # Use internal IP "192.168.10.130" if on the same network
    port = 11000
    user = "tang"
    password = "Yogptcommune1"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(f"Connecting to {host}:{port} as {user}")
        ssh.connect(host, port=port, username=user, password=password)
        print("Connected!")

        stdin, stdout, stderr = ssh.exec_command('nvidia-smi')
        print("Output:")
        print(stdout.read().decode())

        ssh.close()
        print("Connection closed.")
    except Exception as e:
        print(f"Error: {e}")

test_ssh()
